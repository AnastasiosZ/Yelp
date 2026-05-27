import numpy as np
import pandas as pd
import os
import warnings
from functools import reduce
from utility_functions import *

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, Normalizer, Word2VecModel
from pyspark.ml.stat import Summarizer

warnings.filterwarnings('ignore')


# Read clean parquet files
read_parquet = lambda fn: spark.read.parquet(f'{DATA_PATH}{fn}.parquet')

### Hadoop/Spark configuration
os.environ['HADOOP_HOME'] = 'C:/hadoop'
os.environ['PATH'] = 'C:/hadoop/bin' + os.pathsep + os.environ['PATH']


spark = (
    SparkSession.builder
    .appName("DSC511_recommender") # type:ignore
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel('ERROR')

# Load Word2Vec model and its vocabulary
W2V_MODEL = Word2VecModel.load(W2V_PATH)
W2V_VOCAB = {r.word for r in W2V_MODEL.getVectors().collect()}



# Load, but don't persist
BUSINESSES_DF = read_parquet('business')


BUSINESS_PIPE = PipelineModel(stages=[
    MeanImputeFallback(),
    VectorAssembler(inputCols=list(ATTRIBUTES.keys()), outputCol='_'),
    Normalizer(inputCol='_', outputCol='bvec')
])


def _save_df(df, cols=None):

    if cols is None: cols=df.columns

    with open('temp.txt', 'w') as f:
        f.write(df.limit(100).select(cols).toPandas().to_string())



class User():

    def __init__(self) -> None:

        # Read user files
        self.user_profile = spark.read.json(f'{USER_PATH}user_profile.json')
        user_reviews = spark.read.json(f'{USER_PATH}user_reviews.json')

        # Aggregate multiple reviews by last star rating
        user_reviews = user_reviews.sort('date').groupBy('business_id').agg(F.last('stars').alias('user_stars'))

        # Join with businesses (shared cached DataFrame)        
        self.user_businesses = BUSINESSES_DF.join(user_reviews, 'business_id').persist()





class Recommendation:
    
    def __init__(self, user: User, query:dict) -> None:

        # Instead of inherit, access User instance
        self.user = user

        self.query = query

        ### Categories filter
        # query['scope'] determines the number of category term synonyms to union with the original
        matched_categories = similar_categories_w2v(W2V_MODEL, W2V_VOCAB, 
                                                    query['categories'], topn = query['scope'])
        
        businesses = BUSINESSES_DF.filter(reduce(lambda a, b: a | b,
           [F.array_contains('categories',c) for c in matched_categories]))


        # Pass through business pipe early on to capture as much signal
        # as possible for imputation
        businesses = BUSINESS_PIPE.transform(businesses)

        # Locality filter and open business filter
        self.businesses = self.locality_filter(businesses)\
                         .filter(F.col('is_open')==True).persist() # type: ignore
        
        # These are the potential businesses to recommend to the user
        # which satisfy the base structure of the query


        # Filter out irrelevant businesses from the user review history
        user_businesses = self.user.user_businesses.filter(reduce(lambda a, b: a | b,
           [F.array_contains('categories',c) for c in matched_categories]))

        # Construct user preference vector
        self.user_vec = BUSINESS_PIPE.transform(user_businesses)\
                        .agg(Summarizer.mean(F.col('bvec'), weightCol=F.col('user_stars')))\
                        .take(1)[0][0].toArray()



    def locality_filter(self, bdf):
        
        if len(self.query['locality']) < 2: 
            raise Exception('Invalid locality filter')
        
        self.locality_mode = self.query['locality'][0].lower()
        
        match self.locality_mode:
            case 'state': return bdf.filter(F.col('state')==self.query['locality'][1])
            case 'city': return bdf.filter(F.col('city')==self.query['locality'][1])
            case 'radial':
                if not isinstance(self.query['locality'][1],tuple):
                    raise Exception('Invalid latitude/longitude format')
                
                if len(self.query['locality'])==2:
                    raise Exception('Missing maximum radial distance')
                
                lat,long = self.query['locality'][1]
                max_dist = self.query['locality'][2]

                return bdf.filter(haversine_spark(F.col('latitude'), F.col('longitude'),
                                                  F.lit(lat), F.lit(long))<=max_dist)


    def currently_open_filter(self, bdf):
        
        day = self.query['datetime']['day'].title()
        previous_day = DAYS[DAYS.index(day)-1]

        mins = int(self.query['datetime']['time'])

        open_mins, close_mins = [f'{day}_{_}_mins' for _ in ['open','close']]
        previous_day_open_mins, previous_day_close_mins = [f'{previous_day}_{_}_mins' for _ in ['open','close']]

        ### Filter businesses:
        # Opened and closed within query day
        # Opened in query day and close next day
        # Opened one day before query day and still open

        return bdf.filter(
            ((F.col(open_mins) <= mins) &
            ((F.col(close_mins) > mins) | (F.col(close_mins) <= F.col(open_mins)))) |
            ((F.col(previous_day_close_mins) <= F.col(previous_day_open_mins)) &
            (F.col(previous_day_close_mins) > mins))
        )


    def drop_reviewed(self, bdf):
        return bdf.join(self.user.user_businesses.select('business_id'),
                        on='business_id', how='left_anti')




    def recommend(self, n=5, currently_open=True, recommend_reviewed=False,
                   trip_duration=False, save_to_file=True, detail=False):
        
        # Acquired potential businesses to recommend from base query structure
        # Now we apply the auxiliary filters

        businesses = self.businesses

        # Currently open filter
        if currently_open: businesses = self.currently_open_filter(businesses)

        # Filter out already reviewed businesses
        if not recommend_reviewed: businesses = self.drop_reviewed(businesses)

        if businesses.limit(1).count() == 0:
            print('No businesses match the query filters.')
            return None

        # Collect as Pandas DataFrame, self.businesses unaffected
        businesses = businesses.toPandas()

        # Cosine similarity (vectorized: stack all L2-normalized vectors and do one matmul)
        B = np.stack(businesses['bvec'].apply(lambda v: v.toArray()).values)
        businesses['cosine_similarity'] = B @ np.diag(ATTRIBUTE_IMPORTANCE) @ self.user_vec


        # Binning cosine similarity in (number of businesses)^(1/2) number of bins
        businesses['similarity_bin'] = pd.qcut(businesses['cosine_similarity'],
                                               int(np.power(len(businesses),1/2)),
                                               duplicates='drop', labels=False)

        # First sorting by similarity bin and then by quality metric
        businesses = businesses.sort_values(['similarity_bin', 'quality'], ascending=False).head(n)

        # Optionally get expected trip duration
        if trip_duration and self.locality_mode=='radial':
            user_lat,user_long = self.query['locality'][1]

            businesses['trip_duration_s'] = businesses.apply(
                lambda row: get_travel_duration_ors(user_lat, user_long,
                            row['latitude'], row['longitude']), axis=1)

            businesses['trip_duration'] = businesses['trip_duration_s'].apply(
                lambda duration: f'{int(duration//60)}m {int(duration%60)}s' if duration else 'N/A')

        # Index the informative columns, depending on the level of detail required
        output_feats = ['name','address','city','stars', 'review_count']
        if detail: output_feats = output_feats + ['categories']

        # Also index trip duration if applicable
        if trip_duration: output_feats = output_feats + ['trip_duration']

        businesses = businesses[output_feats]

        # Print recommended businesses
        print(businesses.to_string(index=False))

        # Optionally additinally save to a .txt file
        if save_to_file:
            with open('recommendations.txt', 'w', encoding='utf-8') as f:
                f.write(businesses.to_string(index=False))





if __name__ == "__main__":

    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')


    USER_LAT, USER_LONG = 27.948375982565096, -82.4648864239378

    query = {'locality':('radial', (USER_LAT, USER_LONG), 3), # 3km maximum distance 
             'categories': ['Nightlife', 'Bars'], # query categories
             'datetime':{'day':'Friday', # query day
                         'time':21*60}, # query time in seconds
             'scope': 2} # categories scope of the query

    user = User()
    rec = Recommendation(user, query)

    rec.recommend(10, trip_duration=False, detail=True)


    print('Stopping SparkSession...')
    spark.stop() # Stop Spark when done

