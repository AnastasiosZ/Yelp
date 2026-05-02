import numpy as np
import pandas as pd
import os
import warnings
from functools import reduce
from utility_functions import *

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer,\
                               Word2VecModel, Imputer
from pyspark.ml.stat import Summarizer

warnings.filterwarnings('ignore')

CLEAN_PATH = 'data/clean/'
USER_PATH = 'data/example/'

# Read clean parquet files
read_parquet = lambda fn: spark.read.parquet(f'{CLEAN_PATH}{fn}.parquet')



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


# Cached once per process: reused by every User/Recommendation instance.
BUSINESSES_DF = read_parquet('business').cache()

# Load Word2Vec model once per process (Flask imports and reuses this).
W2V_MODEL = Word2VecModel.load(W2V_PATH)


class User():
    """
    Represents the query user and their review history.

    Loads the user profile and JSONL reviews from 'data/example/',
    collapses repeat (user, business) reviews to their mean star rating,
    joins against the cached business table, builds the shared
    ``business_pipe`` (DefaultValueImputer → Imputer(mode) →
    VectorAssembler → Normalizer(L2)), and applies it to produce the
    user's reviewed-business vectors. The user preference vector
    (``user_bvector``) is computed once here as the ``avg(stars)``-
    weighted mean of those L2-normalized business vectors, so every
    subsequent ``Recommendation.recommend`` call scores against a
    pre-computed NumPy array.

    Attributes
    ----------
    user_profile : pyspark.sql.DataFrame
        Single-row profile DataFrame from user_profile.json.
    user_businesses : pyspark.sql.DataFrame
        DataFrame of businesses the user has reviewed, with an
        ``avg(stars)`` column and the ``bvector_L2`` attribute vector.
    business_pipe : pyspark.ml.Pipeline
        The unfitted attribute-vector pipeline, refit per DataFrame so
        the imputation modes reflect each demographic.
    user_bvector : numpy.ndarray
        Weighted-mean L2 vector used as the user's preference vector.
    """

    def __init__(self) -> None:
        """
        Load the user profile and reviews, aggregate duplicates, join to
        the shared cached ``BUSINESSES_DF``, build the business pipeline,
        and pre-compute the user preference vector.
        """

        # Read user files
        self.user_profile = spark.read.json(f'{USER_PATH}user_profile.json')
        user_reviews = spark.read.json(f'{USER_PATH}user_reviews.json')

        # Aggregate multiple reviews by mean star rating
        user_reviews = user_reviews.groupBy('business_id').agg(F.avg('stars'))

        # Join with businesses (shared cached DataFrame)
        user_reviews = user_reviews.join(BUSINESSES_DF, 'business_id')

        # Define the pipe that businesses pass through to get their vector representation
        self._define_business_pipe()

        # Get vector representation of the businesses the user reviewed
        self.user_businesses = self.business_vectors(user_reviews)

        # User reviewed businesses pass through the SAME fitted pipe, then weighted-averaged
        self.user_bvector = self.user_businesses.agg(Summarizer.mean(F.col('bvector_L2'),
                                                     weightCol=F.col('avg(stars)'))\
                                                .alias('weighted_mean')).collect()[0][0]\
                                                .toArray()
        
    def _define_business_pipe(self) -> None:
        """
        Construct the shared attribute-vector pipeline.

        Stages:
          1. ``DefaultValueImputer`` — fills any column that is entirely
             NULL for the current DataFrame with a global default (the
             dataset-wide mode), so that stage 2 never sees an all-null
             column.
          2. ``Imputer(strategy='mode')`` — mode-imputes the remaining
             per-column NULLs. Writes to ``<col>_imp``.
          3. ``VectorAssembler`` — concatenates the imputed numeric
             columns into a single ``bvector`` column.
          4. ``Normalizer(L2)`` — L2-normalizes to ``bvector_L2`` so
             downstream cosine similarity reduces to a dot product.

        The pipeline is refit per DataFrame (see ``business_vectors``)
        so the imputed modes reflect each demographic — i.e. the user's
        reviewed businesses and the candidate pool each contribute their
        own modes rather than sharing a global one.
        """

        attributes_cols_imp = [col+'_imp' for col in attributes_cols]

        default_imputer = DefaultValueImputer()
        modeimp = Imputer(inputCols=attributes_cols, outputCols=attributes_cols_imp,
                          strategy='mode')
        vass = VectorAssembler(inputCols=attributes_cols_imp, outputCol='bvector')
        L2norm = Normalizer(inputCol='bvector', outputCol='bvector_L2')

        self.business_pipe = Pipeline(stages=[default_imputer, modeimp, vass, L2norm])


    def business_vectors(self, bdf):
        """
        Fit ``self.business_pipe`` on ``bdf`` and transform it.

        Refitting per DataFrame is intentional: it lets the Imputer's
        mode reflect the specific demographic (the user's reviewed
        businesses, or the filtered candidate pool) rather than
        borrowing a global mode from the full business table.

        Parameters
        ----------
        bdf : pyspark.sql.DataFrame
            Business DataFrame containing at least ``attributes_cols``.

        Returns
        -------
        pyspark.sql.DataFrame
            ``bdf`` with added ``*_imp``, ``bvector`` and ``bvector_L2``
            columns.
        """
        return self.business_pipe.fit(bdf).transform(bdf)




class Recommendation(User):
    """
    Content-based business recommender built on top of User.

    At construction time applies the top-level query filters (locality,
    is_open=True, and category expansion via Word2Vec) to produce
    self.businesses — the candidate pool. recommend() then applies
    the optional per-call filters (currently-open, drop-reviewed),
    refits the shared ``business_pipe`` on the filtered pool (so the
    mode imputation reflects the candidates' own demographic), scores
    each candidate by cosine similarity against the pre-computed
    ``user_bvector``, and returns the top-n ranked by
    (similarity bin, quality).

    Parameters
    ----------
    query : dict
        Must contain locality (tuple — see locality_filter),
        categories (str or list[str]), and
        datetime (dict with day and time in minutes).

    Attributes
    ----------
    w2v_model : pyspark.ml.feature.Word2VecModel
        Shared process-wide category embedding model.
    query : dict
        The raw query dict, stored for later stages.
    locality_mode : {'state', 'city', 'radial'}
        Set by locality_filter; drives the trip-duration branch (only for radial).
    businesses : pyspark.sql.DataFrame
        Candidate pool after top-level filters.

    Raises
    ------
    ValueError
        If no query category can be found in the Word2Vec vocabulary.
    """

    def __init__(self, query:dict) -> None:
        """
        Apply top-level filters (locality → is_open → categories) to build
        the candidate business pool.

        Parameters
        ----------
        query : dict
            Query spec — see the class docstring for the expected shape.
        """
        super().__init__()

        # Shared module-level Word2Vec model (loaded once per process)
        self.w2v_model = W2V_MODEL

        self.query = query

        # Locality filter and open business filter (reuse cached businesses)
        businesses = self.locality_filter(BUSINESSES_DF).filter(F.col('is_open')==True) #type:ignore

        ### Categories filter
        # self.scope determines the number of category term synonyms to union with the original
        self.scope = query['scope']

        matched_categories = similar_categories_w2v(query['categories'], model=self.w2v_model,
                                                    topn = self.scope)
        if not matched_categories:
            raise ValueError("No matching categories found in Word2Vec vocabulary for query categories")

        self.businesses = businesses.filter(reduce(lambda a, b: a | b,
           [F.array_contains('categories',c) for c in matched_categories]))
        



    def locality_filter(self, bdf):
        """
        Restrict bdf to businesses matching the query's locality clause.

        Dispatches on query['locality'][0]:

        - 'state' — equality match on the state column.
        - 'city' — equality match on the city column.
        - 'radial' — expects query['locality'] = ('radial', (lat, lng), km)
          and keeps rows whose great-circle distance (haversine_spark)
          from the user is within the given radius.

        Also sets self.locality_mode as a side-effect so later stages
        (e.g. trip-duration, which is only valid in radial mode) can
        branch on it.

        Parameters
        ----------
        bdf : pyspark.sql.DataFrame
            Business DataFrame with state, city, latitude and
            longitude columns.

        Returns
        -------
        pyspark.sql.DataFrame
            Filtered business DataFrame.

        Raises
        ------
        Exception
            On malformed or incomplete locality tuples.
        """

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
        """
        Keep only businesses open at the query's day and time (minutes).

        Parses the hours.<Day> string ("HH:MM-HH:MM") with a regex
        into open/close hour/minute columns, then keeps rows where either:

        - the business is 24-hour (close time 00:00 with a non-zero
          open hour — the dataset's convention), or
        - open ≤ query_time < close (in total minutes).

        Parameters
        ----------
        bdf : pyspark.sql.DataFrame
            Business DataFrame with the ``hours`` struct column.

        Returns
        -------
        pyspark.sql.DataFrame
            Filtered business DataFrame.
        """
        
        day = self.query['datetime']['day']
        mins = self.query['datetime']['time']

        reg_pattern = r'(\d+):(\d+)-(\d+):(\d+)'

        bdf = bdf\
            .withColumns({'open_h':F.regexp_extract(col(f'hours.{day}'), reg_pattern, 1).cast(IntegerType()),
                          'open_m':F.regexp_extract(col(f'hours.{day}'), reg_pattern, 2).cast(IntegerType()),
                          'close_h':F.regexp_extract(col(f'hours.{day}'), reg_pattern, 3).cast(IntegerType()),
                          'close_m':F.regexp_extract(col(f'hours.{day}'), reg_pattern, 4).cast(IntegerType())})\
            .filter(((col('close_h')==0) & (col('close_m')==0) & (col('open_h')!=0)) |
                    ((col('open_h')*60 + col('open_m') <= mins) &
                    (col('close_h')*60 + col('close_m') > mins)))

        return bdf.drop('open_h', 'open_m', 'close_h', 'close_m')
    

    def drop_reviewed(self, bdf):
        """
        Remove businesses the user has already reviewed from bdf.

        Implemented as a left-anti join on business_id against
        self.user_businesses, so candidates the user has rated are
        excluded from the recommendation pool.

        Parameters
        ----------
        bdf : pyspark.sql.DataFrame
            Candidate business DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            bdf minus the user's already-reviewed businesses.
        """

        return bdf.join(self.user_businesses.select('business_id'),
                        on='business_id', how='left_anti')




    def recommend(self, n=5, currently_open=True, recommend_reviewed=False,
                   trip_duration=False, save_to_file=True, detail=False):
        """
        Produce and display the top-n business recommendations for the user.

        Ranks candidate businesses (already constrained by locality, is_open and
        category filters from __init__) by cosine similarity between their L2-
        normalized attribute vectors and the pre-computed ``user_bvector``,
        then breaks ties within similarity bins using the cached ``quality``
        metric (``log10(review_count+1) * stars²``, materialized in
        ``clean_data.ipynb``).

        Parameters
        ----------
        n : int, default 5
            Number of recommendations to return.
        currently_open : bool, default True
            If True, filter out businesses closed at the query day/time.
        recommend_reviewed : bool, default False
            If False, exclude businesses the user has already reviewed.
        trip_duration : bool, default False
            If True, query OpenRouteService for driving duration from the user
            to each recommended business. Only valid when locality_mode ==
            'radial'.
        save_to_file : bool, default True
            If True, save formatted results to 'recommendations.txt'.
        detail : bool, default False
            If True, include the 'categories' column in the output.

        Side-effects
        ------------
        Prints the formatted recommendation table to stdout. Optionally writes
        the same table to 'recommendations.txt' in the current working
        directory.

        Returns
        -------
        None
        """

        # Start each new recommendation fresh (locality, is_open, categories filters applied)
        businesses = self.businesses

        # Currently open filter
        if currently_open: businesses = self.currently_open_filter(businesses)

        # Filter out already reviewed businesses
        if not recommend_reviewed: businesses = self.drop_reviewed(businesses)

        if businesses.limit(1).count() == 0:
            print('No businesses match the current filters.')
            return

        businesses = self.business_vectors(businesses).toPandas()

        # Cosine similarity
        businesses['cosine_similarity'] = businesses['bvector_L2']\
                                         .map(lambda b: b.toArray() @ self.user_bvector) # type:ignore

        # Binning cosine similarity in (number of businesses)^(1/3) number of bins, drawing
        # inspiration from the Rice formula for the exponent
        businesses['similarity_bin'] = pd.qcut(businesses['cosine_similarity'],
                                               int(np.power(len(businesses),1/3)),
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
        output_feats = ['name','address','city','review_count', 'stars']
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
             'scope': 5} # categories scope of the query

    rec = Recommendation(query) 
    
    rec.recommend(10, trip_duration=False)


    print('Stopping SparkSession...')
    spark.stop() # Stop Spark when done

