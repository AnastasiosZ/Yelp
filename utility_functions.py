from pyspark.sql.functions import col, sin, cos, asin, sqrt, radians
from dotenv import load_dotenv
import os
import requests
import numpy as np

from pyspark.ml import Transformer


# --- Constants ---
W2V_PATH = 'models/w2v_categories'


# Business attribute columns used to build the profile vector.
# All numeric by construction: the boolean attributes were cast to ByteType
# as ``<name>_num`` columns in clean_data.ipynb, and the string attributes
# (NoiseLevel, RestaurantsAttire) were pre-indexed with StringIndexer so
# they can be fed directly to pyspark.ml.feature.Imputer (which only
# accepts numeric types).
attributes_cols = ['DogsAllowed_num','GoodForKids_num','OutdoorSeating_num',
                   'RestaurantsGoodForGroups_num','WheelchairAccessible_num',
                   'RestaurantsPriceRange2','NoiseLevel_indexed', 'RestaurantsAttire_indexed']



from pyspark.ml import Transformer
from pyspark.sql import functions as F

class DefaultValueImputer(Transformer):
    """
    Custom Spark ML ``Transformer`` that plugs the ``Imputer`` gap when a
    column is entirely NULL for the current DataFrame.

    ``pyspark.ml.feature.Imputer(strategy='mode')`` cannot compute a mode
    for a column with zero non-null values — it raises at fit time. This
    transformer runs first in ``business_pipe``: it counts non-nulls for
    every ``attributes_cols`` column in a single pass, and for any column
    whose count is zero, fills it with a global default (the dataset-wide
    mode pre-computed on the full business table in EDA). After this
    stage, every column has at least one non-null value and the
    downstream ``Imputer`` can safely fit its per-column mode on the
    remaining (non-default) rows.

    The dataset-wide modes are aligned by position with ``attributes_cols``.
    """

    def __init__(self):

        default_values = [0,1,0,1,1,2,0,0]
        self.fill_dict = dict(zip(attributes_cols, default_values))

    def _transform(self, df): # type:ignore
        """
        Fill all-NULL attribute columns with their dataset-wide defaults.

        Detects all-null columns in a single aggregation pass (one
        ``F.count`` per column, evaluated together), then narrows the
        ``fillna`` call to just those columns.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Business DataFrame containing ``attributes_cols``.

        Returns
        -------
        pyspark.sql.DataFrame
            Same schema as ``df`` with all-null attribute columns
            defaulted.
        """

        counts = df.select([F.count(c).alias(c) for c in self.fill_dict]).first().asDict()

        null_only_dict = {c: self.fill_dict[c] for c,v in counts.items() if v == 0}

        return df.fillna(null_only_dict)


def haversine_spark(lat1_col, lon1_col, lat2_col, lon2_col):
    """
    Great-circle distance between two lat/long points as a Spark Column.

    Implemented purely with Spark SQL primitives (no Python UDF) so the
    expression can be pushed down and stays vectorized across the cluster.
    Uses an Earth radius of 6371 km, so the returned column is in
    kilometers.

    Parameters
    ----------
    lat1_col, lon1_col : pyspark.sql.Column
        Latitude/longitude of the first point, in degrees.
    lat2_col, lon2_col : pyspark.sql.Column
        Latitude/longitude of the second point, in degrees.

    Returns
    -------
    pyspark.sql.Column
        Column of distances in kilometers.
    """
    R = 6371.0

    dlat = radians(lat2_col - lat1_col)
    dlon = radians(lon2_col - lon1_col)

    a = sin(dlat/2)**2 + cos(radians(lat1_col)) * cos(radians(lat2_col)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    return R * c




def get_travel_duration_ors(origin_lat, origin_lng, dest_lat, dest_lng):
    """
    Query OpenRouteService for driving duration between two coordinates.

    Reads ORS_API_KEY from the process environment (populated from
    .env via python-dotenv). Uses the driving-car profile and
    returns the route's total duration in seconds. Any non-2xx response
    (including rate-limit errors, which are common on the free tier) is
    swallowed and None is returned so the caller can substitute
    'N/A' in the UI.

    Parameters
    ----------
    origin_lat, origin_lng : float
        Starting point (the user's location) in degrees.
    dest_lat, dest_lng : float
        Destination business coordinates in degrees.

    Returns
    -------
    float or None
        Driving duration in seconds, or ``None`` if the API call failed.
    """

    # We stored the API key in the .env folder and we access it here
    load_dotenv() 

    api_key = os.getenv('ORS_API_KEY')
    url = 'https://api.openrouteservice.org/v2/directions/driving-car'
    headers = {'Authorization': api_key}
    params = {
        'start': f'{origin_lng},{origin_lat}',
        'end': f'{dest_lng},{dest_lat}',
    }
    response = requests.get(url, headers=headers, params=params)
    if not response.ok:
        return None
    return response.json()['features'][0]['properties']['summary']['duration']


def similar_categories_w2v(category_filter, topn=5, model=None):
    """
    Find categories semantically similar to the input category/categories
    using a trained Word2Vec model over the business category vocabulary.

    Parameters:
        category_filter (str or list of str): single category or list of
            categories to look up in the Word2Vec vocabulary.
        topn (int): number of similar categories to retrieve per input
            category (default 5).
        model (Word2VecModel, optional): a pre-loaded Word2VecModel. If
            None, the model is loaded from W2V_PATH.

    Returns:
        list of str: the input category/categories merged with their
            top-n most similar categories from the Word2Vec vocabulary.

    Raises:
        Exception: if a category is not present in the Word2Vec vocabulary.
    """
    from pyspark.ml.feature import Word2VecModel

    if topn==0: return category_filter

    if isinstance(category_filter, list):
        if model is None:
            model = Word2VecModel.load(W2V_PATH)
        results = set()
        for cat in category_filter:
            result = similar_categories_w2v(cat, topn, model=model)
            if result: results.update(result)
        return list(results)

    if model is None:
        model = Word2VecModel.load(W2V_PATH)
    vocab = set(row['word'] for row in model.getVectors().collect())

    if category_filter not in vocab:
        raise Exception(f"Category '{category_filter}' does not exist in the category vocabulary!")

    synonyms = model.findSynonyms(category_filter, topn)
    similar = [row['word'] for row in synonyms.collect()]
    return np.union1d(similar, [category_filter]).tolist()
