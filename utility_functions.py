from dotenv import load_dotenv
from functools import reduce
import os
import requests
import numpy as np

import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.ml.feature import Word2VecModel



# --- Constants ---
W2V_PATH = 'models/w2v_categories'
DATA_PATH = 'data/clean/'
USER_PATH = 'data/example/'


ATTRIBUTES = {
    'GoodForDancing':             {'default': 0, 'importance': 2},
    'DogsAllowed':                {'default': 0, 'importance': 3},
    'WheelchairAccessible':       {'default': 0, 'importance': 5},
    'NoiseLevel':                 {'default': 1, 'importance': 4},
    'RestaurantsAttire':          {'default': 0, 'importance': 3},
    'ByAppointmentOnly':          {'default': 0, 'importance': 2},
    'RestaurantsGoodForGroups':   {'default': 1, 'importance': 4},
    'RestaurantsReservations':    {'default': 0, 'importance': 3},
    'OutdoorSeating':             {'default': 0, 'importance': 3},
    'GoodForKids':                {'default': 0, 'importance': 4},
    'RestaurantsDelivery':        {'default': 0, 'importance': 2},
    'RestaurantsTakeOut':         {'default': 1, 'importance': 3},
    'RestaurantsPriceRange2':     {'default': 2, 'importance': 5},
    'BusinessAcceptsCreditCards': {'default': 1, 'importance': 4},
    'Ambience_romantic':          {'default': 0, 'importance': 3},
    'Ambience_intimate':          {'default': 0, 'importance': 3},
    'Ambience_touristy':          {'default': 0, 'importance': 2},
    'Ambience_hipster':           {'default': 0, 'importance': 2},
    'Ambience_divey':             {'default': 0, 'importance': 2},
    'Ambience_classy':            {'default': 0, 'importance': 3},
    'Ambience_trendy':            {'default': 0, 'importance': 3},
    'Ambience_upscale':           {'default': 0, 'importance': 3},
    'Ambience_casual':            {'default': 0, 'importance': 3}
}

ATTRIBUTE_IMPORTANCE = [attr['importance'] for attr in ATTRIBUTES.values()]

DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']


class MeanImputeFallback(Transformer):

    def _transform(self, df): # type:ignore

        counts = df.select([F.count(c).alias(c) for c in list(ATTRIBUTES.keys())]).first().asDict()
        means = df.select([F.mean(c).alias(c) for c in list(ATTRIBUTES.keys())]).first().asDict()

        fill_dict = {
            c: (ATTRIBUTES[c]['default'] if counts[c] == 0 else means[c])
            for c in list(ATTRIBUTES.keys())
        }

        return reduce(lambda acc, item: acc.withColumn(
                          item[0], F.when(F.col(item[0]).isNull(), F.lit(item[1])).otherwise(F.col(item[0]))),
                      fill_dict.items(), df)






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

    dlat = F.radians(lat2_col - lat1_col)
    dlon = F.radians(lon2_col - lon1_col)

    a = F.sin(dlat/2)**2 + F.cos(F.radians(lat1_col)) * F.cos(F.radians(lat2_col)) * F.sin(dlon/2)**2
    c = 2 * F.asin(F.sqrt(a))

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




def similar_categories_w2v(model, vocab, category_filter: str | list[str] , topn=5):

    # If given a list of categories, self-call each category
    if isinstance(category_filter,list):
        
        results = set()
        for cat in category_filter:
            result = similar_categories_w2v(model, vocab, cat, topn)
            if result: results.update(result)
        return list(results)
    
    if category_filter not in vocab:
        raise Exception(f"Category '{category_filter}' does not exist in the category vocabulary!")

    if topn==0: return [category_filter]

    synonyms = model.findSynonymsArray(category_filter, topn)
    similar = set([word[0] for word in synonyms])
    similar.add(category_filter)

    return list(similar)

