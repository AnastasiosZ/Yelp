"""
Flask web interface for the Yelp Recommendation System.

Initializes Spark and the base User object ONCE at module import
(cold start: ~20-30s). Subsequent requests reuse the live SparkSession.
"""

import os
import sys
import json
from datetime import datetime

# Add project root so we can import recommender.py / utility_functions.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root so recommender.py's relative paths work
os.chdir(PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify

# Heavy imports: Spark, User, Recommendation (constructs SparkSession on import)
# W2V_MODEL/W2V_VOCAB are the shared, process-wide Word2Vec assets loaded by recommender.py.
from recommender import spark, User, Recommendation, get_recommendation, W2V_MODEL, W2V_VOCAB  # noqa: E402

# Additional imports needed to reproduce recommend() logic
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utility_functions import get_travel_duration_ors, ATTRIBUTE_IMPORTANCE  # noqa: E402


# Sorted vocab list for the dashboard's category multi-select.
W2V_VOCAB_SORTED = sorted(W2V_VOCAB)

# Build the shared User instance once at startup so the first /recommend
# request doesn't pay the cost of reading user files + joining with businesses.
USER = User()


# -------- App setup --------
app = Flask(__name__)

USER_PROFILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'example', 'user_profile.json')
USER_REVIEWS_PATH = os.path.join(PROJECT_ROOT, 'data', 'example', 'user_reviews.json')

# Default location fallback (downtown Tampa, FL)
DEFAULT_LAT = 27.9483
DEFAULT_LNG = -82.4648


# -------- Helpers --------
def load_user_profile():
    """
    Read the JSONL user profile from ``USER_PROFILE_PATH``.

    The file stores a single JSON object on one line (JSONL convention for
    this project). Returns an empty dict if the file is empty.

    Returns
    -------
    dict
        The parsed profile object, or ``{}`` if no line was found.
    """
    with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
        # File is single-line JSON
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def save_user_profile(profile):
    """
    Write ``profile`` back to ``USER_PROFILE_PATH`` as a single JSONL line.

    Overwrites the file; the trailing newline is required to preserve the
    JSONL format the rest of the pipeline (and Spark's JSON reader) expects.

    Parameters
    ----------
    profile : dict
        Profile object to serialize.
    """
    with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(profile))
        f.write('\n')


def load_user_reviews():
    """
    Read all of the user's reviews from the JSONL file at ``USER_REVIEWS_PATH``.

    One JSON object per line; blank lines are skipped.

    Returns
    -------
    list[dict]
        List of review dicts in file order.
    """
    with open(USER_REVIEWS_PATH, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_user_reviews(reviews):
    """
    Overwrite ``USER_REVIEWS_PATH`` with ``reviews`` in JSONL format.

    Each review is written as a single JSON line followed by ``\\n`` to
    preserve the newline-delimited format Spark's JSON reader expects.

    Parameters
    ----------
    reviews : list[dict]
        Review objects to persist.
    """
    with open(USER_REVIEWS_PATH, 'w', encoding='utf-8') as f:
        for r in reviews:
            f.write(json.dumps(r))
            f.write('\n')


def get_recommendations_with_geo(rec, n=5, currently_open=True, recommend_reviewed=False,
                                 trip_duration=False, detail=False):
    """
    Produce top-n recommendations as JSON-ready dicts for the web UI.

    Mirrors the ranking logic of ``Recommendation.recommend`` but emits a
    list of plain dicts including ``latitude``/``longitude`` for the
    Leaflet map. Scores candidates via importance-weighted cosine against
    the pre-computed ``rec.user_vec``.
    """
    businesses = rec.businesses

    if currently_open:
        businesses = rec.currently_open_filter(businesses)

    if not recommend_reviewed:
        businesses = rec.drop_reviewed(businesses)

    if businesses.limit(1).count() == 0:
        return []

    businesses_pd = businesses.toPandas()

    B = np.stack(businesses_pd['bvec'].apply(lambda v: v.toArray()).values)
    businesses_pd['cosine_similarity'] = B @ np.diag(ATTRIBUTE_IMPORTANCE) @ rec.user_vec

    try:
        n_bins = max(int(np.power(len(businesses_pd), 1/2)), 1)
        businesses_pd['similarity_bin'] = pd.qcut(
            businesses_pd['cosine_similarity'], n_bins, duplicates='drop', labels=False
        )
    except ValueError:
        businesses_pd['similarity_bin'] = 0

    businesses_pd = businesses_pd.sort_values(
        ['similarity_bin', 'quality'], ascending=False
    ).head(n)

    if trip_duration and rec.locality_mode == 'radial':
        user_lat, user_long = rec.query['locality'][1]
        businesses_pd['trip_duration_s'] = businesses_pd.apply(
            lambda row: get_travel_duration_ors(
                user_lat, user_long, row['latitude'], row['longitude']
            ),
            axis=1,
        )
        businesses_pd['trip_duration'] = businesses_pd['trip_duration_s'].apply(
            lambda d: f'{int(d // 60)}m {int(d % 60)}s' if d else 'N/A'
        )

    results = []
    for _, row in businesses_pd.iterrows():
        item = {
            'business_id': row.get('business_id'),
            'name': row.get('name'),
            'address': row.get('address'),
            'city': row.get('city'),
            'review_count': int(row['review_count']) if pd.notna(row.get('review_count')) else None,
            'stars': float(row['stars']) if pd.notna(row.get('stars')) else None,
            'latitude': float(row['latitude']) if pd.notna(row.get('latitude')) else None,
            'longitude': float(row['longitude']) if pd.notna(row.get('longitude')) else None,
        }

        if detail:
            cats = row.get('categories')
            if isinstance(cats, list):
                item['categories'] = ', '.join(cats)
            elif cats is None:
                item['categories'] = ''
            else:
                item['categories'] = str(cats)

        if trip_duration and rec.locality_mode == 'radial':
            item['trip_duration'] = row.get('trip_duration', 'N/A')

        results.append(item)

    return results


# -------- Routes --------
@app.route('/', methods=['GET'])
def dashboard():
    """
    Render the main dashboard (query form, Leaflet map, results table).

    The user's saved latitude/longitude (from the profile JSON) are passed
    into the template so the map centers on them and the radial-locality
    fields are pre-filled; fall back to downtown Tampa when missing.

    Returns
    -------
    str
        Rendered HTML for ``dashboard.html``.
    """
    profile = load_user_profile()
    user_lat = profile.get('latitude', DEFAULT_LAT)
    user_lng = profile.get('longitude', DEFAULT_LNG)
    return render_template('dashboard.html',
                           user_lat=user_lat,
                           user_lng=user_lng,
                           user_name=profile.get('name', 'User'))


@app.route('/categories', methods=['GET'])
def categories():
    """
    Return the full sorted Word2Vec vocabulary (1311 tokens) as JSON.

    Served from the preloaded ``W2V_VOCAB`` list — no Spark work per
    request. The dashboard's searchable multi-select hits this endpoint
    once on page load to populate its suggestions.

    Returns
    -------
    flask.Response
        JSON of the form ``{"categories": [...]}``.
    """
    return jsonify({'categories': W2V_VOCAB_SORTED})


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Run the recommender for a dashboard query and return JSON results.

    Parses the posted JSON body, normalizes it into the ``query`` dict
    shape expected by ``Recommendation.__init__``, constructs a
    ``Recommendation``, and delegates the ranking step to
    ``get_recommendations_with_geo``. The response additionally includes
    the user's lat/lng (for the blue "me" map marker) and the flags
    controlling which optional columns the table should render.

    Expected JSON body keys
    -----------------------
    locality_mode : {'radial', 'city', 'state'}
    latitude, longitude, radius : float  (radial mode)
    city / state : str  (city / state modes)
    categories : list[str] | str
    day : str
    time_minutes : int
    n, currently_open, recommend_reviewed, trip_duration, detail

    Returns
    -------
    flask.Response
        JSON with ``results``, ``user_lat``, ``user_lng``, ``detail``,
        ``trip_duration``. Returns HTTP 400 with an ``error`` message on
        bad input or backend failure.
    """
    try:
        data = request.get_json(force=True) or {}

        locality_mode = data.get('locality_mode', 'radial')
        if locality_mode == 'state':
            locality = ('state', data.get('state', 'FL'))
        elif locality_mode == 'city':
            locality = ('city', data.get('city', 'Tampa'))
        elif locality_mode == 'radial':
            lat = float(data.get('latitude', DEFAULT_LAT))
            lng = float(data.get('longitude', DEFAULT_LNG))
            radius = float(data.get('radius', 5))
            locality = ('radial', (lat, lng), radius)
        else:
            return jsonify({'error': f'Unknown locality mode: {locality_mode}'}), 400

        # Parse categories
        raw_cats = data.get('categories', 'Bars, Nightlife')
        if isinstance(raw_cats, list):
            categories = [c.strip() for c in raw_cats if c and str(c).strip()]
        else:
            categories = [c.strip() for c in str(raw_cats).split(',') if c.strip()]

        day = data.get('day', 'Friday')
        time_minutes = int(data.get('time_minutes', 21 * 60))
        scope = int(data.get('scope', 5))

        query = {
            'locality': locality,
            'categories': categories,
            'datetime': {'day': day, 'time': time_minutes},
            'scope': scope,
        }

        n = int(data.get('n', 5))
        currently_open = bool(data.get('currently_open', True))
        recommend_reviewed = bool(data.get('recommend_reviewed', False))
        trip_duration = bool(data.get('trip_duration', False))
        detail = bool(data.get('detail', False))

        rec = get_recommendation(USER, query)
        results = get_recommendations_with_geo(
            rec,
            n=n,
            currently_open=currently_open,
            recommend_reviewed=recommend_reviewed,
            trip_duration=trip_duration,
            detail=detail,
        )

        # Return user lat/lng for radial mode, else default from profile
        profile = load_user_profile()
        if locality_mode == 'radial':
            user_lat, user_lng = locality[1]
        else:
            user_lat = profile.get('latitude', DEFAULT_LAT)
            user_lng = profile.get('longitude', DEFAULT_LNG)

        return jsonify({
            'results': results,
            'user_lat': user_lat,
            'user_lng': user_lng,
            'detail': detail,
            'trip_duration': trip_duration,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/profile', methods=['GET'])
def profile():
    """
    Render the profile page — user info card plus the review list.

    Loads the JSONL profile and reviews from disk each call (cheap, small
    files) so the page always reflects the latest state after an edit.

    Returns
    -------
    str
        Rendered HTML for ``profile.html``.
    """
    prof = load_user_profile()
    reviews = load_user_reviews()
    user_lat = prof.get('latitude', DEFAULT_LAT)
    user_lng = prof.get('longitude', DEFAULT_LNG)
    return render_template('profile.html', profile=prof, reviews=reviews,
                           user_lat=user_lat, user_lng=user_lng)


@app.route('/profile/edit', methods=['POST'])
def profile_edit():
    """
    Update the user's home latitude/longitude in ``user_profile.json``.

    Reads the current profile, overwrites only the ``latitude`` and/or
    ``longitude`` fields present in the JSON body, and writes the profile
    back as JSONL. Other fields are preserved as-is.

    Returns
    -------
    flask.Response
        ``{"status": "ok", "profile": {...}}`` on success, or HTTP 400
        with an ``error`` message on bad input.
    """
    try:
        data = request.get_json(force=True) or {}
        prof = load_user_profile()
        if 'latitude' in data:
            prof['latitude'] = float(data['latitude'])
        if 'longitude' in data:
            prof['longitude'] = float(data['longitude'])
        save_user_profile(prof)
        return jsonify({'status': 'ok', 'profile': prof})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/profile/review', methods=['POST'])
def profile_add_review():
    """
    Append a new review to ``user_reviews.json`` (JSONL).

    Fills in sensible defaults for ``user_id`` (from the profile),
    ``useful``/``funny``/``cool`` (0), and ``date`` (now) when the client
    omits them, coerces ``stars`` to float, then appends and rewrites the
    file.

    Returns
    -------
    flask.Response
        ``{"status": "ok", "review": {...}}`` on success, or HTTP 400
        with an ``error`` message on bad input.
    """
    try:
        data = request.get_json(force=True) or {}
        reviews = load_user_reviews()
        # Sensible defaults if missing
        data.setdefault('user_id', load_user_profile().get('user_id', ''))
        data.setdefault('useful', 0)
        data.setdefault('funny', 0)
        data.setdefault('cool', 0)
        data.setdefault('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if 'stars' in data:
            data['stars'] = float(data['stars'])
        reviews.append(data)
        save_user_reviews(reviews)
        return jsonify({'status': 'ok', 'review': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/profile/review/<review_id>', methods=['PUT'])
def profile_update_review(review_id):
    """
    Patch a review identified by ``review_id``.

    Updates ``stars`` (coerced to float) and/or ``text`` when present in
    the JSON body, and also merges any other keys supplied — except
    ``review_id`` itself, which is immutable. Returns 404 if no review
    with that id exists.

    Parameters
    ----------
    review_id : str
        The ``review_id`` of the review to update.

    Returns
    -------
    flask.Response
        ``{"status": "ok"}`` on success, HTTP 404 if not found, or HTTP
        400 with an ``error`` message on bad input.
    """
    try:
        data = request.get_json(force=True) or {}
        reviews = load_user_reviews()
        found = False
        for r in reviews:
            if r.get('review_id') == review_id:
                if 'stars' in data:
                    r['stars'] = float(data['stars'])
                if 'text' in data:
                    r['text'] = data['text']
                # Allow other fields to be updated
                for k, v in data.items():
                    if k not in ('stars', 'text', 'review_id'):
                        r[k] = v
                found = True
                break
        if not found:
            return jsonify({'error': f'Review {review_id} not found'}), 404
        save_user_reviews(reviews)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/profile/review/<review_id>', methods=['DELETE'])
def profile_delete_review(review_id):
    """
    Delete the review with the given ``review_id`` from ``user_reviews.json``.

    Rewrites the file with the matching entry removed. Returns 404 if no
    review with that id was found (the file is not modified in that case).

    Parameters
    ----------
    review_id : str
        The ``review_id`` of the review to remove.

    Returns
    -------
    flask.Response
        ``{"status": "ok"}`` on success, HTTP 404 if not found, or HTTP
        400 with an ``error`` message on backend failure.
    """
    try:
        reviews = load_user_reviews()
        new_reviews = [r for r in reviews if r.get('review_id') != review_id]
        if len(new_reviews) == len(reviews):
            return jsonify({'error': f'Review {review_id} not found'}), 404
        save_user_reviews(new_reviews)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Disable reloader so Spark doesn't initialize twice
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
