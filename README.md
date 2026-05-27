# Yelp Business Recommendation System

![Demo](demo.gif)

A content-based business recommender built on the Yelp Open Dataset with Apache Spark, exposed both as a Python backend and an interactive Flask web app. Given a user's review history and a query (locality, categories, day, time), it returns the top-N businesses ranked by how well their attributes match the user's taste, filtered by where and when the user actually wants to go.

## Premise

Most recommenders rank against a global popularity prior or a sparse user-item matrix. This project takes a different cut: each business is reduced to a dense vector over its own attributes (price range, noise level, attire, dancing, ambience, kid-friendliness, outdoor seating, etc.), the user is summarized as the star-weighted mean of the businesses they reviewed, and recommendations are produced by an *importance-weighted* cosine similarity in that shared attribute space. Locality, opening hours, and a popularity tiebreaker layer on top to keep the results actionable rather than just similar.

Everything that can be precomputed is pushed into the cleaning stage so the request-time path is small: boolean and string attributes are cast/encoded to numeric columns, opening hours are parsed once into `<Day>_open_mins` / `<Day>_close_mins` integer columns, and the popularity score `quality = round(log10(review_count + 1) * stars², 3)` is materialized directly into `business.parquet`. The runtime pipeline is purely numeric and never touches the raw attribute structs or hour strings.

## Backend

The backend lives in [recommender.py](recommender.py) and [utility_functions.py](utility_functions.py) and is organized around two classes plus an instance cache:

- **`User`** — reads `user_profile.json` and `user_reviews.json` from [data/example/](data/example/), aggregates duplicate (user, business) review pairs by taking the most recent star rating (`F.last('stars')` after sorting by date), and joins the result against the cached `BUSINESSES_DF`. The joined `user_businesses` DataFrame is persisted for reuse.
- **`Recommendation(User, query)`** — runs the category-and-locality filter cascade, materializes the candidate pool, computes the user's preference vector against this *specific* category context, and exposes `recommend(...)` for the auxiliary filters and ranking step.
- **`get_recommendation(user, query)`** — wraps construction in a process-wide `RECOMMENDATIONS` cache keyed by `(id(user), sorted categories, scope, locality)`. Auxiliary query parameters (`datetime`, `n`, `currently_open`, `recommend_reviewed`, `trip_duration`, `detail`) reuse the cached instance, so re-running the same base query with a different day/time or result count skips all Spark work and only re-runs the Pandas-side ranking.

### Query shape

```python
query = {
    'locality':   ('radial', (lat, lng), radius_km),    # or ('state', 'FL') / ('city', 'Tampa')
    'categories': ['Nightlife', 'Bars'],
    'datetime':   {'day': 'Friday', 'time': 21*60},     # minutes since midnight
    'scope':      2,                                    # W2V synonym breadth (0 = strict)
}
```

### Pipeline

1. **Category expansion** — a 100-D Word2Vec model trained on the business category vocabulary expands each query category via `findSynonyms` (capped at `scope` neighbours per term, `scope=0` keeping the term as-is). The expanded set is checked against the model's vocabulary upfront, so an out-of-vocab category raises immediately. Businesses are filtered to those whose `categories` array contains at least one matched term.
2. **Attribute vectorization** — a Spark ML `PipelineModel` (`MeanImputeFallback → VectorAssembler → Normalizer(L2)`) is applied to the category-filtered pool. The custom `MeanImputeFallback` transformer per-column fills nulls with the *column mean* on the current DataFrame, or with a hard-coded default (see `ATTRIBUTES` in [utility_functions.py](utility_functions.py)) when the column is entirely null. Because the means come from a wider candidate pool than the post-locality survivors, the vectorization is run *before* the locality and open-business filters to maximize the signal available for imputation. The pipeline itself is a `PipelineModel`, not a `Pipeline` — there is no `fit` step at request time.
3. **Locality filter** — state/city equality, or radial distance using a pure Spark-SQL haversine implementation (no UDF). Followed by an `is_open=True` filter to drop permanently closed businesses. The resulting pool is persisted as `self.businesses`.
4. **User preference vector** — the user's reviewed businesses are restricted to the same category set, then passed through `BUSINESS_PIPE`, and `Summarizer.mean(bvec, weightCol=user_stars)` produces the star-weighted mean L2 vector. If the user has no reviews in these categories, the recommender falls back to the unweighted mean of the candidate pool (a neutral profile against which `quality` does the work).
5. **Auxiliary filters** (per `recommend()` call) —
   - **Currently-open**: uses the pre-materialized `<Day>_open_mins` / `<Day>_close_mins` columns; handles both same-day and overnight (previous-day close-time spilling over) cases without any string parsing.
   - **Drop-reviewed**: left-anti join against the user's reviewed business IDs.
6. **Scoring** — the candidate pool is collected to Pandas; each business's L2 vector is scored against the user vector with an *importance-weighted* dot product `B @ diag(ATTRIBUTE_IMPORTANCE) @ user_vec`, where `ATTRIBUTE_IMPORTANCE` is the hand-tuned per-attribute weight vector defined in [utility_functions.py](utility_functions.py) (e.g. `RestaurantsPriceRange2` = 5, `WheelchairAccessible` = 5, ambience flags = 2–3). Scores are then bucketed into `sqrt(n_candidates)` quantile bins (`pd.qcut(..., duplicates='drop')`), and the top-N is selected by sorting on `(similarity_bin, quality)` descending — so the pre-materialized `quality = log10(reviews+1) * stars²` acts as the intra-bin tiebreaker.
7. **Optional trip duration** — OpenRouteService driving-time lookup is called only for the final top-N rows in radial mode, so the slow free-tier API never sees the full candidate pool.

### Performance

The module-level `SparkSession`, `BUSINESSES_DF`, `W2V_MODEL`, `W2V_VOCAB`, and `BUSINESS_PIPE` are all built once on import and shared by every `User`/`Recommendation` instance — there is no per-query disk I/O or model reload. On top of that, the `RECOMMENDATIONS` cache means repeated dashboard queries that only differ in their auxiliary parameters skip the entire Spark stage.

## Web app

[web/app.py](web/app.py) wraps the backend in a Flask app with a dashboard and a profile page. The dashboard is a two-column layout: a query form on the left, a Leaflet map (OpenStreetMap tiles, no API key) and results table on the right.

### Features

- **Locality picker** — toggle between `state`, `city`, and `radial`; in radial mode the latitude/longitude inputs sit side-by-side and the radius is a slider with a live-updating value.
- **Category multi-select** — a searchable chip UI backed by the full sorted Word2Vec vocabulary served from `/categories`. Substring matching, keyboard navigation (arrow keys, Enter, Backspace, Escape), and a guarantee that any selection will pass `similar_categories_w2v` without raising.
- **Search-scope slider** — `Small ↔ Large`, controls how many Word2Vec synonyms each selected category is expanded with (maps directly to `findSynonyms`' `topn`).
- **Day and time defaults** — Mon–Sun dropdown defaults to today, time input defaults to now (both client-side).
- **Result options** — toggles for `currently_open`, `recommend_reviewed`, `trip_duration` (radial only), and `detail` (adds the categories column).
- **Interactive map** — a persistent blue user marker and red business markers; the table and markers are linked, with hover/click highlighting and `fitBounds` to auto-frame each result set.
- **Profile page** — view and edit the user's location, and add/edit/delete reviews directly via `GET /profile`, `POST /profile/edit`, `POST /profile/review`, `PUT /profile/review/<id>`, `DELETE /profile/review/<id>`. Changes are written back to the JSONL files the backend reads.

The first request blocks ~20–30 s on the Spark cold-start. The `USER` instance is built once at startup so the first `/recommend` doesn't also pay for the user-side join, and the shared `RECOMMENDATIONS` cache makes subsequent re-queries with different auxiliary parameters effectively instant.

## Data

Raw Yelp JSON (businesses, users, reviews, tips, check-ins) is cleaned once in [preprocessing.ipynb](preprocessing.ipynb) and written to typed Parquet under `data/clean/`. The cleaned parquet files and the trained Word2Vec model are not tracked in git — regenerate them locally before running. An example user **"Doug"** — a party-animal persona in Tampa, FL — lives in [data/example/](data/example/) and drives the demos.

## Components

- **[recommender.py](recommender.py)** — `User`, `Recommendation`, `get_recommendation`, the shared `BUSINESS_PIPE`, and the `RECOMMENDATIONS` cache.
- **[utility_functions.py](utility_functions.py)** — the `ATTRIBUTES` table (with per-column default + importance weight), `ATTRIBUTE_IMPORTANCE`, `DAYS`, the `MeanImputeFallback` Spark ML transformer, `haversine_spark`, `get_travel_duration_ors`, and `similar_categories_w2v`.
- **[preprocessing.ipynb](preprocessing.ipynb)** — one-time ETL from raw JSON to typed Parquet, including the `quality` column and the per-day `*_open_mins` / `*_close_mins` columns.
- **[web/](web/)** — Flask app: dashboard, profile page, Leaflet map, Word2Vec-backed category picker.

## Running

First activate the virtual environment, then:

```bash
python recommender.py     # CLI — runs the hard-coded Tampa/Nightlife query in __main__ and exercises the recommendation cache
python web/app.py         # Web UI at http://127.0.0.1:5000
```

---

# Versions

### Python
Python 3.11.9 ([python.org](https://www.python.org/downloads/release/python-3119/))

### Java
Java JDK 21 LTS (Eclipse Temurin 21 LTS) ([adoptium.net](https://adoptium.net/en-GB/temurin/releases?version=21&os=any&arch=any))
Don't forget to set the environment variables and paths.

### Hadoop winutils/hadoop.dll
Version 3.3.6 ([cdarlint/winutils](https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin)) in `C:/hadoop/bin/`.

### Libraries
- PySpark 3.5.1
- All libraries and their versions saved in `requirements.txt`
