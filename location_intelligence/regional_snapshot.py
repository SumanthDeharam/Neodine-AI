"""Radius searches over the bundled Marathahalli OpenFreeMap POI extract.

Data © OpenStreetMap contributors (ODbL 1.0), via OpenFreeMap/OpenMapTiles.
The compressed JSON contains its source tile URL, coverage, date and license.
"""
import gzip
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

from location_intelligence.models import CompetitorPOI, DataState, EnvironmentPOI, QueryResult
from location_intelligence.overpass_provider import OverpassProvider


@lru_cache(maxsize=1)
def _load_snapshot():
    path = Path(__file__).with_name("marathahalli_places.json.gz")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, EOFError, ValueError) as exc:
        logging.getLogger(__name__).warning("Local map extract unavailable: %s", exc)
        return None


def search_snapshot(lat, lon, radius):
    """Return None when any part of the search circle falls outside the extract."""
    snapshot = _load_snapshot()
    if not snapshot or radius <= 0:
        return None
    west, south, east, north = snapshot["bbox"]
    if not (south < lat < north and west < lon < east):
        return None
    angle = radius / 6371000
    lat_margin = math.degrees(angle)
    if lat - lat_margin < south or lat + lat_margin > north:
        return None
    lon_margin = math.degrees(math.asin(math.sin(angle) / math.cos(math.radians(lat))))
    if lon - lon_margin < west or lon + lon_margin > east:
        return None

    source = f"OpenFreeMap / OpenMapTiles · © OpenStreetMap contributors · {snapshot['date']} snapshot"
    competitors, pois = [], []
    for feature_id, place_lat, place_lon, tag, kind, name in snapshot["records"]:
        if abs(place_lat - lat) > lat_margin or abs(place_lon - lon) > lon_margin:
            continue
        distance = OverpassProvider.haversine(lat, lon, place_lat, place_lon)
        if distance > radius:
            continue
        category = kind.replace("_", " ").title()
        details = dict(name=name or f"Unnamed {category}", lat=place_lat, lng=place_lon,
                       distance=distance, source=source)
        if tag == "food":
            competitors.append(CompetitorPOI(source_id=f"openmaptiles:poi:{feature_id}",
                                             osm_type=None, category=category, address="", **details))
        else:
            pois.append(EnvironmentPOI(type="office" if tag == "office" else kind,
                                       tags={tag: kind}, **details))
    competitors.sort(key=lambda place: place.distance)
    count = len(competitors) + len(pois)
    return QueryResult(status=DataState.REAL_DATA if count else DataState.NO_RESULTS,
                       competitors=competitors, pois=pois, raw_count=count, http_status=200,
                       provider_name=source, snapshot_date=snapshot["date"])
