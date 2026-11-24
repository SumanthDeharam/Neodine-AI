import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from location_intelligence.models import (
    QueryResult, DataState, CompetitorPOI, EnvironmentPOI
)
from location_intelligence.provider import LocationDataProvider
from location_intelligence.overpass_provider import OverpassProvider
from location_intelligence.regional_snapshot import search_snapshot
from location_intelligence.geocoder import GeocoderService
from location_intelligence.analyzer import LocationMarketAnalyzer

logger = logging.getLogger("location_intelligence")

class LocationIntelligenceService:
    """
    Unified Location Intelligence Service.
    Coordinates geocoding, external provider queries, caching, analysis, and database updates.
    """

    def __init__(self, provider: Optional[LocationDataProvider] = None):
        self.provider = provider or OverpassProvider()
        self.use_snapshot = provider is None

    def fetch_and_analyze(self, latitude: float, longitude: float, radius: float) -> Dict[str, Any]:
        """
        Executes immediate external location query and market analysis for given coordinates and radius.
        """
        lat, lon, is_valid = GeocoderService.validate_and_resolve(None, latitude, longitude)
        if not is_valid:
            error_qr = QueryResult(
                status=DataState.INVALID_LOCATION,
                error_message="Invalid latitude or longitude coordinates provided."
            )
            return LocationMarketAnalyzer.analyze(error_qr, latitude or 0.0, longitude or 0.0, radius)

        query_result = search_snapshot(lat, lon, radius) if self.use_snapshot else None
        if query_result is None:
            query_result = self.provider.search_nearby(lat, lon, radius)
        analysis_payload = LocationMarketAnalyzer.analyze(query_result, lat, lon, radius)
        return analysis_payload

    def sync_and_get_dashboard_data(self, loc_intel_model, db_session, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves dashboard market analysis for a LocationIntelligence SQLAlchemy model instance.
        Performs caching logic and updates DB record cleanly.
        """
        if not loc_intel_model or loc_intel_model.latitude is None or loc_intel_model.longitude is None:
            return {
                'success': False,
                'configured': False,
                'message': 'Restaurant location is required to calculate nearby competitors.'
            }

        import time
        t_total_0 = time.perf_counter()

        lat = loc_intel_model.latitude
        lon = loc_intel_model.longitude
        radius = loc_intel_model.radius or 1000

        # Validate coordinates
        t_geo_0 = time.perf_counter()
        lat, lon, is_valid = GeocoderService.validate_and_resolve(loc_intel_model.address, lat, lon)
        t_geo_ms = round((time.perf_counter() - t_geo_0) * 1000.0, 1)

        if not is_valid:
            error_qr = QueryResult(
                status=DataState.INVALID_LOCATION,
                error_message="Coordinates could not be validated for restaurant location."
            )
            payload = LocationMarketAnalyzer.analyze(error_qr, loc_intel_model.latitude or 0.0, loc_intel_model.longitude or 0.0, radius)
            payload['address'] = loc_intel_model.address
            payload['latitude'] = loc_intel_model.latitude
            payload['longitude'] = loc_intel_model.longitude
            payload['radius'] = radius
            payload['has_google_key'] = bool((loc_intel_model.google_maps_key or '').strip())
            return payload

        # Update model coordinates if resolved coordinates changed
        if is_valid and (loc_intel_model.latitude != lat or loc_intel_model.longitude != lon):
            loc_intel_model.latitude = lat
            loc_intel_model.longitude = lon
            loc_intel_model.competitors_cached = None
            loc_intel_model.nearby_data_cached = None

        need_sync = force_refresh
        if not need_sync:
            if not loc_intel_model.competitors_cached or not loc_intel_model.nearby_data_cached:
                need_sync = True
            elif loc_intel_model.last_updated:
                # Refresh if older than 24 hours
                if datetime.utcnow() - loc_intel_model.last_updated > timedelta(hours=24):
                    need_sync = True

        query_result = search_snapshot(lat, lon, radius) if self.use_snapshot else None
        t_ov_0 = time.perf_counter()

        if query_result is not None:
            # The versioned extract already supplies a local cache for the whole circle.
            t_ov_ms = 0.0
        elif need_sync:
            logger.info(f"[LOCATION-AI] Performing fresh Overpass query for location ({lat}, {lon}), radius {radius}m")
            query_result = self.provider.search_nearby(lat, lon, radius)
            t_ov_ms = round((time.perf_counter() - t_ov_0) * 1000.0, 1)

            # Persist to database cache only on success/real data
            if query_result.status in (DataState.REAL_DATA, DataState.NO_RESULTS, DataState.ZERO_RESULTS):
                try:
                    competitors_json = json.dumps([c.to_dict() for c in query_result.competitors])
                    pois_json = json.dumps([p.to_dict() for p in query_result.pois])
                    
                    loc_intel_model.competitors_cached = competitors_json
                    loc_intel_model.nearby_data_cached = pois_json
                    loc_intel_model.last_updated = datetime.utcnow()
                    db_session.commit()
                    logger.info(f"[LOCATION-AI] Successfully updated DB cache for restaurant_id {loc_intel_model.restaurant_id}")
                except Exception as db_err:
                    db_session.rollback()
                    logger.error(f"[LOCATION-AI] Database caching error: {db_err}")
        else:
            # Reconstruct QueryResult from database cache
            try:
                from dataclasses import fields
                def _to_dc(cls, d):
                    if isinstance(d, cls): return d
                    valid_keys = {f.name for f in fields(cls)}
                    cleaned = {k: v for k, v in d.items() if k in valid_keys}
                    return cls(**cleaned)

                raw_comps = json.loads(loc_intel_model.competitors_cached) if loc_intel_model.competitors_cached else []
                raw_pois = json.loads(loc_intel_model.nearby_data_cached) if loc_intel_model.nearby_data_cached else []

                competitors = [_to_dc(CompetitorPOI, c) for c in raw_comps if isinstance(c, dict)]
                pois = [_to_dc(EnvironmentPOI, p) for p in raw_pois if isinstance(p, dict)]

                status = DataState.REAL_DATA if competitors or pois else DataState.NO_RESULTS
                query_result = QueryResult(
                    status=status,
                    competitors=competitors,
                    pois=pois,
                    raw_count=len(competitors) + len(pois),
                    http_status=200,
                    provider_name="OpenStreetMap Overpass API",
                    perf_metrics={'overpass_ms': 0.0, 'parsing_ms': 2.0, 'distance_ms': 0.0}
                )
                t_ov_ms = round((time.perf_counter() - t_ov_0) * 1000.0, 1)
            except Exception as cache_err:
                logger.warning(f"[LOCATION-AI] Failed parsing DB cache: {cache_err}. Performing fresh fetch.")
                query_result = self.provider.search_nearby(lat, lon, radius)
                t_ov_ms = round((time.perf_counter() - t_ov_0) * 1000.0, 1)

        t_an_0 = time.perf_counter()
        payload = LocationMarketAnalyzer.analyze(query_result, lat, lon, radius)
        t_an_ms = round((time.perf_counter() - t_an_0) * 1000.0, 1)

        t_total_ms = round((time.perf_counter() - t_total_0) * 1000.0, 1)

        perf = payload.get('perf_metrics', {})
        print(f"[LOCATION-AI-PERF] Geocoding: {t_geo_ms} ms")
        print(f"[LOCATION-AI-PERF] Overpass request: {t_ov_ms} ms")
        print(f"[LOCATION-AI-PERF] Parsing: {perf.get('parsing_ms', 0.0)} ms")
        print(f"[LOCATION-AI-PERF] Distance calculations: {perf.get('distance_ms', 0.0)} ms")
        print(f"[LOCATION-AI-PERF] Environment analysis: {perf.get('env_ms', 0.0)} ms")
        print(f"[LOCATION-AI-PERF] Market analysis: {perf.get('market_ms', 0.0)} ms")
        print(f"[LOCATION-AI-PERF] Total: {t_total_ms} ms")
        
        # Inject standard model fields expected by UI contract
        payload['address'] = loc_intel_model.address
        payload['latitude'] = lat
        payload['longitude'] = lon
        payload['radius'] = radius
        payload['has_google_key'] = bool((loc_intel_model.google_maps_key or '').strip())

        return payload
