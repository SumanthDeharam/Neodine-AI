import math
import time
import requests
import logging
from typing import List, Dict, Any, Tuple, Optional
from location_intelligence.provider import LocationDataProvider
from location_intelligence.models import QueryResult, DataState, CompetitorPOI, EnvironmentPOI

logger = logging.getLogger("location_intelligence")

class OverpassProvider(LocationDataProvider):
    """
    OpenStreetMap Overpass API Provider for real nearby market data.
    Dynamically queries Overpass QL without using Google Places API or paid keys.
    """
    
    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter"
    ]
    
    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'NeoDineAI-LocationEngine/3.0 (OpenStreetMap Real Market Analysis)',
            'Accept': 'application/json'
        }

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates exact geographic distance in meters between two lat/lon pairs using Haversine formula.
        """
        R = 6371000.0  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2.0) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2.0) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(R * c, 1)

    def _build_overpass_query(self, lat: float, lon: float, radius: float) -> str:
        """
        Constructs light, high-performance Overpass QL query searching for food businesses
        and environment POIs within radius.
        """
        r_int = int(radius)
        query = f"""[out:json][timeout:{self.timeout}];
(
  nwr["amenity"~"^(restaurant|cafe|fast_food|food_court|bakery|pub|bar|bistro|ice_cream|school|college|university|hospital|clinic|marketplace)$"](around:{r_int},{lat},{lon});
  nwr["shop"~"^(mall|supermarket|department_store|bakery)$"](around:{r_int},{lat},{lon});
  nwr["tourism"~"^(hotel|hostel|guest_house)$"](around:{r_int},{lat},{lon});
  nwr["highway"~"^(bus_stop)$"](around:{r_int},{lat},{lon});
  nwr["railway"~"^(station|subway_entrance)$"](around:{r_int},{lat},{lon});
  nwr["office"](around:{r_int},{lat},{lon});
);
out center;"""
        return query


    def search_nearby(self, latitude: float, longitude: float, radius: float) -> QueryResult:
        """
        Executes external Overpass query against mirror endpoints, parses raw OSM JSON objects,
        calculates exact Haversine distances, filters by user radius, and returns normalized result.
        """
        if latitude is None or longitude is None or not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0) or (latitude == 0.0 and longitude == 0.0):
            logger.error(f"[LOCATION-AI] Invalid coordinates: ({latitude}, {longitude})")
            return QueryResult(
                status=DataState.INVALID_LOCATION,
                error_message="Invalid latitude or longitude coordinates provided."
            )

        radius = max(100.0, float(radius or 1000))
        query_str = self._build_overpass_query(latitude, longitude, radius)
        
        last_exception = None
        http_status = 500
        raw_elements = []

        t_req_0 = time.perf_counter()
        for ep in self.OVERPASS_ENDPOINTS:
            try:
                logger.info(f"[LOCATION-AI] Querying Overpass mirror: {ep}")
                res = requests.post(ep, data={"data": query_str}, headers=self.headers, timeout=self.timeout)
                http_status = res.status_code
                if res.status_code == 200:
                    payload = res.json()
                    if payload.get('remark') or not isinstance(payload.get('elements'), list):
                        raise ValueError('Incomplete Overpass response')
                    raw_elements = payload['elements']
                    logger.info(f"[LOCATION-AI] Overpass HTTP 200 OK from {ep}. Returned {len(raw_elements)} OSM elements.")
                    break
                else:
                    logger.warning(f"[LOCATION-AI] Overpass endpoint {ep} returned status {res.status_code}")
            except Exception as err:
                http_status = 502
                last_exception = err
                logger.warning(f"[LOCATION-AI] Exception reaching Overpass endpoint {ep}: {err}")

        overpass_ms = round((time.perf_counter() - t_req_0) * 1000.0, 1)

        if http_status != 200:
            error_msg = f"Overpass API could not be reached: {str(last_exception or http_status)}"
            logger.error(f"[LOCATION-AI] {error_msg}")
            return QueryResult(
                status=DataState.PROVIDER_ERROR,
                http_status=http_status,
                error_message=error_msg,
                perf_metrics={'overpass_ms': overpass_ms, 'parsing_ms': 0.0, 'distance_ms': 0.0}
            )

        t_parse_0 = time.perf_counter()
        dist_total_time = 0.0

        competitors: List[CompetitorPOI] = []
        pois: List[EnvironmentPOI] = []
        seen_competitor_keys = set()

        food_amenities = {'restaurant', 'cafe', 'fast_food', 'food_court', 'bakery', 'pub', 'bar', 'bistro', 'ice_cream'}

        for el in raw_elements:
            tags = el.get('tags', {})
            el_type = el.get('type', 'node')
            el_id = el.get('id')
            
            # Extract coordinates (node lat/lon or way/relation center lat/lon)
            el_lat = el.get('lat', el.get('center', {}).get('lat'))
            el_lon = el.get('lon', el.get('center', {}).get('lon'))

            if el_lat is None or el_lon is None:
                continue

            try:
                el_lat = float(el_lat)
                el_lon = float(el_lon)
            except (ValueError, TypeError):
                continue

            # Calculate actual geographic distance (measured for perf tracking)
            t_d0 = time.perf_counter()
            dist = self.haversine(latitude, longitude, el_lat, el_lon)
            dist_total_time += (time.perf_counter() - t_d0)
            
            # Strict radius filtering
            if not math.isfinite(dist) or dist > radius:
                continue

            source_id = f"{el_type}/{el_id}" if el_id else None
            amenity = tags.get('amenity', '').lower()

            if tags.get('shop') == 'bakery':
                amenity = 'bakery'
            if amenity in food_amenities or any(k in tags for k in ['cuisine', 'menu']):
                # Normalize Food Competitor
                display_name = tags.get('name') or tags.get('brand') or tags.get('name:en')
                if not display_name:
                    display_name = f"Unnamed {amenity.replace('_', ' ').title() or 'Food Business'}"

                # Deduplicate by source_id or name+location key
                dedup_key = source_id or (display_name.lower(), round(el_lat, 4), round(el_lon, 4))
                if dedup_key in seen_competitor_keys:
                    continue
                seen_competitor_keys.add(dedup_key)

                category = tags.get('cuisine') or amenity or 'restaurant'
                category_formatted = category.replace('_', ' ').title()

                # Build address representation
                addr_parts = []
                if tags.get('addr:housenumber'): addr_parts.append(tags.get('addr:housenumber'))
                if tags.get('addr:street'): addr_parts.append(tags.get('addr:street'))
                if tags.get('addr:suburb'): addr_parts.append(tags.get('addr:suburb'))
                if tags.get('addr:city'): addr_parts.append(tags.get('addr:city'))
                address_str = ", ".join(addr_parts) if addr_parts else tags.get('addr:full') or "Local Area"

                competitors.append(CompetitorPOI(
                    source_id=source_id,
                    osm_type=el_type,
                    osm_id=el_id,
                    name=display_name,
                    category=category_formatted,
                    cuisine=tags.get('cuisine'),
                    address=address_str,
                    lat=el_lat,
                    lng=el_lon,
                    distance=dist,
                    rating=None,  # Ratings NOT present in OSM; MUST remain None
                    reviews=0,
                    price_level=None,  # Price NOT present in OSM; MUST remain None
                    opening_hours=tags.get('opening_hours'),
                    website=tags.get('website') or tags.get('contact:website'),
                    phone=tags.get('phone') or tags.get('contact:phone'),
                    source="OpenStreetMap"
                ))
            else:
                # Normalize Environment POI
                p_name = tags.get('name') or tags.get('brand') or f"POI ({amenity or tags.get('building') or 'spot'})"
                p_type = amenity or tags.get('building') or tags.get('shop') or tags.get('tourism') or tags.get('highway') or 'poi'
                pois.append(EnvironmentPOI(
                    type=p_type,
                    name=p_name,
                    vicinity=tags.get('addr:street', 'Local Area'),
                    lat=el_lat,
                    lng=el_lon,
                    distance=dist,
                    tags=tags,
                    source="OpenStreetMap"
                ))

        # Sort competitors by distance ascending
        competitors = sorted(competitors, key=lambda x: x.distance)

        parsing_ms = round((time.perf_counter() - t_parse_0) * 1000.0, 1)
        distance_ms = round(dist_total_time * 1000.0, 1)
        perf_metrics = {
            'overpass_ms': overpass_ms,
            'parsing_ms': parsing_ms,
            'distance_ms': distance_ms
        }

        # Log detailed location intelligence output
        logger.info(f"[LOCATION-AI] Coordinates: {latitude}, {longitude}")
        logger.info(f"[LOCATION-AI] Radius: {radius}m")
        logger.info(f"[LOCATION-AI] HTTP Status: {http_status}")
        logger.info(f"[LOCATION-AI] OSM Objects Returned: {len(raw_elements)}")
        logger.info(f"[LOCATION-AI] Food Businesses Found: {len(competitors)}")
        logger.info(f"[LOCATION-AI] After Radius Filtering: {len(competitors)}")
        logger.info(f"[LOCATION-AI] Final Competitors: {len(competitors)}")

        # Determine Data State
        if len(competitors) == 0 and len(pois) == 0:
            return QueryResult(
                status=DataState.NO_RESULTS,
                competitors=[],
                pois=[],
                raw_count=len(raw_elements),
                http_status=200,
                provider_name="OpenStreetMap Overpass",
                perf_metrics=perf_metrics
            )

        return QueryResult(
            status=DataState.REAL_DATA,
            competitors=competitors,
            pois=pois,
            raw_count=len(raw_elements),
            http_status=200,
            provider_name="OpenStreetMap Overpass",
            perf_metrics=perf_metrics
        )
