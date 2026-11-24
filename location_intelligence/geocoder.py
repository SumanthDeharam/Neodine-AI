import logging
from typing import Any, Tuple, Optional

logger = logging.getLogger("location_intelligence")

class GeocoderService:
    """
    Geocoding helper for validating coordinates and resolving target address bounds.
    """
    
    # Standard fallback coordinates for well-known hub names if user searches address without lat/lon
    CITY_COORDINATES = {
        'marathahalli': (12.956366, 77.706997),
        'bengaluru': (12.9716, 77.5946),
        'bangalore': (12.9716, 77.5946),
        'koramangala': (12.9352, 77.6245),
        'hsr': (12.9121, 77.6445),
        'indiranagar': (12.9784, 77.6408),
        'whitefield': (12.9698, 77.7500),
        'hyderabad': (17.3850, 78.4867),
        'gachibowli': (17.4401, 78.3489),
        'secunderabad': (17.4399, 78.4983),
        'mumbai': (19.0760, 72.8777),
        'bandra': (19.0596, 72.8295),
        'andheri': (19.1197, 72.8464),
        'powai': (19.1176, 72.9060),
        'delhi': (28.6139, 77.2090),
        'new delhi': (28.6139, 77.2090),
        'gurgaon': (28.4595, 77.0266),
        'noida': (28.5355, 77.3910),
        'chennai': (13.0827, 80.2707),
        'pune': (18.5204, 73.8567),
        'kolkata': (22.5726, 88.3639)
    }

    @classmethod
    def validate_and_resolve(cls, address: Optional[str], latitude: Any, longitude: Any) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Validates latitude and longitude.
        If address is provided, checks if address contains a specific sub-area key in CITY_COORDINATES
        (e.g., 'whitefield', 'marathahalli', 'koramangala', 'indiranagar').
        Otherwise falls back to provided non-zero lat/lon or Nominatim geocoding.
        Returns: (latitude, longitude, is_valid)
        """
        addr_clean = (address or '').strip().lower()
        
        cities = {'bengaluru', 'bangalore', 'hyderabad', 'mumbai', 'delhi',
                  'new delhi', 'chennai', 'pune', 'kolkata'}
        places = sorted(cls.CITY_COORDINATES, key=lambda key: key in cities)

        # 1. If address specifies a known sub-area key in CITY_COORDINATES, check if provided lat/lon are default city center or zero/missing
        if addr_clean:
            for key in places:
                coords = cls.CITY_COORDINATES[key]
                if key in addr_clean and key not in ('bengaluru', 'bangalore'):
                    if latitude is None or longitude is None:
                        logger.info(f"[LOCATION-AI] Resolved address '{address}' to sub-area coordinates {coords}")
                        return coords[0], coords[1], True
                    try:
                        lat_f = float(latitude)
                        lon_f = float(longitude)
                        # If lat/lon are default center (12.9716, 77.5946) or 0 while address is a specific sub-area (like Whitefield), override with sub-area coords
                        if (abs(lat_f - 12.9716) < 0.005 and abs(lon_f - 77.5946) < 0.005) or (lat_f == 0.0 and lon_f == 0.0):
                            logger.info(f"[LOCATION-AI] Overriding default city center coords with sub-area '{key}' coords {coords}")
                            return coords[0], coords[1], True
                    except (ValueError, TypeError):
                        pass

        # 2. If lat/lon explicitly provided, non-zero, and NOT generic default center when an address is provided
        try:
            if latitude is not None and longitude is not None:
                lat_f = float(latitude)
                lon_f = float(longitude)
                is_default_center = (abs(lat_f - 12.9716) < 0.005 and abs(lon_f - 77.5946) < 0.005)
                # If non-zero and not default center, or if no address provided, accept provided lat/lon
                if -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0 and not (lat_f == 0.0 and lon_f == 0.0):
                    if not is_default_center or not addr_clean or addr_clean in ('bengaluru', 'bangalore', 'bengaluru, karnataka'):
                        return lat_f, lon_f, True
        except (ValueError, TypeError):
            pass

        # 3. Fallback: check CITY_COORDINATES for any match
        if addr_clean:
            for key in places:
                coords = cls.CITY_COORDINATES[key]
                if key in addr_clean:
                    logger.info(f"[LOCATION-AI] Resolved address '{address}' to coordinates {coords}")
                    return coords[0], coords[1], True

            # 4. External Nominatim lookup fallback if address is not in CITY_COORDINATES
            try:
                import requests
                headers = {'User-Agent': 'NeoDineAI-LocationEngine/3.0'}
                url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(address)}&limit=1"
                res = requests.get(url, headers=headers, timeout=4)
                if res.status_code == 200:
                    data = res.json()
                    if data and len(data) > 0:
                        n_lat = float(data[0]['lat'])
                        n_lon = float(data[0]['lon'])
                        logger.info(f"[LOCATION-AI] Nominatim resolved '{address}' to ({n_lat}, {n_lon})")
                        return n_lat, n_lon, True
            except Exception as n_err:
                logger.warning(f"[LOCATION-AI] Nominatim geocode lookup failed for '{address}': {n_err}")

        # If provided lat/lon was non-zero default center, return default center as last resort
        try:
            if latitude is not None and longitude is not None:
                lat_f = float(latitude)
                lon_f = float(longitude)
                if -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0 and not (lat_f == 0.0 and lon_f == 0.0):
                    return lat_f, lon_f, True
        except (ValueError, TypeError):
            pass

        logger.warning(f"[LOCATION-AI] Could not resolve valid coordinates for address '{address}' (lat: {latitude}, lon: {longitude})")
        return None, None, False

