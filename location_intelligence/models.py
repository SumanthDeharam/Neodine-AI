from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

class DataState(str, Enum):
    REAL_DATA = "REAL_DATA"
    NO_RESULTS = "NO_RESULTS"
    ZERO_RESULTS = "ZERO_RESULTS"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    GEOCODING_ERROR = "GEOCODING_ERROR"
    INVALID_LOCATION = "INVALID_LOCATION"

@dataclass
class CompetitorPOI:
    source_id: Optional[str] = None
    osm_type: Optional[str] = "node"
    osm_id: Optional[int] = None
    name: str = "Unnamed Food Outlet"
    category: str = "Dining"
    cuisine: Optional[str] = None
    address: str = "Local Area"
    lat: float = 0.0
    lng: float = 0.0
    distance: float = 0.0
    rating: Optional[float] = None  # Ratings unavailable in OSM; DO NOT FABRICATE
    reviews: int = 0
    price_level: Optional[int] = None  # Price unavailable in OSM; DO NOT FABRICATE
    opening_hours: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    source: str = "OpenStreetMap Overpass API"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EnvironmentPOI:
    type: str = "general"
    name: str = "Unknown POI"
    vicinity: str = ""
    lat: float = 0.0
    lng: float = 0.0
    distance: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    source: str = "OpenStreetMap Overpass API"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class QueryResult:
    status: DataState
    competitors: List[CompetitorPOI] = field(default_factory=list)
    pois: List[EnvironmentPOI] = field(default_factory=list)
    raw_count: int = 0
    http_status: int = 500
    error_message: Optional[str] = None
    provider_name: str = "OpenStreetMap Overpass API"
    perf_metrics: Dict[str, float] = field(default_factory=dict)
    snapshot_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'competitors': [c.to_dict() for c in self.competitors],
            'pois': [p.to_dict() for p in self.pois],
            'raw_count': self.raw_count,
            'http_status': self.http_status,
            'error_message': self.error_message,
            'provider_name': self.provider_name,
            'snapshot_date': self.snapshot_date
        }
