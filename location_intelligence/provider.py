from abc import ABC, abstractmethod
from location_intelligence.models import QueryResult

class LocationDataProvider(ABC):
    """
    Abstract interface for Location Data Providers.
    Allows seamlessly plugging in different real location data sources.
    """
    
    @abstractmethod
    def search_nearby(self, latitude: float, longitude: float, radius: float) -> QueryResult:
        """
        Queries nearby food competitors and environment POIs within radius (in meters)
        from specified coordinates.
        """
        pass
