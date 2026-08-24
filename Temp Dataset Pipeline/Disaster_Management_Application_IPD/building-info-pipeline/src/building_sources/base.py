from abc import ABC, abstractmethod


from src.models.aoi import AOI
from src.models.output import BuildingSearchResult

class BaseBuildingSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the building source provider."""
        pass

    @abstractmethod
    def search(self, aoi: AOI) -> BuildingSearchResult:
        """
        Retrieves building footprints intersecting the AOI.
        Must return a strictly formatted BuildingSearchResult detailing network or query failures.
        """
        pass
