import math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class InputEvent:
    """
    Represents the strictly validated input required to start the building info pipeline.
    """
    event_id: str
    latitude: float
    longitude: float
    fire_datetime: Optional[datetime] = None

    def __post_init__(self):
        # 1. Finite numeric validation
        if not math.isfinite(self.latitude):
            raise ValueError(f"Latitude must be a finite number, got {self.latitude}")
        if not math.isfinite(self.longitude):
            raise ValueError(f"Longitude must be a finite number, got {self.longitude}")

        # 2. Strict boundary validation (no silent correction)
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Latitude {self.latitude} is out of bounds [-90, 90]")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Longitude {self.longitude} is out of bounds [-180, 180]")

        # 3. Datetime timezone-aware validation if present
        if self.fire_datetime is not None:
            if self.fire_datetime.tzinfo is None or self.fire_datetime.tzinfo.utcoffset(self.fire_datetime) is None:
                raise ValueError("fire_datetime must be timezone-aware (explicit offset/UTC)")
