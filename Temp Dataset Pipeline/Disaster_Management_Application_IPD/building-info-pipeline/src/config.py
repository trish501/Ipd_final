import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class Settings:
    _instance = None
    _env_data: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_env()
        return cls._instance

    def _load_env(self):
        # Locate the repository root .env file
        # The project root is the parent directory of building-info-pipeline/
        base_dir = Path(__file__).resolve().parent.parent.parent
        env_path = base_dir / ".env"
        
        if not env_path.exists():
            logger.warning(f"No .env file found at {env_path}. Relying purely on system environment variables.")
            return

        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        self._env_data[key] = val
                        # Also inject into os.environ for other libraries if needed
                        os.environ.setdefault(key, val)
        except Exception as e:
            logger.error(f"Failed to parse .env file at {env_path}: {e}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # System environment variables take precedence
        sys_val = os.getenv(key)
        if sys_val is not None:
            return sys_val
        return self._env_data.get(key, default)

    # Specific configuration getters can be added here
    @property
    def free_only(self) -> bool:
        val = self.get("FREE_ONLY", "True").lower()
        return val in ("true", "1", "yes", "t")
        
    @property
    def aoi_buffer_meters(self) -> float:
        """
        Buffer distance in meters around the input coordinate to form the AOI.
        Defaults to 500 meters (1km x 1km bounding box).
        """
        val = self.get("AOI_BUFFER_METERS", "100.0")
        try:
            return float(val)
        except ValueError:
            logger.error(f"Invalid AOI_BUFFER_METERS value '{val}', defaulting to 100.0")
            return 100.0

    @property
    def building_source_mode(self) -> str:
        """
        FASTEST (default) or MAX_COVERAGE
        """
        return self.get("BUILDING_SOURCE_MODE", "FASTEST").upper()

    @property
    def enable_rgb(self) -> bool:
        """
        Whether to run RGB/imagery tasks. Default false.
        """
        val = self.get("ENABLE_RGB", "False").lower()
        return val in ("true", "1", "yes", "t")

    @property
    def min_structural_pixels_short_axis(self) -> float:
        return float(self.get("MIN_STRUCTURAL_PIXELS_SHORT_AXIS", "2.5"))

    @property
    def min_structural_pixels_long_axis(self) -> float:
        return float(self.get("MIN_STRUCTURAL_PIXELS_LONG_AXIS", "2.5"))

    @property
    def min_interior_valid_pixels(self) -> int:
        return int(self.get("MIN_INTERIOR_VALID_PIXELS", "6"))

    @property
    def min_coverage_ratio(self) -> float:
        return float(self.get("MIN_COVERAGE_RATIO", "0.9"))

    @property
    def image_width_meters(self) -> float:
        return float(self.get("IMAGE_WIDTH_METERS", "2000.0"))

    @property
    def image_height_meters(self) -> float:
        return float(self.get("IMAGE_HEIGHT_METERS", "2000.0"))
        
    @property
    def generate_debug_overlay(self) -> bool:
        val = self.get("GENERATE_DEBUG_OVERLAY", "False").lower()
        return val in ("true", "1", "yes", "t")

    @property
    def label_mode(self) -> str:
        """
        Label mode for verified buildings in overlay: NONE, ID_ONLY, FULL. Default: ID_ONLY.
        """
        return self.get("LABEL_MODE", "ID_ONLY").upper()

    @property
    def display_upscaling_factor(self) -> int:
        return int(self.get("DISPLAY_UPSCALING_FACTOR", "4"))

settings = Settings()
