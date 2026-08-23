# Building Info Pipeline

A standalone Python project designed to retrieve building geometries and attributes (via Overpass/OpenStreetMap) and corresponding free/open RGB imagery (e.g., Copernicus Sentinel-2) for requested locations.

## Architectural Constraints

This project adheres to strict isolation rules:
- **Zero dependencies on other pipelines**: It does not read, import, or depend on any previous urban-fire pipelines.
- **Environment**: Relies entirely on the global Python environment and existing root `.env`. It does not create its own `.venv` or `.env`.
- **No Mock Data**: Real providers only. No placeholder classes, fake imagery, or synthetic data.
- **Scope**: Designed specifically to pull building polygons and open imagery. It **does not** implement YOLO, object detection, population estimation, or commercial high-res imagery integrations.

## Directory Structure

- `src/models`: Pydantic/dataclass models defining exactly the expected inputs and outputs.
- `src/utils`: Utilities like consistent thread-safe logging.
- `src/config.py`: Environment configuration loading logic (safely traversing to parent `.env` if available).
- `tests/`: Unit and integration tests.
- `main.py`: Executable entry point.

## Execution

Run the entry point via Python:
```bash
python main.py --event_id 1234 --lat 34.0522 --lon -118.2437 --datetime 2026-08-22T10:00:00Z
```

## Testing

Run tests to verify the foundation:
```bash
python -m unittest discover tests/
```
