# Building Info Pipeline

A standalone Python project designed to retrieve building geometries and attributes from large-scale partition-based providers (Google Open Buildings and Microsoft Global ML Building Footprints) and retrieve corresponding free/open RGB imagery (e.g., Copernicus Sentinel-2) for requested locations. 

The pipeline filters out spurious features using multispectral indices (NDVI, NDBI, NDWI) and builds verified building footprints ready for export.

## Architectural Constraints

This project adheres to strict isolation rules:
- **Zero dependencies on other pipelines**: It does not read, import, or depend on any previous urban-fire pipelines.
- **Environment**: Relies entirely on the global Python environment.
- **No Mock Data**: Real providers only (Google and Microsoft datasets via S2 Quadkey and quadtree spatial partitions). No placeholder classes or synthetic data.
- **Scope**: Designed specifically to pull building polygons, validate them against spectral indices using Sentinel-2, and render bounding box overlays.

## Execution

Run the entry point via Python by specifying a CSV file of target events:
```bash
python main.py
```
(By default, it will look for `urban-fire-highres-pipeline/data/csv/image_metadata.csv` or you can provide the path if invoked programmatically.)

## Dependencies

Install the requirements from the provided file:
```bash
pip install -r requirements.txt
```
