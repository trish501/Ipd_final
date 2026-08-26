# Building Info Pipeline

A standalone Python project designed to retrieve building geometries and attributes from large-scale partition-based providers (Google Open Buildings and Microsoft Global ML Building Footprints) and retrieve corresponding high-resolution RGB imagery (e.g., Esri World Imagery) for requested locations.

The pipeline filters out spurious features using multispectral indices (NDVI, NDBI, NDWI) and builds verified building footprints ready for export.

## Architectural Constraints

This project adheres to strict isolation rules:
- **Zero dependencies on other pipelines**: It does not read, import, or depend on any previous urban-fire pipelines.
- **Environment**: Relies entirely on the global Python environment.
- **No Mock Data**: Real providers only (Google and Microsoft datasets via S2 Quadkey and quadtree spatial partitions). No placeholder classes or synthetic data.
- **Scope**: Designed specifically to pull building polygons, calculate exact dimensions (length, width, area), and render bounding box overlays on high-res imagery.

## Execution

Run the entry point via Python by specifying coordinates interactively or via CLI arguments:
```bash
python main.py
```

## Dependencies

Install the requirements from the provided file:
```bash
pip install -r requirements.txt
```
