# Natural Earth 10m Urban Areas

* **Dataset name:** Natural Earth 10m Urban Areas
* **Official source:** https://naciscdn.org/naturalearth/10m/cultural/ne_10m_urban_areas.zip
* **Purpose in this project:** Used by the `urban-fire-highres-pipeline` to filter fire events, ensuring they only occur within urban areas before attempting satellite imagery download.
* **Date/source version:** v5.1.1 (or latest available at 10m scale).
* **Required companion files:** 
  - `ne_10m_urban_areas.shp`
  - `ne_10m_urban_areas.shx`
  - `ne_10m_urban_areas.dbf`
  - `ne_10m_urban_areas.prj`
  - `ne_10m_urban_areas.cpg`

**Explicit statement:** These files are production dependencies and must not be removed by cleanup scripts. They are treated as source/static production data, not generated cache or output.
