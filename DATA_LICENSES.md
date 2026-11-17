# Data used by AreaGIS Kit and their licences

AreaGIS Kit' code is GPL-3.0-or-later. **Data keeps its own licence**, independent of the code.
AreaGIS Kit never bundles or redistributes these datasets: each run streams only the pixels for
the user's area, and every report carries the required credits.

| Dataset | Used by | Licence | Commercial use | Status |
|---|---|---|---|---|
| Impact Observatory / Microsoft / Esri 10 m Annual LULC v02 | Land, Flood, Hazard, Crop, Urban | CC BY 4.0 | Yes, with credit | Confirmed |
| ESA WorldCover 2021 | Land, Urban (validation) | CC BY 4.0 | Yes, with credit | Confirmed |
| ASF Global 30 m HAND | Flood, Hazard | CC0 1.0 (+ Copernicus DEM credit) | Yes | Confirmed (registry.opendata.aws/glo-30-hand) |
| Copernicus DEM GLO-30 | Flood, Hazard | Copernicus DEM licence: free, worldwide, attribution | Yes, with credit | Re-check before a paid release |
| Sentinel-1 RTC (Planetary Computer) | Flood | Copernicus Sentinel data terms; RTC processing by Catalyst/Microsoft | Check RTC processing terms | Re-check before a paid release |
| Sentinel-2 L2A | Crop, Landslide, Urban | Copernicus Sentinel data terms (free, full, open) | Yes, with credit | Re-check before a paid release |
| JRC Global Surface Water | Flood | Free with attribution (EC JRC) | Yes, with credit | Re-check before a paid release |
| Landsat Collection 2 Level-2 | Urban | U.S. public domain | Yes | Confirmed |
| OpenStreetMap (via Overpass) | Urban, Health | ODbL 1.0 | Yes; attribution, share-alike for derived databases | Confirmed |
| Meta / CIESIN High Resolution Settlement Layer (HRSL, 30 m population) | Urban, Health | CC BY 4.0 | Yes, with credit | Confirmed (dataforgood-fb-data, AWS Open Data) |
| Sentinel-5P TROPOMI NO₂ L2 (Planetary Computer) | Urban | Copernicus Sentinel data terms (free, full, open) | Yes, with credit | Re-check before a paid release |
| MODIS LST MOD11A1/MYD11A1/MOD11A2, NDVI MOD13Q1 (Collection 6.1) | Urban, Health, Crop | NASA EOSDIS: no restrictions on use | Yes (credit requested) | Confirmed |
| MODIS burned area MCD64A1 | Crop | NASA EOSDIS: no restrictions on use | Yes (credit requested) | Confirmed |
| USDA NASS Cropland Data Layer | Crop (validation, USA) | U.S. public domain | Yes | Confirmed |
| GHS-POP / GHS-BUILT-S (JRC, GHSL R2023A) | Landslide, LULC, Urban (check) | CC BY 4.0 | Yes, with credit | Confirmed |
| SoilGrids 250 m (ISRIC) | Landslide | CC BY 4.0 | Yes, with credit | Confirmed |
| GloFAS flood hazard maps v2.1 | Flood, Urban | ODbL 1.0 | Yes; attribution, share-alike for derived databases | Confirmed |
| Open-Meteo APIs (GloFAS river discharge and forecast; CAMS PM2.5) | Flood, Urban | Data CC BY 4.0; the free API is for non-commercial use | Commercial use needs an Open-Meteo API subscription | Confirmed — check before paid consultancy use |
| NASA POWER (rain, air temperature, humidity) | Flood, Landslide, Crop, Urban | NASA open data, no restrictions | Yes (credit requested) | Confirmed |
| Copernicus GFM via EODC STAC | Flood | Copernicus EMS terms, free with attribution | Yes, with credit | Confirmed |
| Meta Relative Wealth Index (HDX) | Urban, Flood (vulnerability) | CC BY-NC 4.0 | **No: non-commercial only** | Confirmed — switch off or replace for paid work |
| Macrostrat geologic maps (API) | Landslide (lithology) | CC BY 4.0 (Macrostrat data) | Yes, with credit | Confirmed |
| NASA POWER soil wetness (GWETROOT/GWETTOP), radiation | Crop, Urban | NASA open data | Yes (credit requested) | Confirmed |
| Global LCZ map (Demuzere et al. 2022, Zenodo) | Urban | CC BY 4.0 | Yes, with credit | Confirmed |
| LANID-US irrigation map (Zenodo) | Crop (validation, USA) | CC BY 4.0 | Yes, with credit | Confirmed |
| GEM Global Seismic Hazard Map v2023.1 (PGA 475-yr, Zenodo 8409647) | Landslide (seismic factor) | CC BY-NC-SA 4.0 | **No: non-commercial only** (GEM licence needed) | Confirmed — option `seismic` switches it off |
| USGS ShakeMap / earthquake catalogue API | Landslide | U.S. public domain | Yes | Confirmed |
| Natural Earth 1:110m land (bundled, location insets) | All | Public domain | Yes | Confirmed |
| User data (cases, boundaries, facilities, inventories) | Health, Hazard, Urban | User's own | User's responsibility | — |

Not used on purpose: WDPA protected areas (no commercial use or redistribution without UNEP-WCMC permission).
Add any new dataset to this table, with its licence checked, before it ships.

## AreaGIS Kit: Flood
- Copernicus Global Flood Monitoring (GFM), © European Union, Copernicus Emergency Management Service; read from EODC STAC (stac.eodc.eu). Free use with attribution under the CEMS terms of service.
- Copernicus EMS Rapid Mapping products, © European Union; used only for validation, with attribution.
- GloFAS global river flood hazard maps v2.1, © European Union 1995–2026, ODbL 1.0.
- JRC global flood depth–damage functions (Huizinga et al. 2017), values as published.
- NASA POWER daily precipitation (MERRA-2), public, no account: https://power.larc.nasa.gov
- GloFAS v4 river discharge reanalysis via Open-Meteo flood API, CC BY 4.0 (free for non-commercial use; commercial use needs an Open-Meteo subscription).
