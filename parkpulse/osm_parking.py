"""Fetch parking polygons from OpenStreetMap via OSMnx."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import osmnx as ox


# Tags for parking features (union of matches).
PARKING_TAGS = {
    "amenity": "parking",
    "landuse": "parking",
    "parking": ["surface", "multi-storey", "underground"],
}


def get_parking_polygons(
    lat: float,
    lon: float,
    radius_m: int,
) -> gpd.GeoDataFrame:
    """Get parking polygons within radius of a point from OpenStreetMap.

    Uses osmnx to fetch features with parking-related tags, keeps only
    polygon geometries, fixes invalid geometries (buffer(0) when needed),
    and returns in EPSG:4326 with an added area_m2 column (area computed
    in EPSG:3857).

    Args:
        lat: Center latitude (WGS84).
        lon: Center longitude (WGS84).
        radius_m: Search radius in meters.

    Returns:
        GeoDataFrame in EPSG:4326 with polygon geometry and an 'area_m2'
        column. May be empty if no parking polygons are found.
    """
    center_point = (lat, lon)
    gdf = ox.features.features_from_point(center_point, PARKING_TAGS, dist=radius_m)

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs="EPSG:4326")

    # Ensure we have a geometry column and CRS
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Keep only polygon geometries (exclude points, lines)
    geom_types = gdf.geometry.geom_type
    poly_mask = geom_types.isin(("Polygon", "MultiPolygon"))
    gdf = gdf.loc[poly_mask].copy()

    if gdf.empty:
        gdf["area_m2"] = gpd.GeoSeries(dtype=float)
        return gdf

    # Fix invalid geometries with buffer(0)
    def fix_geom(geom):
        if geom is None or geom.is_empty:
            return geom
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom

    gdf["geometry"] = gdf.geometry.apply(fix_geom)
    # Drop rows that became empty/invalid after fix
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    if gdf.empty:
        gdf["area_m2"] = gpd.GeoSeries(dtype=float)
        return gdf

    # Compute area in meters by projecting to EPSG:3857
    gdf_3857 = gdf.to_crs("EPSG:3857")
    gdf["area_m2"] = gdf_3857.geometry.area

    return gdf
