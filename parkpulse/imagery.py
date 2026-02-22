"""Fetch aerial/satellite imagery tiles."""

from __future__ import annotations

import numpy as np
import contextily as ctx
import xyzservices.providers as xyz_providers


def _get_tile_provider(source: str):
    """Resolve provider name like 'Esri.WorldImagery' to an xyzservices TileProvider."""
    obj = xyz_providers
    for part in source.split("."):
        obj = getattr(obj, part)
    return obj


def fetch_aerial_mosaic(
    bounds_wgs84: tuple[float, float, float, float],
    zoom: int = 19,
    source: str = "Esri.WorldImagery",
) -> np.ndarray:
    """Fetch aerial/satellite tiles for a WGS84 bounding box and return an RGB image.

    Uses contextily to download tiles (e.g. Esri World Imagery) and merges
    them into a single image array. The source must be an xyzservices provider
    name (e.g. "Esri.WorldImagery").

    Args:
        bounds_wgs84: Bounding box as (west, south, east, north) in WGS84.
        zoom: Tile zoom level (default 19 for high detail).
        source: xyzservices provider name (default "Esri.WorldImagery").

    Returns:
        RGB image as a numpy array of shape (height, width, 3), dtype uint8.
    """
    w, s, e, n = bounds_wgs84
    provider = _get_tile_provider(source)
    img, _extent = ctx.bounds2img(w, s, e, n, zoom=zoom, source=provider, ll=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Ensure 3-channel RGB (contextily may return RGBA)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3].copy()
    return np.asarray(img, dtype=np.uint8)
