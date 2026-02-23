"""
ParkPulse Streamlit app: parking capacity estimation from place name.
"""

from __future__ import annotations

import streamlit as st
from streamlit_folium import folium_static
import geopandas as gpd
import numpy as np
import pandas as pd

from parkpulse.geocode import geocode_place
from parkpulse.osm_parking import get_parking_polygons
from parkpulse.imagery import fetch_aerial_mosaic
from parkpulse.detect import load_model, detect_cars
from parkpulse.estimate import (
    estimate_spots_from_cars,
    estimate_spots_from_area,
)
from parkpulse.viz import draw_detections


# ----- Cached helpers -----

@st.cache_data(ttl=3600)
def _cached_geocode(place: str) -> tuple[float, float]:
    return geocode_place(place)


@st.cache_data(ttl=3600)
def _cached_parking_polygons(lat: float, lon: float, radius_m: int) -> gpd.GeoDataFrame:
    return get_parking_polygons(lat, lon, radius_m)


@st.cache_data(ttl=3600)
def _cached_aerial_mosaic(
    west: float, south: float, east: float, north: float,
    zoom: int,
    source: str = "Esri.WorldImagery",
) -> np.ndarray:
    return fetch_aerial_mosaic((west, south, east, north), zoom=zoom, source=source)


# ----- Page -----

st.set_page_config(page_title="ParkPulse", page_icon="🅿️", layout="wide")

st.title("ParkPulse")
st.markdown(
    "Estimate parking capacity from a **place name**: we geocode the place, find nearby parking areas on OpenStreetMap, "
    "fetch aerial imagery for the largest lots, run vehicle detection (YOLO), and combine car counts with area-based estimates."
)

st.divider()

place = st.text_input(
    "Place name or address",
    value="Hartsfield–Jackson Atlanta International Airport",
    help="Geocode this place to get the search center.",
    key="place_input",
)
radius_m = st.slider("Search radius (m)", min_value=500, max_value=5000, value=2500, step=100)
k = st.slider("Number of largest parking areas to analyze (K)", min_value=1, max_value=8, value=2)

# Reliability/blending: constants only (no UI for judges)
min_coverage = 0.05
blend_strength = 0.6

# Advanced: tuning knobs (hidden unless needed)
zoom = 19
occupancy = 0.6
det_conf = 0.05
imgsz = 1536
tile = 1024
overlap = 256
auto_zoom_retry = True
min_img_height = 2048
show_reliability_debug = False

with st.expander("Advanced", expanded=False):
    zoom = st.slider("Imagery zoom level (base)", min_value=17, max_value=20, value=19)
    occupancy = st.slider(
        "Assumed occupancy (for spot estimate from cars)",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.1,
    )
    det_conf = st.slider(
        "Detection confidence (lower helps aerial cars)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )
    imgsz = st.select_slider(
        "YOLO imgsz (bigger detects small cars, slower)",
        options=[960, 1280, 1536, 1792, 2048],
        value=1536,
    )
    tile = st.select_slider("Tile size", options=[768, 1024, 1280, 1536, 1792], value=1024)
    overlap = st.select_slider("Tile overlap", options=[128, 256, 384, 512], value=256)
    auto_zoom_retry = st.checkbox("Auto retry imagery at higher zoom if image is too small", value=True)
    min_img_height = st.select_slider(
        "Min acceptable image height (px)",
        options=[1024, 1536, 2048, 2560],
        value=2048,
    )
    show_reliability_debug = st.checkbox("Debug: show detection coverage & blend note", value=False)

run = st.button("Run", type="primary")
if not run:
    st.stop()


# ----- Run pipeline -----

with st.spinner("Geocoding place…"):
    try:
        lat, lon = _cached_geocode(place)
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        st.stop()
    st.success(f"Coordinates: {lat:.5f}, {lon:.5f}")

with st.spinner("Fetching parking polygons from OpenStreetMap…"):
    try:
        gdf = _cached_parking_polygons(lat, lon, radius_m)
    except Exception as e:
        st.error(f"Failed to fetch parking data: {e}")
        st.stop()

# ---- DEBUG: how many polygons did we get? ----
st.write("len(gdf):", len(gdf))

if gdf is None or gdf.empty:
    st.warning("No parking polygons found in this area. Try a larger radius or another place.")
    st.stop()

# Sort by area desc, take top K
gdf_sorted = gdf.sort_values("area_m2", ascending=False).head(k).reset_index(drop=True)


st.write("len(gdf_sorted):", len(gdf_sorted))

# show the top-K areas so you can verify what they are
st.write("Top-K parking polygons by area (m²):")
st.write(gdf_sorted[["area_m2"]])

# label each polygon so it shows distinctly on the map
gdf_sorted = gdf_sorted.copy()
gdf_sorted["area_rank"] = [f"Area {i+1}" for i in range(len(gdf_sorted))]

st.subheader("Parking areas on map")
try:
    # color by rank (categorical) so overlapping polygons are easier to see
    m = gdf_sorted.explore(
        column="area_rank",
        categorical=True,
        tooltip=["area_rank", "area_m2"],
        popup=True,
        name="Top parking areas",
    )
    folium_static(m, width=None, height=400)
except Exception as e:
    st.warning(f"Could not draw map: {e}")

# Load YOLO once
with st.spinner("Loading YOLO model…"):
    try:
        
        load_model("models/best.pt")
    except Exception as e:
        st.error(f"Failed to load detection model: {e}")
        st.stop()

# ---- UPDATED TOTALS ----
total_cars = 0
total_capacity = 0
total_free = 0

st.subheader(f"Top {k} parking areas: imagery, detections, and estimates")

for i, (_, row) in enumerate(gdf_sorted.iterrows()):
    geom = row.geometry
    area_m2 = float(row["area_m2"])
    west, south, east, north = geom.bounds

    with st.expander(f"Area {i+1} — {area_m2:,.0f} m²", expanded=True):
        col_img, col_metrics = st.columns([2, 1])

        # --- imagery fetch with auto zoom retry ---
        used_zoom = zoom
        with st.spinner(f"Fetching imagery for area {i+1}…"):
            try:
                img = _cached_aerial_mosaic(west, south, east, north, zoom=used_zoom)
                if auto_zoom_retry and img.shape[0] < min_img_height and used_zoom < 20:
                    st.caption(f"Image height {img.shape[0]}px is small → retrying at zoom {used_zoom + 1}…")
                    used_zoom = used_zoom + 1
                    img = _cached_aerial_mosaic(west, south, east, north, zoom=used_zoom)
            except Exception as e:
                st.error(f"Imagery failed: {e}")
                continue

        st.caption(f"Fetched image shape: {img.shape} (zoom used: {used_zoom})")

        # --- detection ---
        try:
            detections = detect_cars(
                img,
                conf=det_conf,
                imgsz=int(imgsz),
                tile=int(tile),
                overlap=int(overlap),
                max_det=5000,
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            detections = []

        n_cars = len(detections)
        spots_from_cars = estimate_spots_from_cars(n_cars, occupancy=occupancy)
        spots_from_area = estimate_spots_from_area(area_m2)

        # Average detection confidence (for "is the model confident?")
        avg_conf = (sum(d["conf"] for d in detections) / len(detections)) if detections else 0.0
        if avg_conf >= 0.7:
            conf_label = "high"
        elif avg_conf >= 0.4:
            conf_label = "medium"
        else:
            conf_label = "low"

        # Capacity uncertainty: ± half-spread between the two estimates, or ±20% of capacity, whichever is larger (min 1)
        spread = abs(spots_from_area - spots_from_cars)
        capacity_uncertainty = max(1, round(max(spread / 2, 0.2 * (spots_from_area + spots_from_cars) / 2)))

        # Parking type from OSM (surface / multi-storey / underground)
        parking_type = None
        if "parking" in row.index and pd.notna(row.get("parking")):
            parking_type = str(row["parking"]).strip().lower()
        if not parking_type or parking_type == "nan":
            parking_type = "—"

        # detection reliability gating + weighted blending
        coverage = (n_cars / spots_from_area) if spots_from_area > 0 else 0.0

        if n_cars == 0 or coverage < float(min_coverage):
            combined = spots_from_area
            reliability_note = "Detection unreliable → using area-based estimate"
            weight_cars = 0.0
        else:
            # Ramp cars weight from 0 at min_coverage to blend_strength near occupancy
            denom = max(1e-6, (occupancy - float(min_coverage)))
            ramp = min(1.0, max(0.0, (coverage - float(min_coverage)) / denom))
            weight_cars = float(blend_strength) * ramp
            combined = round((1 - weight_cars) * spots_from_area + weight_cars * spots_from_cars)
            reliability_note = f"Detection OK → blended (cars weight={weight_cars:.2f})"

        # ---- NEW: occupancy/free derived metrics ----
        capacity = int(max(0, combined))
        occupied = int(max(0, n_cars))
        free_spots = int(max(0, capacity - occupied))
        occupancy_pct = (occupied / capacity) if capacity > 0 else 0.0

        # ---- UPDATED TOTALS ----
        total_cars += occupied
        total_capacity += capacity
        total_free += free_spots

        with col_img:
            annotated = draw_detections(img, detections)
            st.image(annotated, use_container_width=True, channels="RGB")

        with col_metrics:
            st.metric("Vehicles detected", n_cars)
            st.metric("Est. spots (from area)", spots_from_area)
            st.metric("Combined estimate (capacity)", f"{capacity} (±{capacity_uncertainty})")
            st.metric("Est. free spots", free_spots)
            st.metric("Est. occupancy", f"{occupancy_pct*100:.1f}%")
            st.caption(f"Area: {area_m2:,.0f} m²")
            st.caption(f"**Parking type (OSM):** {parking_type}")
            st.caption(f"**Detection confidence:** {conf_label} ({avg_conf:.2f})")

            if show_reliability_debug:
                st.metric("Detection coverage (cars / area spots)", f"{coverage:.3f}")
                st.caption(reliability_note)

st.divider()
st.subheader("Totals across analyzed areas")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total vehicles detected", total_cars)
with c2:
    st.metric("Total estimated capacity (spots)", total_capacity)
with c3:
    st.metric("Total estimated free spots", total_free)
with c4:
    overall_occ = (total_cars / total_capacity) if total_capacity > 0 else 0.0
    st.metric("Overall estimated occupancy", f"{overall_occ*100:.1f}%")