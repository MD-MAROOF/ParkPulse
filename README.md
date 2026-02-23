# ParkPulse
 Map and estimate parking capacity from a place name using satellite imagery, OpenStreetMap, and YOLO vehicle detection.


# 🚗 ParkPulse

> Estimating Parking Capacity & Occupancy from Aerial Imagery

Urban infrastructure planning relies heavily on accurate parking data, yet obtaining real-time or up-to-date parking capacity and utilization information is often expensive and manual. Airports, stadiums, and large commercial hubs operate massive parking facilities without transparent visibility into usage patterns.

**ParkPulse** explores a simple but powerful question:

> Can we estimate parking capacity and occupancy using only aerial imagery and publicly available geospatial data?

The goal is to build a scalable, automated system that eliminates manual surveys and costly sensor deployments.

---

## 📌 Overview

ParkPulse is an end-to-end **geospatial + computer vision pipeline** that:

- Accepts a place name or address
- Identifies nearby parking areas using OpenStreetMap
- Fetches high-resolution aerial imagery
- Detects parked vehicles using YOLO-based object detection
- Estimates:
  - Total parking capacity
  - Occupied spots
  - Free spots
  - Occupancy percentage
- Visualizes:
  - Parking polygons on an interactive map
  - Vehicle detections on aerial imagery
  - Area-wise and aggregated metrics

The system operates around a specific point of interest (e.g., airports, stadiums, malls) and scales automatically to multiple parking areas.
---------------------------------

**Demonstration link**: https://youtu.be/MyN5VNzvWIs?si=35XAP3DVlMQ5ZDx0



## 🏗️ System Architecture

### 1️⃣ Geospatial Mapping Layer

- Geocoding: Converts place name → latitude/longitude  
- OpenStreetMap query: Fetches parking polygons within a specified radius  
- Sorts parking areas by size and selects top-K lots  
- Renders polygons on an interactive map  

---

### 2️⃣ Aerial Imagery Pipeline

- Fetches satellite imagery (Esri World Imagery)
- Automatically retries with higher zoom if resolution is insufficient
- Crops imagery to parking polygon boundaries
- Ensures resolution is suitable for small-object detection

---

### 3️⃣ Vehicle Detection (Computer Vision)

- Uses Ultralytics YOLO
- Fine-tuned for aerial / VisDrone-style vehicle detection
- CPU-optimized tiled inference

#### Key Enhancements

- Tiled inference for small-object recall
- Global Non-Maximum Suppression (NMS) across tiles
- Size and aspect-ratio filtering to reduce false positives
- Confidence averaging
- Cap on maximum detections to prevent truncation

#### Outputs

- Bounding boxes
- Class labels
- Detection confidence scores

---

### 4️⃣ Capacity Estimation Strategy

ParkPulse combines two independent estimation approaches:

#### 📐 Area-Based Estimate
- Derived from parking lot area (m²)
- Uses assumed average area per parking spot

#### 🚘 Detection-Based Estimate
- Based on detected vehicles
- Adjusted by assumed occupancy rate

#### 🧠 Blended Estimate
The final capacity is computed using:

- Detection coverage threshold
- Weighted blending between area-based and detection-based estimates
- Uncertainty bounds (± values)

This hybrid strategy improves robustness when detection quality varies.

---

## 📊 Output Metrics

For each parking area:

- Estimated capacity
- Vehicles detected
- Estimated occupied spots
- Estimated free spots
- Estimated occupancy %
- Detection confidence level
- Parking type (from OSM)

### Aggregated Totals

- Total capacity
- Total vehicles
- Total free spots
- Overall occupancy %

---

## ⚠️ Challenges

- Detecting small vehicles in high-altitude aerial imagery
- Balancing precision vs recall
- Handling very large parking lots without memory overflow
- Preventing detection truncation from model limits
- Ensuring acceptable runtime on CPU-only systems
- Avoiding capacity distortion from imperfect detection

---

## 🏆 Accomplishments

- Built a fully automated end-to-end geospatial + CV pipeline
- Improved recall using tiled inference
- Designed a blended capacity estimation strategy
- Added uncertainty bounds for transparency
- Optimized inference to run on CPU
- Scaled across multiple parking areas around major POIs

---

## 📚 Key Learnings

- Small-object detection requires tiling and resolution management
- Pure detection-based capacity estimation can be unstable
- Blending independent estimators improves robustness
- Post-processing (NMS, filtering) is as important as model accuracy
- Geospatial data quality significantly affects downstream results
- Interpretability and uncertainty estimation is critical in infrastructure tools

---

## 🚀 Future Improvements

- GPU acceleration support
- Real-time monitoring integration
- Multi-temporal imagery analysis
- Improved spot-level segmentation
- API deployment for infrastructure dashboards

---


## Tech stack

- **Python 3.10+**
- Geopy (Nominatim), OSMnx, GeoPandas, Contextily, Ultralytics YOLO, Streamlit, Folium, OpenCV

## Setup

1. **Clone the repo**sh
   git clone https://github.com/MD-MAROOF/ParkPulse.git
   
   cd ParkPulse
3. **Create a virtual environment (recommended)**

        python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows PowerShell
   # or: source .venv/bin/activate   # Linux/macOS

4. **Install dependencies**

      pip install -r requirements.txt

5. **Add the YOLO model **
Place your vehicle-detection checkpoint at models/best.pt. (Required for the app to run.)

**RUN**

   streamlit run app.py



