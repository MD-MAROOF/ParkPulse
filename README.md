# ParkPulse
 Map and estimate parking capacity from a place name using satellite imagery, OpenStreetMap, and YOLO vehicle detection.


## What it does

- **Geocode** a place (e.g. airport, stadium, address) and get nearby **parking polygons** from OpenStreetMap
- **Display** parking areas on an interactive **Folium map**
- **Fetch aerial/satellite imagery** (Esri World Imagery) for the largest lots
- **Detect vehicles** (YOLO) and **estimate** total capacity, free spots, and occupancy
- **Show** annotated imagery, capacity ± uncertainty, parking type (surface / multi-storey / underground), and detection confidence

## Tech stack

- **Python 3.10+**
- Geopy (Nominatim), OSMnx, GeoPandas, Contextily, Ultralytics YOLO, Streamlit, Folium, OpenCV

## Setup

1. **Clone the repo**sh
   git clone https://github.com/MD-MAROOF/ParkPulse.git
   cd ParkPulse
2. **Create a virtual environment (recommended)**

        python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows PowerShell
   # or: source .venv/bin/activate   # Linux/macOS

3. **Install dependencies**

      pip install -r requirements.txt

4. **Add the YOLO model **
Place your vehicle-detection checkpoint at models/best.pt. (Required for the app to run.)

**RUN**

   streamlit run app.py


 **PROJECT STRUCTURE**
- app.py — Streamlit app (UI + pipeline)
- parkpulse/ — Core package: geocode, OSM parking, imagery, detection, estimate, viz
- models/ — Put best.pt here (not in repo if large)
- requirements.txt — Python dependencies
