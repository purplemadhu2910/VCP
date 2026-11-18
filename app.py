"""
=============================================================================
COMMERCIAL VEHICLE TRANSPORTATION UNIT RATE PREDICTION PORTAL
Enterprise Freight Rate Estimation, Routing & Quotation System
=============================================================================
Champion Model: Gradient Boosting Regressor (R² = 98.57%, MAE = ₹502.19)
Trained on: 10,000 Verified Commercial Logistics Trips Across India
Layout: 60% Enterprise Input & Quotation Form | 40% Interactive Route Map
=============================================================================
"""

import os
import math
import hashlib
from datetime import date, datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor

import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# 1. Page Configuration & Professional Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Commercial Freight Rate Engine | Enterprise Quotation Portal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Enterprise CSS Theme (Light Clean Background, Deep Navy & Royal Blue Accents)
st.markdown("""
<style>
    /* Global Background & Typography */
    .main {
        background-color: #f8fafc;
    }
    
    /* Top Enterprise Header */
    .enterprise-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
        padding: 22px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
    }
    .enterprise-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #ffffff;
    }
    .enterprise-subtitle {
        font-size: 13.5px;
        color: #93c5fd;
        margin-top: 5px;
        margin-bottom: 0;
    }
    .status-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid #10b981;
        color: #a7f3d0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 600;
    }

    /* Section Container Cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        padding: 12px 14px !important;
        margin-bottom: 14px;
        box-sizing: border-box !important;
    }
    .corridor-banner {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 12px;
        margin-bottom: 4px;
        box-sizing: border-box;
        width: 100%;
        max-width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px 12px;
    }
    .section-title {
        font-size: 15.5px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-desc {
        font-size: 12px;
        font-weight: 500;
        color: #ffffff !important;
        background: #0f172a;
        border-radius: 6px;
        padding: 7px 12px;
        margin-bottom: 14px;
        line-height: 1.4;
    }
    @media (prefers-color-scheme: dark) {
        .section-title {
            color: #ffffff !important;
        }
        .quote-title {
            color: #ffffff !important;
        }
        .quote-val-card {
            background: #1e293b !important;
            border-color: #334155 !important;
        }
        .quote-val-num {
            color: #f8fafc !important;
        }
    }
    
    /* Route Summary Card (Under Map) */
    .route-summary-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .route-header-tag {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #2563eb;
        margin-bottom: 6px;
    }

    /* Results Quotation Box */
    .quote-box {
        background: linear-gradient(to right bottom, #ffffff, #f8fafc);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 22px;
        margin-top: 20px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.12);
    }
    .quote-id {
        font-family: monospace;
        font-weight: 700;
        color: #1e40af;
        background: #eff6ff;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .quote-val-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .quote-val-title {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .quote-val-num {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 3px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Load or Auto-Sync Champion Model Assets
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "champion_model.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "y_train.csv")

@st.cache_resource
def load_assets():
    expected_cols = pd.read_csv(SCHEMA_PATH, nrows=1).columns.tolist()
    
    # Try loading existing pickle; if environment version mismatch occurs, auto-train cleanly
    try:
        model = joblib.load(MODEL_PATH)
        # Test a dummy prediction to verify health
        dummy_x = pd.DataFrame(0, index=[0], columns=expected_cols)
        model.predict(dummy_x)
    except Exception:
        # Auto-train champion model in active environment (~3 seconds)
        X_tr = pd.read_csv(SCHEMA_PATH)
        y_tr = pd.read_csv(Y_TRAIN_PATH).values.ravel()
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        joblib.dump(model, MODEL_PATH)
        
    return model, expected_cols

model, expected_cols = load_assets()

# -----------------------------------------------------------------------------
# 3. Master Data Constants (GPS Coordinates & Realistic Indian Hubs)
# -----------------------------------------------------------------------------
CITY_COORDINATES = {
    "Ahmedabad": (23.0225, 72.5714),
    "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Indore": (22.7196, 75.8577),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Ludhiana": (30.9010, 75.8573),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Patna": (25.5941, 85.1376),
    "Pune": (18.5204, 73.8567),
    "Rajkot": (22.3039, 70.8022),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Visakhapatnam": (17.6868, 83.2185),
    "Kanpur": (26.4499, 80.3319),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Meerut": (28.9845, 77.7064),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Solapur": (17.6599, 75.9064),
    "Amritsar": (31.6340, 74.8723),
    "Jalandhar": (31.3260, 75.5762),
    "Jodhpur": (26.2389, 73.0243),
    "Udaipur": (24.5854, 73.7125),
    "Kota": (25.2138, 75.8648),
    "Gwalior": (26.2183, 78.1828),
    "Jabalpur": (23.1815, 79.9864),
    "Raipur": (21.2514, 81.6296),
    "Bhubaneswar": (20.2961, 85.8245),
    "Cuttack": (20.4625, 85.8828),
    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    "Guwahati": (26.1445, 91.7362),
    "Dehradun": (30.3165, 78.0322),
    "Mysore": (12.2958, 76.6394),
    "Hubli": (15.3647, 75.1240),
    "Mangalore": (12.9141, 74.8560),
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kozhikode": (11.2588, 75.7804),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Vijayawada": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Warangal": (17.9689, 79.5941),
    "Tirupati": (13.6288, 79.4192),
    "Noida": (28.5355, 77.3910),
    "Gurgaon": (28.4595, 77.0266),
    "Ghaziabad": (28.6692, 77.4538),
    "Faridabad": (28.4089, 77.3178),
    "Navi Mumbai": (19.0330, 73.0297),
    "Thane": (19.2183, 72.9781),
    "Gandhinagar": (23.2156, 72.6369),
    "Bhavnagar": (21.7645, 72.1519),
    "Jamnagar": (22.4707, 70.0577),
    "Junagadh": (21.5222, 70.4579),
    "Anand": (22.5645, 72.9289),
    "Bharuch": (21.7051, 72.9959),
    "Vapi": (20.3893, 72.9106),
    "Valsad": (20.5992, 72.9342),
    "Silvassa": (20.2763, 73.0083),
    "Daman": (20.3974, 72.8328)
}

MAJOR_INDIAN_CITIES = sorted(list(CITY_COORDINATES.keys()))

DEFAULT_CITY_DISTANCES = {
    ("Ahmedabad", "Mumbai"): 530.0,
    ("Mumbai", "Ahmedabad"): 530.0,
    ("Ahmedabad", "Surat"): 260.0,
    ("Surat", "Ahmedabad"): 260.0,
    ("Ahmedabad", "Vadodara"): 110.0,
    ("Vadodara", "Ahmedabad"): 110.0,
    ("Ahmedabad", "Delhi"): 940.0,
    ("Delhi", "Ahmedabad"): 940.0,
    ("Mumbai", "Delhi"): 1420.0,
    ("Delhi", "Mumbai"): 1420.0,
    ("Mumbai", "Pune"): 150.0,
    ("Pune", "Mumbai"): 150.0,
    ("Bengaluru", "Chennai"): 350.0,
    ("Chennai", "Bengaluru"): 350.0,
    ("Bengaluru", "Hyderabad"): 570.0,
    ("Hyderabad", "Bengaluru"): 570.0,
    ("Delhi", "Jaipur"): 280.0,
    ("Jaipur", "Delhi"): 280.0,
    ("Delhi", "Lucknow"): 550.0,
    ("Lucknow", "Delhi"): 550.0,
    ("Delhi", "Kolkata"): 1500.0,
    ("Kolkata", "Delhi"): 1500.0,
    ("Indore", "Bhopal"): 195.0,
    ("Bhopal", "Indore"): 195.0,
    ("Nagpur", "Mumbai"): 820.0,
    ("Mumbai", "Nagpur"): 820.0,
    ("Vadodara", "Surat"): 155.0,
    ("Surat", "Vadodara"): 155.0
}

def resolve_city_coords(query):
    """Case-insensitive and substring lookup for city GPS coordinates."""
    if not query or not isinstance(query, str):
        return None
    q = query.strip().lower()
    for c, coords in CITY_COORDINATES.items():
        if c.lower() == q:
            return coords
    for c, coords in CITY_COORDINATES.items():
        if c.lower() in q or q in c.lower():
            return coords
    return None

def haversine_distance(coord1, coord2):
    """Calculates aerial great-circle distance between two GPS coordinates in kilometers."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def calculate_route_distance(origin_str, dest_str):
    """
    Computes highway road distance between two typed locations:
    1. Direct table match if available
    2. Haversine aerial distance with 1.22x Indian national highway curvature factor
    3. Safe default fallback
    """
    if not origin_str or not dest_str:
        return 500.0
    o = origin_str.strip().title()
    d = dest_str.strip().title()
    if o.lower() == d.lower():
        return 15.0
    if (o, d) in DEFAULT_CITY_DISTANCES:
        return DEFAULT_CITY_DISTANCES[(o, d)]
    if (d, o) in DEFAULT_CITY_DISTANCES:
        return DEFAULT_CITY_DISTANCES[(d, o)]
    c1 = resolve_city_coords(origin_str)
    c2 = resolve_city_coords(dest_str)
    if c1 and c2:
        aerial = haversine_distance(c1, c2)
        road_km = round(aerial * 1.22, 0)
        return max(15.0, road_km)
    return 500.0

# -----------------------------------------------------------------------------
# 4. Enterprise Top Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="enterprise-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 class="enterprise-title">Commercial Freight Rate Engine & Interactive Route Portal</h1>
            <p class="enterprise-subtitle">AI-Powered Unit Rate Quotation Engine with Real-Time Inter-State Corridor Mapping</p>
        </div>
        <div style="text-align: right;">
            <span class="status-badge">&bull; ML Production Engine Online</span>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 5px;">Model: Gradient Boosting (R² 98.6% | MAE ₹502)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Two-Column Dashboard Layout (60% Form | 40% Map)
# -----------------------------------------------------------------------------
col_form, col_map = st.columns([1.5, 1.0], gap="large")

# =============================================================================
# LEFT COLUMN: USER INPUT FORM & QUOTATION RESULTS (60%)
# =============================================================================
with col_form:
    
    # --- SECTION A: ROUTE & TRIP DETAILS ---
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">SECTION A: Route & Transit Details</div>
        <div class="section-desc">Type pickup and delivery locations to automatically calculate line-haul distance and highway corridor transit metrics.</div>
        """, unsafe_allow_html=True)
        
        loc_c1, loc_c2 = st.columns(2)
        with loc_c1:
            from_location = st.text_input(
                "From Location (Origin Pickup Hub)",
                value="Ahmedabad",
                placeholder="e.g. Ahmedabad, Surat, Vadodara, Delhi...",
                help="Type the pickup city or loading facility name."
            ).strip()
        with loc_c2:
            to_location = st.text_input(
                "To Location (Destination Delivery Hub)",
                value="Mumbai",
                placeholder="e.g. Mumbai, Pune, Delhi, Bengaluru...",
                help="Type the drop-off city or delivery warehouse name."
            ).strip()

        # Validation: Check if empty or same location
        is_empty_location = not from_location or not to_location
        is_same_city = (from_location.lower() == to_location.lower()) and not is_empty_location
        
        if is_empty_location:
            st.warning("Please type both Origin and Destination locations.")
        elif is_same_city:
            st.error("**Validation Alert:** Origin and Destination cannot be the same city. Please enter different locations for inter-city freight transit.")

        # Calculate Distance automatically from typed locations
        auto_dist = calculate_route_distance(from_location, to_location)
        
        # Distance numerical input pre-populated with calculated distance
        dist_c1, dist_c2 = st.columns([1.2, 1.0])
        with dist_c1:
            distance = st.number_input(
                "Distance (Km)",
                min_value=5.0,
                max_value=6000.0,
                value=float(auto_dist),
                step=10.0,
                help="Calculated highway route distance between locations (editable)."
            )
        with dist_c2:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-top: 24px; text-align: center;">
                <span style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block;">Calculated Road Distance</span>
                <span style="font-size: 18px; font-weight: 800; color: #1e40af;">{distance:,.0f} Km</span>
            </div>
            """, unsafe_allow_html=True)
            
        # Automatic line-haul calculations (transit days and inquiry date are handled in background)
        calc_days = max(1, int(round(distance / 350.0)))
        transit_days = calc_days
        created_date = date.today()
        
        # Active Route Banner (Strictly contained within Section A container)
        from_display = from_location.title() if from_location else "Origin"
        to_display = to_location.title() if to_location else "Destination"
        st.markdown(f"""
        <div class="corridor-banner">
            <div style="font-weight: 700; color: #1e40af; font-size: 13.5px; word-break: break-word;">
                Corridor: {from_display} &nbsp;➔&nbsp; {to_display}
            </div>
            <div style="font-size: 12px; color: #2563eb; font-weight: 600; background: #ffffff; border: 1px solid #dbeafe; padding: 4px 10px; border-radius: 6px; white-space: nowrap;">
                {distance:,.0f} Km &bull; {transit_days} Days Transit
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- SECTION B: VEHICLE & CARGO SPECIFICATIONS ---
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">SECTION B: Vehicle & Cargo Specifications</div>
        <div class="section-desc">Specifies fleet category, trailer bed dimensions, and payload tonnage to evaluate capacity and space utilization.</div>
        """, unsafe_allow_html=True)
        
        v_c1, v_c2 = st.columns(2)
        with v_c1:
            vehicle_cat = st.selectbox(
                "Vehicle Category",
                options=["Truck", "SLBT", "LCV", "LBT", "Taurus", "HBT", "LPT"],
                index=0,
                help="Vehicle classification (Truck: Multi-Axle, Taurus: 16-25T, SLBT/LBT: Low Bed Trailers, LCV: Light Commercial)."
            )
        with v_c2:
            odc_flag = st.radio(
                "Cargo Dimension Classification",
                options=["Non-ODC", "ODC"],
                index=0,
                horizontal=True,
                help="Over Dimensional Cargo (ODC) exceeds standard trailer boundaries and requires special transit permits."
            )
            
        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #475569; margin-top: 6px; margin-bottom: 2px;'>Trailer Bed Dimensions (Feet):</div>", unsafe_allow_html=True)
        dim1, dim2, dim3 = st.columns(3)
        with dim1:
            length = st.number_input("Length (ft)", min_value=10.0, max_value=60.0, value=32.0, step=1.0)
        with dim2:
            height = st.number_input("Height (ft)", min_value=4.0, max_value=15.0, value=8.0, step=0.5)
        with dim3:
            width = st.number_input("Width (ft)", min_value=4.0, max_value=15.0, value=8.0, step=0.5)
            
        calc_volume = round(length * height * width, 2)
        st.caption(f"Calculated 3D Cargo Space: **{calc_volume:,.1f} cu.ft.**")
        
        cap_c1, cap_c2 = st.columns(2)
        with cap_c1:
            capacity = st.number_input("Vehicle Capacity (MT)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
        with cap_c2:
            uom = st.selectbox("Unit of Measure (Vehicle_UOM)", options=["MT", "Nos"], index=0)
    
    # --- SECTION C: COMMERCIAL & VENDOR DETAILS ---
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">SECTION C: Commercial & Vendor Details</div>
        <div class="section-desc">Captures consignment shipment quantity, contracted 3PL transporter, and current diesel benchmark price for fuel indexation.</div>
        """, unsafe_allow_html=True)
        
        com_c1, com_c2 = st.columns(2)
        with com_c1:
            quantity = st.number_input(
                f"Consignment Quantity ({uom})",
                min_value=0.5,
                max_value=100.0,
                value=18.0,
                step=1.0,
                help="Total volume or count of freight tendered for shipment."
            )
        with com_c2:
            vendor_name = st.selectbox(
                "Transporter / 3PL Partner",
                options=[
                    "TCI Freight", "VRL Logistics", "Rivigo Services", "Delhivery Freight",
                    "Safexpress Logistics", "Gati Transport", "Shree Maruti Courier", "Agarwal Packers"
                ],
                index=0,
                help="Primary commercial freight provider or 3PL contractor."
            )
            
        fuel_price = st.number_input(
            "Diesel Benchmark Price (Rs./Ltr)",
            min_value=70.0,
            max_value=120.0,
            value=89.50,
            step=0.50,
            help="Official state diesel retail price benchmark for fuel surcharge indexation."
        )
    
    # Rate Calculation Button
    calc_button = st.button(
        "Calculate Fair Market Rate & Generate Quotation",
        type="primary",
        use_container_width=True,
        disabled=(is_same_city or is_empty_location)
    )

# =============================================================================
# RIGHT COLUMN: INTERACTIVE ROUTE MAP & CORRIDOR SUMMARY (40%)
# =============================================================================
with col_map:
    with st.container(border=True):
        st.markdown("""
        <div class="section-title">Interactive Freight Corridor Map</div>
        <div class="section-desc">Live geographic visualization of origin pickup and destination delivery terminals along national highway transit corridors.</div>
        """, unsafe_allow_html=True)
        
        # Retrieve coordinates for origin and destination
        origin_coords = resolve_city_coords(from_location) or (23.0225, 72.5714)
        dest_coords = resolve_city_coords(to_location) or (19.0760, 72.8777)
        
        # Check if Google Maps API key is configured in environment
        gmaps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
        
        if gmaps_api_key:
            # Use Google Maps Directions Embed API
            embed_url = f"https://www.google.com/maps/embed/v1/directions?key={gmaps_api_key}&origin={from_location},India&destination={to_location},India&mode=driving"
            st.components.v1.iframe(embed_url, height=450, scrolling=False)
        else:
            # Fallback to high-performance Folium with OpenStreetMap
            mid_lat = (origin_coords[0] + dest_coords[0]) / 2.0
            mid_lon = (origin_coords[1] + dest_coords[1]) / 2.0
            
            m = folium.Map(
                location=[mid_lat, mid_lon],
                zoom_start=5,
                tiles="OpenStreetMap",
                control_scale=True
            )
            
            # 1. Origin Marker (Green Pickup Hub)
            folium.Marker(
                location=origin_coords,
                popup=folium.Popup(f"<b>Origin: {from_display}</b><br>Loading Terminal", max_width=200),
                tooltip=f"Origin: {from_display}",
                icon=folium.Icon(color="green", icon="play", prefix="fa")
            ).add_to(m)
            
            # 2. Destination Marker (Red Delivery Hub)
            folium.Marker(
                location=dest_coords,
                popup=folium.Popup(f"<b>Destination: {to_display}</b><br>Delivery Warehouse", max_width=200),
                tooltip=f"Destination: {to_display}",
                icon=folium.Icon(color="red", icon="flag", prefix="fa")
            ).add_to(m)
            
            # 3. Route Line (Connecting Source & Destination)
            if not is_same_city and not is_empty_location:
                folium.PolyLine(
                    locations=[origin_coords, dest_coords],
                    color="#2563eb",
                    weight=4.5,
                    opacity=0.85,
                    dash_array="8, 6",
                    tooltip=f"Transit Route: {from_display} ➔ {to_display} ({distance:,.0f} Km)"
                ).add_to(m)
                
                # Fit map view bounds dynamically to fit both markers
                m.fit_bounds([origin_coords, dest_coords], padding=(40, 40))
            
            # Render map in Streamlit (returned_objects=[] avoids unnecessary script reruns on pan/zoom)
            st_folium(m, height=440, width="100%", returned_objects=[])
    
    # Route Corridor Summary Card Below Map
    daily_velocity = round(distance / max(1, transit_days), 1)
    st.markdown(f"""
    <div class="route-summary-card">
        <div class="route-header-tag">Corridor Intelligence Summary</div>
        <div style="font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">
            {from_display} &nbsp;➔&nbsp; {to_display}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Corridor Distance</div>
                <div style="font-size: 17px; font-weight: 800; color: #1e40af;">{distance:,.0f} Km</div>
            </div>
            <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Transit Duration</div>
                <div style="font-size: 17px; font-weight: 800; color: #047857;">{transit_days} Days</div>
            </div>
            <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Daily Travel Intensity</div>
                <div style="font-size: 17px; font-weight: 800; color: #b45309;">{daily_velocity:,.1f} Km/day</div>
            </div>
            <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Carrier Partner</div>
                <div style="font-size: 14px; font-weight: 700; color: #334155; margin-top: 3px;">{vendor_name}</div>
            </div>
        </div>
        <div style="font-size: 11px; color: #64748b; margin-top: 10px; line-height: 1.4;">
            <b>Operational Protocol:</b> Line-haul transit route follows national highway freight corridors. Map pins and polyline illustrate geographical dispatch coordinates.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# BOTTOM SECTION: ML PREDICTION & FORMAL FREIGHT QUOTATION
# =============================================================================
if calc_button and not is_same_city and not is_empty_location:
    # 1. Capture Raw Inquiry (Includes From_Location & To_Location)
    raw_input = {
        "From_Location": from_display,
        "To_Location": to_display,
        "Vehicle_Category": vehicle_cat,
        "Vehicle_Length": length,
        "Vehicle_Height": height,
        "Vehicle_Width": width,
        "Vehicle_Capacity": capacity,
        "ODC_Flag": odc_flag,
        "Vehicle_Transit_Days": transit_days,
        "Distance_Km": distance,
        "Fuel_Price": fuel_price,
        "Vehicle_UOM": uom,
        "Vehicle_Unit_Quantity": quantity,
        "Vendor_Name": vendor_name,
        "Created_Date": pd.to_datetime(created_date)
    }
    
    df_raw = pd.DataFrame([raw_input])
    
    # 2. Domain Feature Engineering (Strictly matches the trained pipeline)
    df_raw["Created_Month"] = df_raw["Created_Date"].dt.month
    df_raw["Vehicle_Volume"] = (df_raw["Vehicle_Length"] * df_raw["Vehicle_Height"] * df_raw["Vehicle_Width"]).round(2)
    df_raw["Capacity_per_Volume"] = (df_raw["Vehicle_Capacity"] / df_raw["Vehicle_Volume"]).round(4)
    df_raw["Km_Per_Day"] = (df_raw["Distance_Km"] / df_raw["Vehicle_Transit_Days"]).round(2)
    
    # 3. Categorical One-Hot Encoding
    categorical_cols = ["Vehicle_Category", "ODC_Flag", "Vehicle_UOM", "Vendor_Name"]
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, dtype=int)
    
    # 4. Strict Schema Alignment with expected_cols (23 Trained Features)
    X_input = pd.DataFrame(0, index=[0], columns=expected_cols)
    for col in expected_cols:
        if col in df_encoded.columns:
            X_input[col] = df_encoded[col].values[0]
            
    # 5. Model Inference (Champion Gradient Boosting Model)
    predicted_rate = float(model.predict(X_input)[0])
    predicted_rate = max(100.0, round(predicted_rate, 2))
    total_freight = round(predicted_rate * quantity, 2)
    daily_intensity = round(distance / max(1, transit_days), 1)
    
    # Generate Unique Quotation Reference ID
    quote_hash = hashlib.md5(f"{from_display}{to_display}{datetime.now()}".encode()).hexdigest()[:6].upper()
    quote_ref = f"QT-{datetime.now().year}-{quote_hash}"
    
    # 6. Render Industry Freight Quotation Result
    with st.container(border=True):
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
            <div>
                <div class="quote-title" style="font-size: 19px; font-weight: 800; color: #0f172a;">Official Commercial Freight Quotation & Rate Benchmark</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 3px;">Issued: {datetime.now().strftime('%d-%b-%Y %H:%M')} &bull; Valid for 48 Hours &bull; Commercial Baseline</div>
            </div>
            <div>
                <span class="quote-id">{quote_ref}</span>
            </div>
        </div>
        
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 12px 16px; margin-bottom: 18px; box-sizing: border-box; width: 100%; max-width: 100%; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #2563eb; letter-spacing: 0.5px; margin-bottom: 2px;">Transit Route Corridor</div>
                <div style="font-size: 15.5px; font-weight: 800; color: #1e40af;">{from_display} &nbsp;➔&nbsp; {to_display}</div>
            </div>
            <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                <div style="background: #ffffff; padding: 6px 12px; border-radius: 6px; border: 1px solid #dbeafe; text-align: center;">
                    <span style="font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block;">Distance</span>
                    <span style="font-size: 13px; font-weight: 700; color: #1e40af;">{distance:,.0f} Km</span>
                </div>
                <div style="background: #ffffff; padding: 6px 12px; border-radius: 6px; border: 1px solid #dbeafe; text-align: center;">
                    <span style="font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block;">Transit Time</span>
                    <span style="font-size: 13px; font-weight: 700; color: #047857;">{transit_days} Days</span>
                </div>
                <div style="background: #ffffff; padding: 6px 12px; border-radius: 6px; border: 1px solid #dbeafe; text-align: center;">
                    <span style="font-size: 10px; color: #64748b; font-weight: 600; text-transform: uppercase; display: block;">Vehicle Spec</span>
                    <span style="font-size: 13px; font-weight: 700; color: #334155;">{vehicle_cat} ({capacity} MT)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Financial Highlight Metrics
        res_c1, res_c2, res_c3 = st.columns(3)
        
        with res_c1:
            st.markdown(f"""
            <div class="quote-val-card">
                <div class="quote-val-title">Predicted Fair Unit Rate</div>
                <div class="quote-val-num" style="color: #1e40af;">₹{predicted_rate:,.2f}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Per {uom} shipped</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_c2:
            st.markdown(f"""
            <div class="quote-val-card">
                <div class="quote-val-title">Estimated Total Freight</div>
                <div class="quote-val-num" style="color: #047857;">₹{total_freight:,.2f}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Net Line-Haul Amount</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_c3:
            st.markdown(f"""
            <div class="quote-val-card">
                <div class="quote-val-title">Daily Travel Intensity</div>
                <div class="quote-val-num" style="color: #b45309;">{daily_intensity:,.1f}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Km covered per day</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Trip Summary Table
        st.markdown("##### Commercial & Operational Trip Summary")
        summary_data = {
            "Attribute": [
                "Origin Hub (Pickup)", "Destination Hub (Delivery)", "Line-Haul Distance", 
                "Committed Transit Time", "Vehicle Category", "Cargo Dimensions (L x H x W)", 
                "Cargo Volume", "Consignment Quantity", "Logistics Carrier Partner", 
                "Diesel Price Benchmark"
            ],
            "Value": [
                from_display, to_display, f"{distance:,.0f} Km", 
                f"{transit_days} Days", f"{vehicle_cat} ({capacity} MT Capacity)", f"{length:.0f}' x {height:.0f}' x {width:.0f}'",
                f"{calc_volume:,.1f} cu.ft.", f"{quantity} {uom}", vendor_name, 
                f"₹{fuel_price:.2f} / Ltr"
            ]
        }
        st.table(pd.DataFrame(summary_data))
        
        # Contextual Pricing Badges
        st.markdown("##### Pricing Intelligence & Surcharge Breakdown")
        if odc_flag == "ODC":
            st.warning("**Over Dimensional Cargo (ODC) Surcharge Factored:** Specialized transit permits, escort coordination, and reduced road transit velocity premium included.")
        else:
            st.info("**Standard Cargo Classification:** Standard road transit pricing rules applied without oversized regulatory surcharges.")
            
        if quantity >= 15.0:
            st.success(f"**Bulk Economies of Scale Activated:** High tender volume ({quantity} {uom}) qualifies for commercial volume discounts.")
        else:
            st.info(f"**Partial/Standard Consignment Volume:** Standard rate brackets applied for {quantity} {uom}.")
            
        if daily_intensity > 350.0:
            st.warning(f"**Express Transit Velocity ({daily_intensity:.0f} Km/day):** Multi-driver or relay transit intensity required to meet scheduled delivery timeline.")

        # Export Quotation
        csv_export = pd.DataFrame([{
            "Quote_Reference": quote_ref,
            "Quote_Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Origin": from_display,
            "Destination": to_display,
            "Distance_Km": distance,
            "Transit_Days": transit_days,
            "Vehicle_Category": vehicle_cat,
            "Vehicle_Capacity_MT": capacity,
            "Cargo_Volume_CuFt": calc_volume,
            "ODC_Flag": odc_flag,
            "Quantity": quantity,
            "UOM": uom,
            "Vendor": vendor_name,
            "Fuel_Price": fuel_price,
            "Predicted_Unit_Rate_INR": predicted_rate,
            "Estimated_Total_Freight_INR": total_freight
        }]).to_csv(index=False)
        
        st.download_button(
            label="Download Formal Quotation (CSV)",
            data=csv_export,
            file_name=f"Freight_Quote_{quote_ref}.csv",
            mime="text/csv",
            use_container_width=True
        )
