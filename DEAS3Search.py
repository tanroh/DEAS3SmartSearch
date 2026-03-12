"""
DEA S3 Smart Search — Streamlit App
====================================
A portable, self-contained web app for searching Digital Earth Australia
satellite data via natural language questions.

Run with:
    streamlit run app.py

Requires ANTHROPIC_API_KEY in environment (or .env file).
"""

import os
import json
import requests
import boto3
import pandas as pd
import folium
import time
import streamlit as st
from datetime import datetime, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from streamlit_folium import st_folium
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DEA S3 Search",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
S3_BUCKET     = "dea-public-data"
S3_REGION     = "ap-southeast-2"
STAC_ENDPOINT = "https://explorer.dea.ga.gov.au/stac/"
S3_HTTP_BASE  = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# ── DEA Collection catalogue ──────────────────────────────────────────────────
DEA_COLLECTIONS = [
    {
        "id": "ga_ls8c_ard_3",
        "sensor": "Landsat 8",
        "title": "Landsat 8 ARD",
        "description": "Surface reflectance ARD. Use for vegetation, land cover, change detection, NDVI, agriculture, drought.",
        "tags": ["vegetation","land cover","change detection","NDVI","agriculture","drought","reflectance","landsat"],
        "temporal": "2013–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls9c_ard_3",
        "sensor": "Landsat 9",
        "title": "Landsat 9 ARD",
        "description": "Continuation of Landsat 8 mission from 2022 onward.",
        "tags": ["vegetation","land cover","recent","2022","2023","2024","landsat"],
        "temporal": "2022–present",
        "resolution": "30m",
    },
    {
        "id": "ga_s2am_ard_3",
        "sensor": "Sentinel-2A",
        "title": "Sentinel-2A ARD",
        "description": "High-resolution (10m) surface reflectance. Better spatial detail than Landsat.",
        "tags": ["high resolution","urban","coastal","agriculture","detail","sentinel"],
        "temporal": "2017–present",
        "resolution": "10m",
    },
    {
        "id": "ga_s2bm_ard_3",
        "sensor": "Sentinel-2B",
        "title": "Sentinel-2B ARD",
        "description": "Paired with 2A for ~5-day revisit at 10m resolution.",
        "tags": ["high resolution","urban","coastal","agriculture","detail","sentinel"],
        "temporal": "2017–present",
        "resolution": "10m",
    },
    {
        "id": "ga_ls_wo_3",
        "sensor": "Landsat",
        "title": "Water Observations (WOfS)",
        "description": "Detects water presence. Use for flood mapping, wetland monitoring, reservoir tracking.",
        "tags": ["water","flood","wetland","inundation","reservoir","drought","river","lake"],
        "temporal": "1986–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_fc_3",
        "sensor": "Landsat",
        "title": "Fractional Cover",
        "description": "Estimates ground cover as fractions of green vegetation, non-green vegetation, and bare soil.",
        "tags": ["vegetation","bare soil","ground cover","pasture","grazing","land degradation","agriculture","drought"],
        "temporal": "1986–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_tcw_percentiles_2",
        "sensor": "Landsat",
        "title": "Tasseled Cap Wetness",
        "description": "Summarises landscape wetness over time. Useful for identifying persistently wet areas.",
        "tags": ["wetness","moisture","seasonal","vegetation stress","time series"],
        "temporal": "1986–present",
        "resolution": "30m",
    },
    {
        "id": "ga_srtm_dem1sv1_0",
        "sensor": "SRTM",
        "title": "Digital Elevation Model",
        "description": "Elevation from the Shuttle Radar Topography Mission. Use for terrain, flood modelling, hydrology.",
        "tags": ["elevation","DEM","terrain","flood modelling","slope","topography","hydrology"],
        "temporal": "2000 (static)",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_landcover_class_cyear_2",
        "sensor": "Landsat",
        "title": "DEA Land Cover",
        "description": "Annual land cover classification. Detects changes in land use across Australia.",
        "tags": ["land cover","land use","classification","urban","forest","clearing","change"],
        "temporal": "1988–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_mangrove_cover_cyear_3",
        "sensor": "Landsat",
        "title": "Mangrove Cover",
        "description": "Annual mapping of mangrove extent and canopy cover around Australian coastlines.",
        "tags": ["mangrove","coastal","wetland","canopy","intertidal","marine"],
        "temporal": "1987–present",
        "resolution": "30m",
    },
]

EXAMPLE_QUESTIONS = [
    "Which areas in the Riverina had standing water after the 2022 floods?",
    "Show me vegetation stress across the Murray-Darling Basin in summer 2023",
    "Before and after scenes of the 2019-20 Black Summer fires in East Gippsland",
    "How has urban extent changed in Greater Western Sydney since 2018?",
    "Get mangrove cover data along the Gulf of Carpentaria for the last 5 years",
    "Show elevation data for flood-prone areas near Lismore",
    "Detect surface disturbance near mine sites in the Pilbara in 2023",
    "Track shoreline change along the Gold Coast over the last decade",
]

# ── Cached clients ────────────────────────────────────────────────────────────
@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )

@st.cache_resource
def get_geocoder():
    return Nominatim(user_agent="dea-streamlit-search/1.0")

@st.cache_resource
def get_ai_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

# ── Geocoding ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(place_name: str, buffer_deg: float = 0.5) -> dict:
    """Convert a place name to a WGS84 bounding box via Nominatim REST API."""
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": place_name + ", Australia",
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 0,
                },
                headers={"User-Agent": "dea-streamlit-search/1.0 rohan.tankey@gmail.com"},
                timeout=10,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Geocoding failed after 3 attempts: {e}")
            time.sleep(2 ** attempt)

    if not data:
        raise ValueError(f'Could not geocode "{place_name}"')

    g = data[0]
    centroid = [float(g["lon"]), float(g["lat"])]
    if "boundingbox" in g:
        s, n, w, e = [float(v) for v in g["boundingbox"]]
        bbox = [w, s, e, n]
    else:
        lon, lat = centroid
        bbox = [lon - buffer_deg, lat - buffer_deg,
                lon + buffer_deg, lat + buffer_deg]

    return {"place": g.get("display_name", place_name), "bbox": bbox, "centroid": centroid}

# ── Claude question parser ────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a geospatial assistant for Digital Earth Australia (DEA). "
    "Extract search parameters from the user's question. "
    "Respond ONLY with valid JSON — no markdown fences, no commentary:\n"
    "{\n"
    '  "place_name": "<location in Australia>",\n'
    '  "start_date": "<YYYY-MM-DD>",\n'
    '  "end_date": "<YYYY-MM-DD>",\n'
    '  "collections": ["<collection_id>"],\n'
    '  "max_cloud_pct": <0-100>,\n'
    '  "reasoning": "<one sentence>"\n'
    "}\n"
    "Rules:\n"
    "- Pick 1-2 most relevant collection IDs from the catalogue\n"
    "- Default to last 12 months if no date mentioned\n"
    "- For water/flood questions set max_cloud_pct=10\n"
    "- For DEM/elevation set max_cloud_pct=100\n"
    f"- Today: {datetime.now().strftime('%Y-%m-%d')}"
)

def parse_question_with_claude(question: str) -> dict:
    """Use Claude to parse a natural language question into search params."""
    ai = get_ai_client()
    if not ai:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    catalogue_json = json.dumps(
        [{"id": c["id"], "title": c["title"], "description": c["description"], "tags": c["tags"]}
         for c in DEA_COLLECTIONS],
        indent=2,
    )
    user_msg = f"Question: {question}\n\nAvailable DEA collections:\n{catalogue_json}"

    response = ai.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def parse_question_local(question: str) -> dict:
    """Keyword-based fallback parser (no API key needed)."""
    q = question.lower()
    now = datetime.now()

    year_match = None
    import re
    m = re.search(r"\b(20\d{2})(?:[–\-](20\d{2}))?\b", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2)) if m.group(2) else int(m.group(1))
        start_date, end_date = f"{y1}-01-01", f"{y2}-12-31"
    elif "last 5 year" in q:
        start_date = f"{now.year - 5}-01-01"
        end_date = now.strftime("%Y-%m-%d")
    elif "last year" in q:
        start_date = f"{now.year - 1}-01-01"
        end_date = f"{now.year - 1}-12-31"
    else:
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

    scored = []
    for col in DEA_COLLECTIONS:
        score = sum(2 for tag in col["tags"] if tag.lower() in q)
        if col["sensor"].lower() in q:
            score += 5
        scored.append((score, col["id"]))
    scored.sort(reverse=True)
    collections = [scored[0][1]] if scored[0][0] == 0 else [c for s, c in scored[:2] if s > 0]

    max_cloud = 10 if any(w in q for w in ["water", "flood"]) else \
                100 if any(w in q for w in ["elevation", "dem", "terrain"]) else 20

    known = ["Riverina","Murray-Darling Basin","East Gippsland","Western Sydney",
             "Greater Western Sydney","Pilbara","Gulf of Carpentaria","Kakadu","Lismore",
             "Lake Eyre","Tasmania","Gold Coast","Kimberley","Top End","Hunter Valley",
             "Snowy Mountains","Cape York","Arnhem Land"]
    place_name = next((p for p in known if p.lower() in question.lower()), "Australia")

    return {
        "place_name": place_name,
        "start_date": start_date,
        "end_date": end_date,
        "collections": collections,
        "max_cloud_pct": max_cloud,
        "reasoning": "Matched by keyword analysis (local fallback — no API key).",
    }


def parse_question(question: str) -> dict:
    """Parse question, falling back to local parser if Claude unavailable."""
    try:
        return parse_question_with_claude(question)
    except Exception:
        return parse_question_local(question)

# ── STAC search ───────────────────────────────────────────────────────────────
def stac_search(collection: str, bbox: list, start_date: str, end_date: str,
                max_results: int = 100) -> list:
    resp = requests.post(
        f"{STAC_ENDPOINT}search",
        json={
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "limit": max_results,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("features", [])


def build_asset_index(features: list, collection_id: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        item_id = feat["id"]
        dt      = feat["properties"].get("datetime", "")
        cloud   = feat["properties"].get("eo:cloud_cover")
        bbox    = feat.get("bbox", [])
        for asset_name, asset in feat.get("assets", {}).items():
            href = asset.get("href", "")
            https_url = href.replace(f"s3://{S3_BUCKET}/", f"{S3_HTTP_BASE}/") \
                if href.startswith("s3://") else href
            rows.append({
                "item_id":    item_id,
                "datetime":   dt[:10] if dt else "",
                "collection": collection_id,
                "cloud_pct":  cloud,
                "asset":      asset_name,
                "media_type": asset.get("type", ""),
                "s3_href":    href,
                "https_url":  https_url,
                "bbox_str":   str(bbox),
            })
    return pd.DataFrame(rows)


def run_search(params: dict, bbox: list, max_results: int = 50) -> pd.DataFrame:
    frames = []
    for coll_id in params["collections"]:
        features = stac_search(coll_id, bbox,
                               params["start_date"], params["end_date"],
                               max_results=max_results)
        if not features:
            continue
        df = build_asset_index(features, coll_id)
        if params["max_cloud_pct"] < 100:
            df = df[df["cloud_pct"].isna() | (df["cloud_pct"] <= params["max_cloud_pct"])]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Map helper ────────────────────────────────────────────────────────────────
def make_map(bbox: list, results: pd.DataFrame | None = None) -> folium.Map:
    """Build a Folium map centred on the bbox with optional scene footprints."""
    west, south, east, north = bbox
    centre = [(south + north) / 2, (west + east) / 2]
    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB dark_matter")

    # Bbox rectangle
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#5fd85f", weight=2, fill=True, fill_opacity=0.05,
    ).add_to(m)

    # Scene footprints (unique items only to avoid overlap)
    if results is not None and not results.empty:
        seen = set()
        for _, row in results.iterrows():
            if row["item_id"] in seen:
                continue
            seen.add(row["item_id"])
            try:
                import ast
                b = ast.literal_eval(row["bbox_str"])
                if len(b) == 4:
                    w2, s2, e2, n2 = b
                    folium.Rectangle(
                        bounds=[[s2, w2], [n2, e2]],
                        color="#3a9e6e", weight=1,
                        fill=True, fill_opacity=0.08,
                        tooltip=f"{row['item_id']}<br>{row['datetime']}"
                               + (f"<br>☁ {row['cloud_pct']:.0f}%" if row["cloud_pct"] is not None else ""),
                    ).add_to(m)
            except Exception:
                pass
    return m

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(active_collection_ids: list = None):
    with st.sidebar:
        st.markdown("## 🛰️ DEA Collections")
        for col in DEA_COLLECTIONS:
            active = active_collection_ids and col["id"] in active_collection_ids
            bg = "background-color:#1a3a1a;" if active else ""
            border = "border-left:3px solid #5fd85f;" if active else "border-left:3px solid #2a3a2a;"
            st.markdown(
                f"""<div style='padding:8px 10px;margin-bottom:6px;border-radius:4px;{bg}{border}'>
                <span style='font-size:13px;font-weight:{"600" if active else "400"};
                color:{"#c8ffc8" if active else "#8aaa8a"}'>{col["title"]}</span><br>
                <span style='font-size:10px;color:#5a7a5a;font-family:monospace'>{col["id"]}</span><br>
                <span style='font-size:10px;color:#4a6a4a'>{col["resolution"]} · {col["temporal"]}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Links**")
        st.markdown("- [DEA Explorer](https://explorer.dea.ga.gov.au/)")
        st.markdown("- [Knowledge Hub](https://knowledge.dea.ga.gov.au/)")
        st.markdown("- [STAC API](https://explorer.dea.ga.gov.au/stac/)")

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .stApp { background: #050e05; }
    h1,h2,h3 { color: #c8e6c8 !important; }
    .stTextArea textarea { background: #0a160a !important; color: #c8e6c8 !important;
        border: 1px solid #2a4a2a !important; font-family: monospace; }
    .stButton button { background: #1a3a1a !important; color: #c8ffc8 !important;
        border: 1px solid #3a6a3a !important; }
    .stButton button:hover { background: #2a5a2a !important; }
    .stSelectbox div, .stMultiSelect div { background: #0a160a !important; }
    .stDataFrame { background: #0a160a !important; }
    div[data-testid="stExpander"] { background: #080f08; border: 1px solid #1a2e1a; border-radius: 6px; }
    .stMetric { background: #080f08; border: 1px solid #1a2e1a; border-radius: 6px; padding: 8px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛰️ DEA S3 Smart Search")
    st.caption("Ask a plain English question about the Australian landscape — returns S3 asset URLs.")

    # ── API key check ──
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "⚠️  `ANTHROPIC_API_KEY` not found in environment. "
            "Using local keyword parser as fallback — set the key for Claude-powered question parsing.",
            icon="🔑",
        )

    # ── Question input ──
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Your question",
            placeholder="e.g. Which areas in the Riverina had standing water after the 2022 floods?",
            height=80,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search", use_container_width=True)
        max_results = st.slider("Max scenes", 5, 100, 20, step=5)

    # ── Example questions ──
    st.markdown("**Examples:**")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 4].button(ex[:48] + ("…" if len(ex) > 48 else ""), key=f"ex_{i}",
                              use_container_width=True):
            st.session_state["prefill"] = ex
            st.rerun()

    # Handle prefill from example button
    if "prefill" in st.session_state:
        question = st.session_state.pop("prefill")

    # ── Run search ──
    if search_clicked and question.strip():
        render_sidebar()

        with st.status("Running search…", expanded=True) as status_box:

            # Step 1: parse
            st.write("🧠 Parsing question…")
            try:
                params = parse_question(question)
            except Exception as e:
                st.error(f"Parsing failed: {e}")
                return

            st.write(f"📍 Location: **{params['place_name']}**")
            st.write(f"📅 Dates: **{params['start_date']}** → **{params['end_date']}**")
            st.write(f"🛰  Collections: **{', '.join(params['collections'])}**")
            st.write(f"☁️  Max cloud: **{params['max_cloud_pct']}%**")
            if params.get("reasoning"):
                st.write(f"💡 _{params['reasoning']}_")

            # Step 2: geocode
            st.write("🗺  Geocoding location…")
            try:
                geo = geocode_place(params["place_name"])
                bbox = geo["bbox"]
                st.write(f"✅ `{geo['place']}`")
                st.write(f"   BBox: `{[round(v, 3) for v in bbox]}`")
            except Exception as e:
                st.error(f"Geocoding failed: {e}")
                return

            # Step 3: STAC search
            st.write("🔍 Querying STAC API…")
            try:
                results = run_search(params, bbox, max_results=max_results)
            except Exception as e:
                st.error(f"STAC search failed: {e}")
                return

            n_scenes = results["item_id"].nunique() if not results.empty else 0
            n_assets = len(results)
            st.write(f"✅ Found **{n_scenes} scenes** / **{n_assets} assets**")
            status_box.update(label="Search complete ✅", state="complete")

        render_sidebar(active_collection_ids=params["collections"])

        # ── Results layout ──
        if results.empty:
            st.info("No results found. Try adjusting the date range, cloud cover, or location.")
            return

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scenes", n_scenes)
        m2.metric("Assets", n_assets)
        avg_cloud = results["cloud_pct"].dropna().mean()
        m3.metric("Avg cloud cover", f"{avg_cloud:.1f}%" if not pd.isna(avg_cloud) else "N/A")
        m4.metric("Collections", results["collection"].nunique())

        tab_map, tab_table, tab_urls, tab_code = st.tabs(["🗺 Map", "📋 Table", "🔗 URLs", "📓 Notebook code"])

        with tab_map:
            m = make_map(bbox, results)
            st_folium(m, width="100%", height=500, returned_objects=[])

        with tab_table:
            display_cols = ["item_id", "datetime", "collection", "cloud_pct", "asset", "media_type"]
            display_cols = [c for c in display_cols if c in results.columns]
            st.dataframe(
                results[display_cols].sort_values(["datetime", "item_id"]),
                use_container_width=True,
                height=400,
            )
            csv = results.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "dea_results.csv", "text/csv")

        with tab_urls:
            st.markdown("**HTTPS-accessible S3 asset URLs** — click to open/download directly.")
            unique_items = results["item_id"].unique()
            for item_id in unique_items[:30]:
                item_rows = results[results["item_id"] == item_id]
                dt = item_rows["datetime"].iloc[0]
                cloud = item_rows["cloud_pct"].iloc[0]
                cloud_str = f" · ☁ {cloud:.0f}%" if pd.notna(cloud) else ""
                with st.expander(f"📦 {item_id}  ({dt}{cloud_str})"):
                    for _, row in item_rows.iterrows():
                        url = row["https_url"]
                        if url:
                            st.markdown(
                                f"`{row['asset']:30s}` [{url.split('/')[-1]}]({url})",
                            )
            if len(unique_items) > 30:
                st.caption(f"Showing first 30 of {len(unique_items)} scenes. Download CSV for all.")

        with tab_code:
            st.markdown("**Reproduce this search in Python:**")
            code = f'''import os, requests, pandas as pd
import anthropic
from geopy.geocoders import Nominatim

# pip install pystac-client boto3 requests anthropic geopy

S3_BUCKET     = "dea-public-data"
S3_REGION     = "ap-southeast-2"
STAC_ENDPOINT = "https://explorer.dea.ga.gov.au/stac/"
S3_HTTP_BASE  = f"https://{{S3_BUCKET}}.s3.{{S3_REGION}}.amazonaws.com"

# Search parameters (from your question)
params = {json.dumps({k: v for k, v in params.items() if k != "reasoning"}, indent=4)}

bbox = {bbox}

# Run STAC search
resp = requests.post(
    STAC_ENDPOINT + "search",
    json={{
        "collections": params["collections"],
        "bbox": bbox,
        "datetime": f"{{params['start_date']}}/{{params['end_date']}}",
        "limit": {max_results},
    }}
)
features = resp.json()["features"]

# Build asset index
rows = []
for feat in features:
    cloud = feat["properties"].get("eo:cloud_cover")
    if cloud is not None and cloud > params["max_cloud_pct"]:
        continue
    for name, asset in feat["assets"].items():
        href = asset.get("href", "")
        rows.append({{
            "item_id":   feat["id"],
            "datetime":  feat["properties"].get("datetime", "")[:10],
            "asset":     name,
            "s3_href":   href,
            "https_url": href.replace(f"s3://{{S3_BUCKET}}/", f"{{S3_HTTP_BASE}}/"),
            "cloud_pct": cloud,
        }})

df = pd.DataFrame(rows)
print(f"{{df['item_id'].nunique()}} scenes / {{len(df)}} assets")
df[["item_id", "datetime", "asset", "https_url"]].head(10)
'''
            st.code(code, language="python")

    else:
        render_sidebar()
        st.markdown("""
        ### How it works

        1. **Type a question** about the Australian landscape
        2. **Claude parses** the question into a location, date range, and recommended DEA collection(s)
        3. **Nominatim geocodes** the location to a bounding box
        4. **DEA STAC API** returns matching satellite scenes
        5. You get **HTTPS-accessible S3 URLs** for every asset — ready to download or use in analysis

        Results include a map of scene footprints, a filterable table, direct asset URLs, and
        copy-paste Python code to reproduce the search.
        """)


if __name__ == "__main__":
    main()
