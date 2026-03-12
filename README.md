# DEA S3 Smart Search — Streamlit App

Ask a plain English question about the Australian landscape and get back
HTTPS-accessible S3 asset URLs from the Digital Earth Australia public bucket.

## Quick start

```bash
# 1. Clone / unzip this folder, then:
cd dea_search

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...

# 5. Run
streamlit run app.py
```

The app will open at http://localhost:8501

## API key

`ANTHROPIC_API_KEY` enables Claude-powered question parsing (recommended).  
Without it, the app falls back to a local keyword parser — still functional,
less accurate on complex questions.

You can also put it in a `.env` file in this directory:
```
ANTHROPIC_API_KEY=sk-ant-...
```
and load it with `python-dotenv` if preferred.

## What it does

| Step | Tool |
|------|------|
| Parse natural language question | Claude (`claude-sonnet-4-20250514`) |
| Convert place name → bounding box | Nominatim (OpenStreetMap) |
| Search satellite metadata | DEA STAC API (`explorer.dea.ga.gov.au/stac`) |
| Return asset URLs | `dea-public-data` S3 bucket (public, no credentials needed) |

## Tabs in the UI

- **Map** — scene footprints on a dark basemap (Folium)
- **Table** — sortable/filterable result DataFrame with CSV export
- **URLs** — expandable per-scene asset links (direct HTTPS to S3)
- **Notebook code** — copy-paste Python to reproduce the search

## Deployment

Works anywhere Python runs:

```bash
# Streamlit Community Cloud — push to GitHub, connect repo, set secret ANTHROPIC_API_KEY
# Docker
docker build -t dea-search .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... dea-search
```

A minimal `Dockerfile` for the Docker option:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Data sources

- **DEA public S3 bucket**: `s3://dea-public-data` (ap-southeast-2) — no AWS credentials needed
- **STAC endpoint**: https://explorer.dea.ga.gov.au/stac/
- **Knowledge Hub**: https://knowledge.dea.ga.gov.au/
