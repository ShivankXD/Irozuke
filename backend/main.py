"""
Irozuke AI — Backend API
FastAPI server for manga colorization pipeline.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from routers import colorize, health

# Ensure runtime directories exist before the app mounts static files
for _dir in ("uploads", "outputs", "models"):
    Path(_dir).mkdir(exist_ok=True)

app = FastAPI(
    title="Irozuke AI API",
    description="Manga colorization backend — upload → colorize → return",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow your frontend to call this API (update origins before production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this when you have a domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ────────────────────────────────────────────────────────
# Outputs are served at /outputs/<filename> so the frontend can display them
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(colorize.router, prefix="/api", tags=["Colorize"])
