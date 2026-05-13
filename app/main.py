"""FastAPI entry point for the Knowledge Graph Explorer.

This module initializes the FastAPI app, mounts static files, and sets up
the Jinja2 templates. It also handles the startup event to load the trained
model predictor into memory exactly once.
"""

import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from loguru import logger

from src.utils.config_loader import load_config
from src.inference.predictor import KGPredictor
from src.utils.logging_config import setup_logging

# Load config early so we can use it for FastAPI setup
config = load_config()

# Initialize structured logging
setup_logging(log_level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title=config["app"]["title"],
    description=config["app"]["description"],
    version="1.0.0",
)

# Setup directories
APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Import routers (importing here to avoid circular imports if routers import `templates`)
from app.routers import predict, explore, stats

app.include_router(predict.router)
app.include_router(explore.router)
app.include_router(stats.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model and artifacts into memory on startup.
    
    This ensures we only pay the high initialization cost (loading PyTorch
    weights and precomputing embeddings) once when the server boots.
    """
    logger.info("FastAPI starting up...")
    start_time = time.time()
    
    checkpoint_path = Path(config["training"]["checkpoint_dir"]) / config["training"]["checkpoint_name"]
    
    try:
        # Load the predictor and attach it to the app state
        # so routers can access it via request.app.state.predictor
        predictor = KGPredictor(str(checkpoint_path), config)
        app.state.predictor = predictor
        
        # Try to load evaluation stats to log on startup
        eval_path = Path(config["evaluation"]["results_dir"]) / "results.json"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                logger.info(
                    f"Model Performance | Hits@1: {results.get('hits_at_1', 0):.4f} "
                    f"| MRR: {results.get('mrr_filtered', 0):.4f}"
                )
                
        elapsed = time.time() - start_time
        logger.info(f"Startup complete | Time: {elapsed:.2f}s")
        
    except FileNotFoundError as e:
        logger.error(f"Failed to load predictor: {e}")
        logger.warning("App started in degraded mode. Model predictions will not work.")
        app.state.predictor = None
        
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        app.state.predictor = None


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint for Docker/Kubernetes probes."""
    return JSONResponse({
        "status": "ok", 
        "model_loaded": hasattr(app.state, "predictor") and app.state.predictor is not None
    })


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=config["app"]["host"],
        port=config["app"]["port"], 
        reload=config["app"]["debug"]
    )
