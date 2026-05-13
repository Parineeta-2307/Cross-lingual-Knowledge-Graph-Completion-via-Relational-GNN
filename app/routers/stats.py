"""Stats endpoints for the dashboard."""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.utils.config_loader import load_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
config = load_config()

@router.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    """Render the statistics dashboard."""
    predictor = getattr(request.app.state, "predictor", None)
    
    context = {
        "request": request,
        "model_loaded": predictor is not None,
    }
    
    # Load eval results if available
    eval_path = Path(config["evaluation"]["results_dir"]) / "results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            context["eval_results"] = json.load(f)
            
    # Load graph stats if available
    graph_stats_path = Path(config["data"]["processed_dir"]) / "graph_stats.json"
    if graph_stats_path.exists():
        with open(graph_stats_path, "r", encoding="utf-8") as f:
            context["graph_stats"] = json.load(f)
            
    # Load relation breakdown if available
    rel_path = Path(config["evaluation"]["results_dir"]) / "relation_results.csv"
    if rel_path.exists():
        import pandas as pd
        df = pd.read_csv(rel_path)
        context["relation_stats"] = df.to_dict(orient="records")
            
    return templates.TemplateResponse("stats.html", context)

@router.get("/api/stats", response_class=JSONResponse)
async def api_stats():
    """JSON API returning all model and dataset metrics."""
    result = {}
    
    # Load eval results
    eval_path = Path(config["evaluation"]["results_dir"]) / "results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            result["model"] = json.load(f)
            
    # Load graph stats
    graph_stats_path = Path(config["data"]["processed_dir"]) / "graph_stats.json"
    if graph_stats_path.exists():
        with open(graph_stats_path, "r", encoding="utf-8") as f:
            result["dataset"] = json.load(f)
            
    return result
