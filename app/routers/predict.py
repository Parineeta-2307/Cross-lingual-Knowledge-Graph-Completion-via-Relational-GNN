"""Prediction endpoints for the web application."""

import json
from typing import Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.utils.config_loader import load_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
config = load_config()

LANG_FLAGS = {"de": "🇩🇪", "ja": "🇯🇵", "nl": "🇳🇱", "en": "🇺🇸"}
LANG_LABELS = {"de": "German", "ja": "Japanese", "nl": "Dutch", "en": "English"}


def _get_common_template_context(request: Request) -> dict:
    """Get common variables needed by all templates."""
    predictor = getattr(request.app.state, "predictor", None)
    
    # Extract relations for the dropdown menu
    relations = []
    if predictor and hasattr(predictor, "relation_to_id"):
        relations = sorted(list(predictor.relation_to_id.keys()))
        
    return {
        "request": request,
        "model_loaded": predictor is not None,
        "relations": relations,
        "lang_flags": LANG_FLAGS,
        "lang_labels": LANG_LABELS,
    }


@router.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Render the main search homepage."""
    context = _get_common_template_context(request)
    
    # Get basic stats for the hero section
    predictor = getattr(request.app.state, "predictor", None)
    if predictor and hasattr(predictor, "graph_stats"):
        context["num_entities"] = predictor.graph_stats.get("num_entities", 0)
        context["num_triples"] = predictor.graph_stats.get("num_triples", 0)
        
    return templates.TemplateResponse("index.html", context)


@router.post("/predict", response_class=HTMLResponse)
async def post_predict(
    request: Request,
    head: str = Form(...),
    relation: str = Form(...),
    top_k: int = Form(5),
):
    """Handle form submission from the index page and render results."""
    context = _get_common_template_context(request)
    context["query_head"] = head
    context["query_relation"] = relation
    context["query_top_k"] = top_k
    
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        context["error"] = "Model is not loaded. Please train the model and start the server again."
        return templates.TemplateResponse("index.html", context)

    try:
        results = predictor.predict(head, relation, top_k=top_k)
        context["results"] = results
        
        # Determine if any result triggered the cold-start fallback
        context["used_fallback"] = any(r.get("cold_start_fallback", False) for r in results)
        
    except ValueError as e:
        context["error"] = str(e)
    except Exception as e:
        context["error"] = f"An unexpected error occurred: {str(e)}"
        
    return templates.TemplateResponse("index.html", context)


@router.get("/api/predict", response_class=JSONResponse)
async def api_predict(
    request: Request,
    head: str,
    relation: str,
    top_k: int = 5,
):
    """JSON API endpoint for programmatic prediction access."""
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        return JSONResponse(
            status_code=503, 
            content={"error": "Model not loaded"}
        )

    try:
        results = predictor.predict(head, relation, top_k=top_k)
        return {
            "query": {"head": head, "relation": relation},
            "predictions": results,
            "model_loaded": True
        }
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/entity/{entity_id}", response_class=HTMLResponse)
async def get_entity(request: Request, entity_id: str):
    """Render the entity detail page with neighborhood graph."""
    context = _get_common_template_context(request)
    
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        context["error"] = "Model is not loaded."
        return templates.TemplateResponse("entity.html", context)
        
    # Get neighborhood data for Cytoscape.js
    neighborhood = predictor.get_neighborhood(entity_id)
    
    if "error" in neighborhood:
        context["error"] = neighborhood["error"]
        return templates.TemplateResponse("entity.html", context)
        
    context["entity"] = neighborhood["center"]
    
    # We pass the neighborhood directly as a JSON string to be parsed by JS
    context["neighborhood_json"] = json.dumps(neighborhood)
    
    # Build known facts table (1-hop neighbors)
    facts = []
    for edge in neighborhood.get("elements", []):
        if "source" in edge["data"]:
            src = edge["data"]["source"]
            dst = edge["data"]["target"]
            rel = edge["data"]["relation"]
            
            # Find the names of the source and target
            src_name = "Unknown"
            dst_name = "Unknown"
            dst_lang = "en"
            
            for node in neighborhood.get("elements", []):
                if "id" in node["data"]:
                    if node["data"]["id"] == src:
                        src_name = node["data"]["label"]
                    if node["data"]["id"] == dst:
                        dst_name = node["data"]["label"]
                        dst_lang = node["data"]["language"]
            
            if src == context["entity"]["id"]:
                facts.append({
                    "relation": rel,
                    "object": dst_name,
                    "language": dst_lang,
                    "direction": "out"
                })
            else:
                facts.append({
                    "relation": rel + " (reverse)",
                    "object": src_name,
                    "language": "en", # Simplified
                    "direction": "in"
                })
                
    context["known_facts"] = facts
    
    return templates.TemplateResponse("entity.html", context)
