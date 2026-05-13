"""Explore endpoints for full graph visualization."""

import json
from pathlib import Path

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.utils.config_loader import load_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
config = load_config()

@router.get("/explore", response_class=HTMLResponse)
async def get_explore(request: Request):
    """Render the full graph explorer page."""
    predictor = getattr(request.app.state, "predictor", None)
    
    relations = []
    if predictor and hasattr(predictor, "relation_to_id"):
        relations = sorted(list(predictor.relation_to_id.keys()))
        
    context = {
        "request": request,
        "model_loaded": predictor is not None,
        "relations": relations,
        "languages": config["data"]["languages"]
    }
    return templates.TemplateResponse("explore.html", context)

@router.get("/api/graph", response_class=JSONResponse)
async def api_graph(
    request: Request, 
    language: str = Query(None),
    limit: int = Query(300, le=1000),
    offset: int = Query(0)
):
    """JSON API returning graph nodes and edges for Cytoscape.js.
    
    In a real massive graph, you'd only return a specific neighborhood or
    a heavily downsampled version. Here we just return the first N nodes
    and their internal edges to keep the browser from crashing.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
        
    # In a full app we would query the PyG object directly.
    # For simplicity, we just use the predictor's dictionaries to construct
    # a subset of nodes, and then we would find edges between them.
    # To keep it simple, we'll just return the first `limit` entities.
    
    nodes = []
    # Collect nodes
    count = 0
    start = offset
    
    # Find matching nodes
    target_nodes = []
    for id_val, name in predictor.id_to_entity.items():
        lang_id = predictor.node_language[id_val].item()
        lang_str = predictor.id_to_lang_str.get(lang_id, "en")
        
        if language and language != lang_str:
            continue
            
        if count >= start and len(target_nodes) < limit:
            target_nodes.append((id_val, name, lang_str))
            
            is_cold = predictor.cold_start_mask[id_val].item()
            nodes.append({
                "data": {
                    "id": str(id_val),
                    "label": name,
                    "language": lang_str,
                    "cold_start": is_cold
                }
            })
            
        count += 1
        if len(target_nodes) >= limit:
            break
            
    # Now find edges strictly between these target_nodes
    # This is O(N) where N is total edges.
    edges = []
    target_ids_set = {n[0] for n in target_nodes}
    
    # Load edges from unified_graph.pt
    unified_graph_path = Path(config["data"]["processed_dir"]) / "unified_graph.pt"
    if unified_graph_path.exists():
        import torch
        graph_data = torch.load(unified_graph_path, map_location="cpu")
        edge_index = graph_data.edge_index
        edge_types = graph_data.edge_type
        
        id_to_rel = {v: k for k, v in predictor.relation_to_id.items()}
        
        # Iterate over all edges, keep if both source and target are in our subset
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            if src in target_ids_set and dst in target_ids_set:
                rel_type = edge_types[i].item()
                rel_name = id_to_rel.get(rel_type, f"rel_{rel_type}")
                
                edges.append({
                    "data": {
                        "source": str(src),
                        "target": str(dst),
                        "relation": rel_name
                    }
                })
                
    return {
        "elements": nodes + edges,
        "total_nodes": predictor.num_entities,
        "filtered_nodes": len(nodes),
        "filtered_edges": len(edges)
    }
