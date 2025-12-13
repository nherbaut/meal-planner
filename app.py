from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
TEMPLATES_DIR = APP_DIR / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

app = FastAPI(title="Meal Planning")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _validate_date_str(monday: str) -> None:
    if not DATE_RE.match(monday):
        raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
    try:
        datetime.strptime(monday, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date value")


def _path_for(monday: str) -> Path:
    return DATA_DIR / f"{monday}.json"


def _shopping_list_path(monday: str) -> Path:
    return DATA_DIR / f"shopping-list-{monday}.json"


def _list_available() -> List[str]:
    dates: List[str] = []
    for p in DATA_DIR.glob("*.json"):
        stem = p.stem
        if stem.startswith("shopping-list-"):
            continue
        if not DATE_RE.match(stem):
            continue
        dates.append(stem)
    dates.sort()
    return dates


def _load_planning(monday: str) -> List[Dict[str, Any]]:
    p = _path_for(monday)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Planning not found")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Stored planning is corrupted")
    if not isinstance(obj, list):
        raise HTTPException(status_code=400, detail="Planning must be a JSON array")
    return obj


def _save_planning(monday: str, planning: Any) -> None:
    if not isinstance(planning, list):
        raise HTTPException(status_code=400, detail="Planning must be a JSON array")
    # validation légère : chaque item doit être un dict avec les champs attendus
    required = {
        "jour",
        "repas",
        "plats",
        "ingredients",
        "courses",
        "duree_preparation_minutes",
        "restes",
    }
    for i, item in enumerate(planning):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Item #{i} must be an object")
        missing = required - set(item.keys())
        if missing:
            raise HTTPException(status_code=400, detail=f"Item #{i} missing keys: {sorted(missing)}")
    _path_for(monday).write_text(json.dumps(planning, ensure_ascii=False, indent=2), encoding="utf-8")


def _wants_json(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "application/json" in accept or "text/json" in accept


def _load_shopping_list(monday: str) -> Dict[str, List[str]]:
    p = _shopping_list_path(monday)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Shopping list not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Stored shopping list is corrupted")
    # rétrocompatibilité: ancienne version stockait un tableau simple
    if isinstance(data, list):
        cleaned = [x for x in data if isinstance(x, str) and x]
        return {"to_buy": cleaned, "bought": []}
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Stored shopping list must be an object")
    to_buy = data.get("to_buy", [])
    bought = data.get("bought", [])
    if not isinstance(to_buy, list) or not all(isinstance(x, str) for x in to_buy):
        raise HTTPException(status_code=500, detail="Stored shopping list has invalid to_buy")
    if not isinstance(bought, list) or not all(isinstance(x, str) for x in bought):
        raise HTTPException(status_code=500, detail="Stored shopping list has invalid bought")
    return {
        "to_buy": [x for x in to_buy if x],
        "bought": [x for x in bought if x],
    }


def _save_shopping_list(monday: str, to_buy: List[str], bought: Optional[List[str]] = None) -> None:
    to_buy_clean = [str(x).strip() for x in to_buy if isinstance(x, str) and str(x).strip()]
    bought_clean = [str(x).strip() for x in (bought or []) if isinstance(x, str) and str(x).strip()]
    payload = {"to_buy": to_buy_clean, "bought": bought_clean}
    _shopping_list_path(monday).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/meal-planning/", response_class=HTMLResponse)
def list_plannings(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "dates": _list_available(),
        },
    )


@app.get("/meal-planning/{monday}", response_class=HTMLResponse)
def get_planning(request: Request, monday: str):
    _validate_date_str(monday)
    p = _path_for(monday)
    if not p.exists():
        # page upload
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "monday": monday,
            },
        )

    if _wants_json(request):
        planning = _load_planning(monday)
        return JSONResponse(content=planning)

    # page planning (HTML) + JS qui refait un GET Accept: application/json
    return templates.TemplateResponse(
        "planning.html",
        {
            "request": request,
            "monday": monday,
        },
    )


@app.post("/meal-planning/{monday}")
async def post_planning(request: Request, monday: str):
    _validate_date_str(monday)

    content_type = (request.headers.get("content-type") or "").lower()
    planning: Any = None

    if "application/json" in content_type:
        planning = await request.json()
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        raw = (form.get("planning_json") or "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="Missing planning_json")
        try:
            planning = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e.msg}")
    else:
        # tente JSON brut
        raw = (await request.body()).decode("utf-8", errors="replace").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty body")
        try:
            planning = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e.msg}")

    _save_planning(monday, planning)
    return RedirectResponse(url=f"/meal-planning/{monday}", status_code=303)


@app.post("/meal-planning/{monday}/shopping-list")
async def create_shopping_list(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    items = payload.get("items")
    to_buy = payload.get("to_buy")
    bought = payload.get("bought")

    if isinstance(items, list):
        if not all(isinstance(x, str) for x in items):
            raise HTTPException(status_code=400, detail="items must be an array of strings")
        to_buy_list = [x for x in items if isinstance(x, str)]
        bought_list: List[str] = []
    else:
        if not isinstance(to_buy, list) or not all(isinstance(x, str) for x in to_buy):
            raise HTTPException(status_code=400, detail="to_buy must be an array of strings")
        if bought is None:
            bought_list = []
        elif not isinstance(bought, list) or not all(isinstance(x, str) for x in bought):
            raise HTTPException(status_code=400, detail="bought must be an array of strings")
        else:
            bought_list = [x for x in bought if isinstance(x, str)]
        to_buy_list = [x for x in to_buy if isinstance(x, str)]

    _save_shopping_list(monday, to_buy_list, bought_list)
    saved = _load_shopping_list(monday)
    return JSONResponse(status_code=201, content={"monday": monday, **saved})


@app.get("/meal-planning/{monday}/shopping-list", response_class=HTMLResponse)
def shopping_list(request: Request, monday: str):
    _validate_date_str(monday)
    payload = _load_shopping_list(monday)

    if _wants_json(request):
        return JSONResponse(content={"monday": monday, **payload})

    return templates.TemplateResponse(
        "shopping_list.html",
        {
            "request": request,
            "to_buy": payload["to_buy"],
            "bought": payload["bought"],
            "monday": monday,
        },
    )
