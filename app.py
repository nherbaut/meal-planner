from __future__ import annotations

import base64
import json
import re
import logging
import os
import requests
from datetime import datetime, timedelta
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logger = logging.getLogger("meal_planning")
if not logger.handlers:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


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


def _build_regen_prompt(monday: str, planning: List[Dict[str, Any]], day: str, repas: str) -> str:
    template = """
aide moi à faire le menu pour un jour de la semaine. pour 2 parents  2 enfants de 3 et 7 ans mangent à la maison les soirs en semaine et le midi et soir en weekend. Les parent ramène les restes des repas des soirs pour le lendemain midi en semaine. On est en {{date}} nous sommes en france dans le sud ouest, il faut des aliments de saison Il faut des aliments qu'on peut trouver en super marché On ne veut pas d'aliments ultra-transformés Il faut que ça plaise aux enfants Pas besoin d'avoir de la viande/poisson le soir, mais des proteines végétales de bonne qualité sont appréciées. Pas besoin de prévoir le dessert Les repas doivent être préparés en moins de 45\", 30\" idéalement 

- liste des plat - liste des courses pour le repas. - durée de préparation - liste de restes - liste des ingrédients

suivivant cet exemple de json:

{
    "jour": "vendredi",
    "repas": "soir",
    "plats": [
      "Soupe de légumes d’hiver",
      "Tartines de fromage"
    ],
    "ingredients": [
      "courge",
      "carottes",
      "poireau",
      "pomme de terre",
      "pain",
      "fromage",
      "huile d'olive",
      "sel"
    ],
    "courses": [
      "courge",
      "carottes",
      "poireaux",
      "pommes de terre",
      "pain",
      "fromage"
    ],
    "duree_preparation_minutes": 30,
    "restes": [
      "Soupe de légumes"
    ]
  }


les repas devraient être variés par rapport à la liste des repas existant, {{json}}
"""
    extra = f"\nGénère exactement un objet JSON pour le jour '{day}' et le repas '{repas}', au format de l'exemple ci-dessus, sans texte additionnel."
    return template.replace("{{date}}", monday).replace("{{json}}", json.dumps(planning, ensure_ascii=False, indent=2)) + extra


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")
    snippet = text.strip()

    # If response already starts with JSON, keep as-is
    if snippet.startswith("{") or snippet.startswith("["):
        return json.loads(snippet)

    # Try fenced blocks first
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                return json.loads(part)

    # Fallback: capture outermost object, otherwise outermost array
    if "{" in text and "}" in text and text.find("{") < text.rfind("}"):
        snippet = text[text.find("{") : text.rfind("}") + 1]
        return json.loads(snippet)
    if "[" in text and "]" in text and text.find("[") < text.rfind("]"):
        snippet = text[text.find("[") : text.rfind("]") + 1]
        return json.loads(snippet)

    return json.loads(snippet)


def _generate_meal(monday: str, day: str, repas: str, planning: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    prompt = _build_regen_prompt(monday, planning, day, repas)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 1,
    }
    logger.info("OpenAI request (meal) monday=%s day=%s repas=%s model=%s", monday, day, repas, OPENAI_MODEL)
    logger.info("OpenAI payload (meal): %s", json.dumps(payload, ensure_ascii=False))
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}") from e
    logger.info("OpenAI response (meal) status=%s", r.status_code)
    logger.info("OpenAI body (meal): %s", r.text)
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=500, detail=f"OpenAI error: {detail}")
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        meal = _extract_json_from_text(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to parse meal JSON: {e}")

    # Normalise/force keys
    meal["jour"] = day
    meal["repas"] = repas
    meal["plats"] = meal.get("plats") if isinstance(meal.get("plats"), list) else []
    meal["ingredients"] = meal.get("ingredients") if isinstance(meal.get("ingredients"), list) else []
    meal["courses"] = meal.get("courses") if isinstance(meal.get("courses"), list) else []
    meal["restes"] = meal.get("restes") if isinstance(meal.get("restes"), list) else []
    if not isinstance(meal.get("duree_preparation_minutes"), (int, float)):
        meal["duree_preparation_minutes"] = None
    else:
        meal["duree_preparation_minutes"] = int(meal["duree_preparation_minutes"])
    return meal


def _normalize_meal(day: str, repas: str, meal: Dict[str, Any]) -> Dict[str, Any]:
    meal = dict(meal or {})
    meal["jour"] = day
    meal["repas"] = repas
    meal["plats"] = meal.get("plats") if isinstance(meal.get("plats"), list) else []
    meal["ingredients"] = meal.get("ingredients") if isinstance(meal.get("ingredients"), list) else []
    meal["courses"] = meal.get("courses") if isinstance(meal.get("courses"), list) else meal.get("courses", [])
    meal["restes"] = meal.get("restes") if isinstance(meal.get("restes"), list) else []
    if not isinstance(meal.get("duree_preparation_minutes"), (int, float)):
        meal["duree_preparation_minutes"] = None
    else:
        meal["duree_preparation_minutes"] = int(meal["duree_preparation_minutes"])
    return meal


def _load_previous_weeks(monday: str, count: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    base = datetime.strptime(monday, "%Y-%m-%d").date()
    prev: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(1, count + 1):
        d = base - timedelta(days=7 * i)
        key = d.isoformat()
        try:
            prev[key] = _load_planning(key)
        except HTTPException:
            continue
    return prev


def _build_week_prompt(monday: str, previous_weeks: Dict[str, List[Dict[str, Any]]]) -> str:
    template = """
aide moi à faire le menu pour une semaine complète. pour 2 parents  2 enfants de 3 et 7 ans mangent à la maison les soirs en semaine et le midi et soir en weekend. Les parent ramène les restes des repas des soirs pour le lendemain midi en semaine. On est en {{date}} nous sommes en france dans le sud ouest, il faut des aliments de saison Il faut des aliments qu'on peut trouver en super marché On ne veut pas d'aliments ultra-transformés Il faut que ça plaise aux enfants Pas besoin d'avoir de la viande/poisson le soir, mais des proteines végétales de bonne qualité sont appréciées. Pas besoin de prévoir le dessert Les repas doivent être préparés en moins de 45", 30" idéalement 

- liste des plat - liste des courses pour le repas. - durée de préparation - liste de restes - liste des ingrédients

Retourne un tableau JSON, avec un objet par repas du soir (jour, repas=soir) pour lundi, mardi, mercredi, jeudi, vendredi, samedi, dimanche, au format:

{
    "jour": "vendredi",
    "repas": "soir",
    "plats": [
      "Soupe de légumes d’hiver",
      "Tartines de fromage"
    ],
    "ingredients": [
      "courge",
      "carottes",
      "poireau",
      "pomme de terre",
      "pain",
      "fromage",
      "huile d'olive",
      "sel"
    ],
    "courses": [
      "courge",
      "carottes",
      "poireaux",
      "pommes de terre",
      "pain",
      "fromage"
    ],
    "duree_preparation_minutes": 30,
    "restes": [
      "Soupe de légumes"
    ]
  }

Semaines précédentes (ne pas répéter les plats/courses proposés) :
{{history}}

Réponds UNIQUEMENT par un tableau JSON valide (pas de texte avant/après, pas de ```). Pas de plat déjà proposé dans les semaines précédentes.
"""
    history = json.dumps(previous_weeks, ensure_ascii=False, indent=2)
    return template.replace("{{date}}", monday).replace("{{history}}", history)

def _generate_week(monday: str) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    previous_weeks = _load_previous_weeks(monday, count=2)
    prompt = _build_week_prompt(monday, previous_weeks)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un générateur JSON strict. Tu réponds uniquement avec le JSON demandé, sans texte additionnel, sans balises Markdown, sans ```.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
    }
    logger.info("OpenAI request (week) monday=%s model=%s", monday, OPENAI_MODEL)
    logger.info("OpenAI payload (week): %s", json.dumps(payload, ensure_ascii=False))
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}") from e
    logger.info("OpenAI response (week) status=%s", r.status_code)
    logger.info("OpenAI body (week): %s", r.text)
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=500, detail=f"OpenAI error: {detail}")
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        arr = _extract_json_from_text(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to parse planning JSON: {e}")
    if not isinstance(arr, list):
        raise HTTPException(status_code=500, detail="Generated planning is not a list")
    normalized: List[Dict[str, Any]] = []
    order = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    for idx, day in enumerate(order):
        # pick meal from returned list matching day, or fallback to current idx
        found: Optional[Dict[str, Any]] = None
        for m in arr:
            if isinstance(m, dict) and str(m.get("jour", "")).strip().lower() == day:
                found = m
                break
        if found is None and idx < len(arr) and isinstance(arr[idx], dict):
            found = arr[idx]
        if not isinstance(found, dict):
            continue
        normalized.append(_normalize_meal(day, "soir", found))
    if not normalized:
        raise HTTPException(status_code=500, detail="Generated planning is empty")
    return normalized


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


@app.post("/meal-planning/{monday}/regenerate")
async def regenerate_meal(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    day = (body.get("jour") or body.get("day") or "").strip().lower()
    repas = (body.get("repas") or "soir").strip().lower()
    if not day:
        raise HTTPException(status_code=400, detail="Missing jour/day")

    planning = _load_planning(monday)
    new_meal = _generate_meal(monday, day, repas, planning)

    replaced = False
    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            planning[i] = new_meal
            replaced = True
            break
    if not replaced:
        planning.append(new_meal)
    _save_planning(monday, planning)
    return JSONResponse(content={"meal": new_meal, "monday": monday})


@app.post("/meal-planning/{monday}/generate-week")
def generate_week(monday: str):
    _validate_date_str(monday)
    planning = _generate_week(monday)
    _save_planning(monday, planning)
    return JSONResponse(status_code=201, content={"monday": monday, "planning": planning})
