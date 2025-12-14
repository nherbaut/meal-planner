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
BUFFER_DIR = DATA_DIR / "buffers"
BUFFER_DIR.mkdir(parents=True, exist_ok=True)

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


def _load_shopping_list(monday: str) -> Dict[str, Any]:
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
    notes = data.get("notes", {})
    if not isinstance(to_buy, list) or not all(isinstance(x, str) for x in to_buy):
        raise HTTPException(status_code=500, detail="Stored shopping list has invalid to_buy")
    if not isinstance(bought, list) or not all(isinstance(x, str) for x in bought):
        raise HTTPException(status_code=500, detail="Stored shopping list has invalid bought")
    if not isinstance(notes, dict):
        notes = {}
    return {
        "to_buy": [x for x in to_buy if x],
        "bought": [x for x in bought if x],
        "notes": {k: v for k, v in notes.items() if isinstance(k, str) and isinstance(v, str)},
    }


def _save_shopping_list(monday: str, to_buy: List[str], bought: Optional[List[str]] = None, notes: Optional[Dict[str, str]] = None) -> None:
    to_buy_clean = [str(x).strip() for x in to_buy if isinstance(x, str) and str(x).strip()]
    bought_clean = [str(x).strip() for x in (bought or []) if isinstance(x, str) and str(x).strip()]
    payload = {"to_buy": to_buy_clean, "bought": bought_clean, "notes": notes or {}}
    _shopping_list_path(monday).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_shopping_list_after_substitution(monday: str, old_meal: Optional[Dict[str, Any]], new_meal: Dict[str, Any]) -> None:
    try:
        payload = _load_shopping_list(monday)
    except HTTPException as e:
        if e.status_code == 404:
            return
        raise

    def norm(s: str) -> str:
        return str(s).strip().lower()

    old_ing = {norm(x) for x in _meal_ingredients(old_meal or {})}
    new_ing_raw = _meal_ingredients(new_meal)
    new_ing = [(x, norm(x)) for x in new_ing_raw]
    meal_title = ", ".join(new_meal.get("plats") or [])

    to_buy = [str(x).strip() for x in payload.get("to_buy", []) if str(x).strip()]
    bought = [str(x).strip() for x in payload.get("bought", []) if str(x).strip()]
    notes = dict(payload.get("notes", {}))

    bought_keys = {norm(x) for x in bought}

    # remove old ingredients from to_buy (but never from bought)
    to_buy_filtered: List[str] = []
    seen_keys: set[str] = set()
    for item in to_buy:
        k = norm(item)
        if k in old_ing:
            continue
        if k in seen_keys:
            continue
        seen_keys.add(k)
        to_buy_filtered.append(item)

    # add new ingredients unless already bought or already in to_buy
    existing_keys = {norm(x) for x in to_buy_filtered}
    for item, k in new_ing:
        if not k:
            continue
        if k in bought_keys:
            continue
        if k in existing_keys:
            continue
        to_buy_filtered.append(item)
        existing_keys.add(k)

    # ensure to_buy has no items present in bought
    to_buy_final = [x for x in to_buy_filtered if norm(x) not in bought_keys]

    # update notes: remove old_ing entries, add new_ing entries with meal plats as reference
    for k in old_ing:
        notes.pop(k, None)
    if meal_title:
        for _, k in new_ing:
            if not k:
                continue
            notes[k] = meal_title

    _save_shopping_list(monday, to_buy_final, bought, notes)


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
    buffer_meals = _load_meal_buffer(monday)
    if not buffer_meals:
        logger.info("Meal buffer empty for %s, generating new pool", monday)
        buffer_meals = _generate_meal_pool(monday, planning)
        _save_meal_buffer(monday, buffer_meals)

    meal = buffer_meals.pop(0)
    _save_meal_buffer(monday, buffer_meals)
    return _normalize_meal(day, repas, meal)


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


def _meal_ingredients(meal: Dict[str, Any]) -> List[str]:
    src = meal.get("courses") if isinstance(meal.get("courses"), list) else meal.get("ingredients")
    if not isinstance(src, list):
        return []
    return [str(x).strip() for x in src if isinstance(x, str) and str(x).strip()]


def _compute_notes_from_planning(monday: str) -> Dict[str, str]:
    try:
        planning = _load_planning(monday)
    except HTTPException:
        return {}
    def norm(s: str) -> str:
        return str(s).strip().lower()
    notes: Dict[str, str] = {}
    for meal in planning:
        if not isinstance(meal, dict):
            continue
        title = ", ".join(meal.get("plats") or []) if isinstance(meal.get("plats"), list) else ""
        for ing in _meal_ingredients(meal):
            k = norm(ing)
            if not k:
                continue
            notes.setdefault(k, title)
    return notes


def _buffer_path(monday: str) -> Path:
    return BUFFER_DIR / f"buffer-{monday}.json"


def _load_meal_buffer(monday: str) -> List[Dict[str, Any]]:
    p = _buffer_path(monday)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Buffer file corrupted, resetting: %s", p)
        return []
    if not isinstance(data, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for m in data:
        if isinstance(m, dict):
            cleaned.append(m)
    return cleaned


def _save_meal_buffer(monday: str, meals: List[Dict[str, Any]]) -> None:
    _buffer_path(monday).write_text(json.dumps(meals, ensure_ascii=False, indent=2), encoding="utf-8")


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
        
    }
    logger.info("OpenAI request (week) monday=%s model=%s", monday, OPENAI_MODEL)
    logger.info("OpenAI payload (week): %s", json.dumps(payload, ensure_ascii=False))
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
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


def _build_pool_prompt(monday: str, planning: List[Dict[str, Any]], previous_weeks: Dict[str, List[Dict[str, Any]]]) -> str:
    return """
Génère 15 repas du soir différents entre eux et différents des repas déjà présents dans cette semaine et la précédente.
Contexte: date du lundi de la semaine en cours: {{date}}.
Semaines à éviter (pas de répétition de plats ou courses) :
{{history}}

Chaque repas doit respecter:
- plats adaptés à 2 parents + 2 enfants (3 et 7 ans)
- Sud-Ouest de la France, produits de saison trouvables en supermarché, pas d’ultra-transformés
- Pas besoin de dessert
- Protéines végétales appréciées
- Préparation < 45 min (idéalement 30)

Réponds par un tableau JSON de 15 objets, format:
{
  "jour": "placeholder",
  "repas": "soir",
  "plats": [...],
  "ingredients": [...],
  "courses": [...],
  "duree_preparation_minutes": 30,
  "restes": [...]
}
Ne mets aucun texte en dehors du tableau JSON.
""".replace("{{date}}", monday).replace("{{history}}", json.dumps({"current": planning, "previous": previous_weeks}, ensure_ascii=False, indent=2))


def _generate_meal_pool(monday: str, planning: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    previous_weeks = _load_previous_weeks(monday, count=1)
    prompt = _build_pool_prompt(monday, planning, previous_weeks)
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un générateur JSON strict. Tu réponds uniquement avec le JSON demandé, sans texte additionnel, sans balises Markdown, sans ```.",
            },
            {"role": "user", "content": prompt},
        ],
        
    }
    logger.info("OpenAI request (pool) monday=%s model=%s", monday, OPENAI_MODEL)
    logger.info("OpenAI payload (pool): %s", json.dumps(payload, ensure_ascii=False))
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}") from e
    logger.info("OpenAI response (pool) status=%s", r.status_code)
    logger.info("OpenAI body (pool): %s", r.text)
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
        raise HTTPException(status_code=500, detail=f"Unable to parse pool JSON: {e}")
    if not isinstance(arr, list):
        raise HTTPException(status_code=500, detail="Generated pool is not a list")
    meals: List[Dict[str, Any]] = [m for m in arr if isinstance(m, dict)]
    if not meals:
        raise HTTPException(status_code=500, detail="Generated pool is empty")
    return meals


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

    # notes : utiliser celles fournies ou les générer depuis le planning
    notes_payload = payload.get("notes")
    if isinstance(notes_payload, dict):
        notes = {str(k): str(v) for k, v in notes_payload.items()}
    else:
        notes = _compute_notes_from_planning(monday)

    # deduplicate and store
    _save_shopping_list(monday, to_buy_list, bought_list, notes)
    saved = _load_shopping_list(monday)
    return JSONResponse(status_code=201, content={"monday": monday, **saved})


@app.get("/meal-planning/{monday}/shopping-list", response_class=HTMLResponse)
def shopping_list(request: Request, monday: str):
    _validate_date_str(monday)
    payload = _load_shopping_list(monday)
    try:
        planning = _load_planning(monday)
    except HTTPException:
        planning = []

    if _wants_json(request):
        return JSONResponse(content={"monday": monday, **payload, "planning": planning})

    return templates.TemplateResponse(
        "shopping_list.html",
        {
            "request": request,
            "to_buy": payload["to_buy"],
            "bought": payload["bought"],
            "notes": payload.get("notes", {}),
            "planning": planning,
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
    old_meal: Optional[Dict[str, Any]] = None

    replaced = False
    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            old_meal = m
            planning[i] = new_meal
            replaced = True
            break
    if not replaced:
        planning.append(new_meal)
    _save_planning(monday, planning)
    try:
        _update_shopping_list_after_substitution(monday, old_meal, new_meal)
    except HTTPException:
        # si pas de liste existante, on ignore
        pass
    return JSONResponse(content={"meal": new_meal, "monday": monday})


@app.post("/meal-planning/{monday}/generate-week")
def generate_week(monday: str):
    _validate_date_str(monday)
    planning = _generate_week(monday)
    _save_planning(monday, planning)
    return JSONResponse(status_code=201, content={"monday": monday, "planning": planning})
