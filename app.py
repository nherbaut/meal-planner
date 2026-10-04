from __future__ import annotations

import base64
import json
import re
import logging
import os
import requests
import copy
import seasonality
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
TEMPLATES_DIR = APP_DIR / "templates"
IMG_DIR = APP_DIR / "img"

DATA_DIR.mkdir(parents=True, exist_ok=True)
BUFFER_DIR = DATA_DIR / "buffers"
BUFFER_DIR.mkdir(parents=True, exist_ok=True)
MEALIE_BUFFER_FILE = BUFFER_DIR / "mealie_buffer.json"
SEASONALITY_FILE = DATA_DIR / "seasonality.json"
DRAFT_DIR = DATA_DIR / "drafts"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

app = FastAPI(title="Meal Planning")
app.mount("/img", StaticFiles(directory=str(IMG_DIR)), name="images")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MEALIE_URL = os.getenv("MEALIE_URL", "").rstrip("/")
MEALIE_TOKEN = os.getenv("MEALIE_TOKEN", "")

logger = logging.getLogger("meal_planning")
if not logger.handlers:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


@app.on_event("startup")
async def _startup_refresh_mealie():
    try:
        _refresh_mealie_buffer()
    except Exception as e:
        logger.warning("Failed to refresh Mealie buffer: %s", e)


def _validate_date_str(monday: str) -> None:
    if not DATE_RE.match(monday):
        raise HTTPException(status_code=400, detail="Invalid date format, expected YYYY-MM-DD")
    try:
        datetime.strptime(monday, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date value")


def _week_monday(date_obj: date) -> date:
    return date_obj - timedelta(days=date_obj.weekday())


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


def _build_regen_prompt(
    monday: str,
    planning: List[Dict[str, Any]],
    day: str,
    repas: str,
    mode: str = "vegetarien",
    required_ingredients: Optional[List[str]] = None,
) -> str:
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
    "duree_preparation_minutes": 30,
    "restes": [
      "Soupe de légumes"
    ]
  }


les repas devraient être variés par rapport à la liste des repas existant, {{json}}
"""
    mode_hint = ""
    if mode == "gourmand":
        mode_hint = "\nVersion gourmande: plus savoureux, viande ou poisson autorisés, tout en restant de saison."
    elif mode == "inspire":
        req = [x for x in (required_ingredients or []) if x]
        if req:
            mode_hint = "\nInspiration: inclure obligatoirement ces ingrédients: " + ", ".join(req) + "."
    extra = f"\nGénère exactement un objet JSON pour le jour '{day}' et le repas '{repas}', au format de l'exemple ci-dessus, sans texte additionnel." + mode_hint
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


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
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
    logger.info("OpenAI request (single) model=%s", OPENAI_MODEL)
    logger.info("OpenAI payload (single): %s", json.dumps(payload, ensure_ascii=False))
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}") from e
    logger.info("OpenAI response (single) status=%s", r.status_code)
    logger.info("OpenAI body (single): %s", r.text)
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=500, detail=f"OpenAI error: {detail}")
    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _generate_meal(
    monday: str,
    day: str,
    repas: str,
    planning: List[Dict[str, Any]],
    mode: str = "vegetarien",
    required_ingredients: Optional[List[str]] = None,
) -> Dict[str, Any]:
    buffer_meals = _load_meal_buffer(monday)
    if not buffer_meals or mode != "vegetarien" or required_ingredients:
        logger.info("Meal buffer empty for %s, generating new pool", monday)
        prompt = _build_regen_prompt(monday, planning, day, repas, mode=mode, required_ingredients=required_ingredients)
        resp = _call_openai(prompt)
        meal = _extract_json_from_text(resp)
        return _normalize_meal(day, repas, meal)

    meal = buffer_meals.pop(0)
    _save_meal_buffer(monday, buffer_meals)
    return _normalize_meal(day, repas, meal)


def _generate_mealie_meal(monday: str, day: str, repas: str) -> Dict[str, Any]:
    recipe = _pop_mealie_recipe()
    if not recipe:
        raise HTTPException(status_code=503, detail="Aucune recette Mealie disponible")
    meal = _mealie_recipe_to_meal(recipe, day, repas)
    return _normalize_meal(day, repas, meal)


def _mealie_recipe_to_meal(recipe: Dict[str, Any], day: str, repas: str) -> Dict[str, Any]:
    return {
        "jour": day,
        "repas": repas,
        "plats": [recipe.get("name", "")] if recipe.get("name") else [],
        "ingredients": recipe.get("ingredients") or [],
        "courses": recipe.get("ingredients") or [],
        "duree_preparation_minutes": recipe.get("duree_preparation_minutes"),
        "restes": [],
        "mealie_slug": recipe.get("slug"),
    }


def _dedupe_strings(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not isinstance(item, str):
            continue
        val = item.strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _append_mealie_dish(meal: Dict[str, Any], recipe: Dict[str, Any]) -> Dict[str, Any]:
    name = str(recipe.get("name") or "").strip()
    if not name:
        return meal
    plats = meal.get("plats") if isinstance(meal.get("plats"), list) else []
    plats = [x for x in plats if isinstance(x, str)]
    if name not in plats:
        plats.append(name)
    meal["plats"] = plats

    ing = recipe.get("ingredients") or []
    ing = [x for x in ing if isinstance(x, str)]
    current = meal.get("courses") if isinstance(meal.get("courses"), list) else meal.get("ingredients") or []
    current = [x for x in current if isinstance(x, str)]
    meal["courses"] = _dedupe_strings(current + ing)
    meal["ingredients"] = meal["courses"]

    restes = meal.get("restes") if isinstance(meal.get("restes"), list) else []
    restes = [x for x in restes if isinstance(x, str)]
    if name not in restes:
        restes.append(name)
    meal["restes"] = restes

    mealie_dishes = meal.get("mealie_dishes")
    if not isinstance(mealie_dishes, list):
        mealie_dishes = []
    exists = any(isinstance(d, dict) and str(d.get("name") or "") == name for d in mealie_dishes)
    if not exists:
        mealie_dishes.append(
            {
                "name": name,
                "slug": recipe.get("slug"),
                "ingredients": ing,
            }
        )
    meal["mealie_dishes"] = mealie_dishes
    return meal


def _remove_mealie_dish(meal: Dict[str, Any], dish: str) -> Dict[str, Any]:
    target = dish.strip().lower()
    plats = meal.get("plats") if isinstance(meal.get("plats"), list) else []
    plats = [x for x in plats if isinstance(x, str)]
    meal["plats"] = [p for p in plats if p.strip().lower() != target]

    restes = meal.get("restes") if isinstance(meal.get("restes"), list) else []
    restes = [x for x in restes if isinstance(x, str)]
    meal["restes"] = [r for r in restes if r.strip().lower() != target]

    mealie_dishes = meal.get("mealie_dishes")
    if isinstance(mealie_dishes, list):
        kept = []
        removed_ingredients: List[str] = []
        for d in mealie_dishes:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "").strip()
            if name.strip().lower() == target:
                removed_ingredients.extend([x for x in d.get("ingredients") or [] if isinstance(x, str)])
                continue
            kept.append(d)
        meal["mealie_dishes"] = kept
        if removed_ingredients:
            current = meal.get("courses") if isinstance(meal.get("courses"), list) else meal.get("ingredients") or []
            current = [x for x in current if isinstance(x, str)]
            remove_keys = {x.strip().lower() for x in removed_ingredients if isinstance(x, str)}
            filtered = [x for x in current if x.strip().lower() not in remove_keys]
            meal["courses"] = _dedupe_strings(filtered)
            meal["ingredients"] = meal["courses"]

    if not meal.get("plats"):
        meal["restes"] = []
        meal["courses"] = []
        meal["ingredients"] = []
        meal["mealie_dishes"] = []

    return meal


def _normalize_meal(day: str, repas: str, meal: Dict[str, Any]) -> Dict[str, Any]:
    meal = dict(meal or {})
    meal["jour"] = day
    meal["repas"] = repas
    meal["plats"] = meal.get("plats") if isinstance(meal.get("plats"), list) else []
    meal["courses"] = _coalesce_courses(meal)
    meal["ingredients"] = meal["courses"]
    meal["restes"] = meal.get("restes") if isinstance(meal.get("restes"), list) else []
    if not isinstance(meal.get("duree_preparation_minutes"), (int, float)):
        meal["duree_preparation_minutes"] = None
    else:
        meal["duree_preparation_minutes"] = int(meal["duree_preparation_minutes"])
    return meal


def _meal_ingredients(meal: Dict[str, Any]) -> List[str]:
    return _coalesce_courses(meal)


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
        request,
        "index.html",
        {
            "dates": _list_available(),
        },
    )


@app.get("/meal-planning/tonight")
def get_tonight_meal(request: Request):
    today = datetime.now().date()
    monday = _week_monday(today).strftime("%Y-%m-%d")
    planning = _load_planning(monday)
    day_names = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    day_name = day_names[today.weekday()]
    meal = None
    for m in planning:
        if not isinstance(m, dict):
            continue
        if str(m.get("jour", "")).strip().lower() != day_name:
            continue
        if str(m.get("repas", "")).strip().lower() != "soir":
            continue
        meal = m
        break
    if not meal:
        raise HTTPException(status_code=404, detail="No meal found for tonight")
    planning_url = str(request.base_url).rstrip("/") + f"/meal-planning/{monday}"
    return JSONResponse(
        content={
            "date": today.strftime("%Y-%m-%d"),
            "jour": day_name,
            "monday": monday,
            "planning_url": planning_url,
            "meal": meal,
        }
    )


WEEK_DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _draft_path(monday: str) -> Path:
    return DATA_DIR / "drafts" / f"{monday}.json"


def _confirmed_path(monday: str) -> Path:
    return DATA_DIR / "confirmed" / monday


def _load_draft(monday: str) -> Dict[str, Any]:
    path = _draft_path(monday)
    if not path.exists():
        return {"choices": {}, "generated": {}, "locked": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("choices"), dict) and isinstance(data.get("generated"), dict):
            return data
    except (ValueError, OSError):
        pass
    raise HTTPException(status_code=500, detail="Stored draft is corrupted")


def _sync_only_planning(monday: str) -> bool:
    if not _path_for(monday).exists() or _confirmed_path(monday).exists():
        return False
    planning = _load_planning(monday)
    return bool(planning) and all(isinstance(meal, dict) and meal.get("mealie_plan_sync") for meal in planning)


def _save_draft(monday: str, draft: Dict[str, Any]) -> None:
    path = _draft_path(monday)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _refresh_draft(monday: str) -> Dict[str, Any]:
    if _path_for(monday).exists() and not _sync_only_planning(monday):
        raise HTTPException(status_code=409, detail="La semaine est déjà validée")
    draft = _load_draft(monday)
    if _sync_only_planning(monday) and not draft.get("locked"):
        draft["locked"] = _load_planning(monday)
    try:
        entries = _fetch_mealie_plan(monday)
        if entries is not None:
            locked = list(_mealie_planned_meals(monday, entries).values())
            if locked != draft.get("locked"):
                draft["locked"] = locked
                _save_draft(monday, draft)
    except (requests.RequestException, ValueError, TypeError) as error:
        logger.warning("Failed to refresh draft calendar for %s: %s", monday, error)
    return draft


def _draft_conflicts(draft: Dict[str, Any]) -> List[str]:
    locked_days = {meal.get("jour") for meal in draft.get("locked", []) if meal.get("repas") == "soir"}
    return [day for day in WEEK_DAYS if day in locked_days and (day in draft["choices"] or day in draft["generated"])]


def _draft_payload(monday: str, draft: Dict[str, Any]) -> Dict[str, Any]:
    mapping = seasonality.load(SEASONALITY_FILE)
    recipes = _load_mealie_buffer()
    start = date.fromisoformat(monday)
    suggestions: Dict[str, List[Dict[str, Any]]] = {}
    used_slugs = {str(meal.get("mealie_slug")) for meal in draft["choices"].values() if meal.get("mealie_slug")}
    used_slugs.update(str(dish.get("slug")) for meal in draft.get("locked", []) for dish in meal.get("mealie_dishes", []) if isinstance(dish, dict) and dish.get("slug"))
    for offset, day in enumerate(WEEK_DAYS):
        month = (start + timedelta(days=offset)).month
        ranked = [{"slug": recipe.get("slug"), "name": recipe.get("name"), "image_url": _mealie_image_url(recipe),
                   "seasonality": seasonality.recipe_score(recipe, month, mapping)} for recipe in recipes]
        ranked = [item for item in ranked if item["seasonality"]["grade"] in {"A", "B", "C"}]
        ranked.sort(key=lambda item: (item["seasonality"]["score"] is None,
                                      -(item["seasonality"]["score"] or 0),
                                      -item["seasonality"]["coverage"] if item["seasonality"]["coverage"] is not None else 0,
                                      str(item.get("name") or "").casefold()))
        first_unused = next((item for item in ranked if item.get("slug") not in used_slugs), None)
        if first_unused is not None:
            ranked.remove(first_unused)
            ranked.insert(0, first_unused)
            used_slugs.add(str(first_unused.get("slug")))
        suggestions[day] = ranked
    image_urls = {str(recipe["slug"]): url for recipe in recipes
                  if recipe.get("slug") and (url := _mealie_image_url(recipe))}
    return {"monday": monday, **draft, "conflicts": _draft_conflicts(draft),
            "suggestions": suggestions, "mealie_images": image_urls}


@app.get("/meal-planning/{monday}/draft")
def get_week_draft(monday: str):
    _validate_date_str(monday)
    return JSONResponse(content=_draft_payload(monday, _refresh_draft(monday)))


@app.get("/meal-planning/{monday}/draft/recipe-seasonality/{slug}")
def get_recipe_seasonality(monday: str, slug: str, day: str):
    _validate_date_str(monday)
    if day not in WEEK_DAYS:
        raise HTTPException(status_code=400, detail="Invalid day")
    recipe = next((item for item in _load_mealie_buffer() if item.get("slug") == slug), None)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recette Mealie introuvable")
    planned_date = date.fromisoformat(monday) + timedelta(days=WEEK_DAYS.index(day))
    mapping = seasonality.load(SEASONALITY_FILE)
    return JSONResponse(content={
        "slug": slug,
        "date": planned_date.isoformat(),
        "seasonality": seasonality.recipe_score(recipe, planned_date.month, mapping),
        "ingredients": seasonality.ingredient_scores(recipe, planned_date.month, mapping),
    })


@app.post("/meal-planning/{monday}/draft/choice")
async def choose_draft_recipe(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        body = await request.json()
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from error
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    day = str(body.get("day") or "").lower()
    if day not in WEEK_DAYS:
        raise HTTPException(status_code=400, detail="Invalid day")
    draft = _refresh_draft(monday)
    slug = str(body.get("slug") or "").strip()
    if slug:
        if any(meal.get("jour") == day and meal.get("repas") == "soir" for meal in draft.get("locked", [])):
            raise HTTPException(status_code=409, detail="Repas Mealie verrouillé")
        recipe = next((r for r in _load_mealie_buffer() if r.get("slug") == slug), None)
        if recipe is None:
            raise HTTPException(status_code=404, detail="Recette Mealie introuvable")
        draft["choices"][day] = _mealie_recipe_to_meal(recipe, day, "soir")
    else:
        draft["choices"].pop(day, None)
    draft["generated"].pop(day, None)
    _save_draft(monday, draft)
    return JSONResponse(content=_draft_payload(monday, draft))


def _seasonal_prompt_summary(monday: str, mapping: Dict[str, Any]) -> str:
    months = {(date.fromisoformat(monday) + timedelta(days=i)).month for i in range(7)}
    peak = []
    avoid = []
    for food in mapping.get("foods", {}).values():
        if food.get("neutral") or not food.get("name"):
            continue
        values = [food.get("months", {}).get(f"{month:02d}") for month in months]
        if values and all(value == 2 for value in values):
            peak.append(food["name"])
        elif values and all(value == 0 for value in values):
            avoid.append(food["name"])
    return f"Pleine saison: {', '.join(sorted(set(peak))[:30])}. Hors saison: {', '.join(sorted(set(avoid))[:30])}."


@app.post("/meal-planning/{monday}/draft/complete")
def complete_week_draft(monday: str):
    _validate_date_str(monday)
    draft = _refresh_draft(monday)
    if _draft_conflicts(draft):
        raise HTTPException(status_code=409, detail="Résolvez les conflits avec le calendrier Mealie")
    locked_days = {meal["jour"] for meal in draft.get("locked", []) if meal.get("repas") == "soir"}
    missing = [day for day in WEEK_DAYS if day not in locked_days and day not in draft["choices"] and day not in draft["generated"]]
    if not missing:
        return JSONResponse(content=_draft_payload(monday, draft))
    selected = [meal.get("plats") for meal in draft.get("locked", []) if meal.get("repas") == "soir"]
    selected += [meal.get("plats") for meal in draft["choices"].values()]
    prompt = (
        f"Compose exactement les dîners manquants pour la semaine du {monday}, Sud-Ouest de la France, "
        "pour 2 adultes et 2 enfants. Recettes familiales, peu transformées, moins de 45 minutes, variées. "
        f"Jours à remplir: {', '.join(missing)}. Repas déjà choisis à ne pas répéter: {json.dumps(selected, ensure_ascii=False)}. "
        f"{_seasonal_prompt_summary(monday, seasonality.load(SEASONALITY_FILE))} "
        "Réponds uniquement par un tableau JSON avec exactement un objet par jour manquant. "
        "Chaque objet contient jour, repas='soir', plats (liste), ingredients (liste), "
        "duree_preparation_minutes (nombre), restes (liste)."
    )
    try:
        generated = _extract_json_from_text(_call_openai(prompt))
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=502, detail=f"Réponse IA invalide: {error}") from error
    if not isinstance(generated, list) or len(generated) != len(missing):
        raise HTTPException(status_code=502, detail="L’IA n’a pas fourni tous les dîners demandés")
    by_day: Dict[str, Dict[str, Any]] = {}
    for meal in generated:
        if not isinstance(meal, dict) or meal.get("jour") not in missing or meal.get("jour") in by_day or not isinstance(meal.get("plats"), list) or not meal["plats"] or not isinstance(meal.get("ingredients"), list):
            raise HTTPException(status_code=502, detail="La réponse IA contient un dîner invalide")
        by_day[meal["jour"]] = _normalize_meal(meal["jour"], "soir", meal)
    if set(by_day) != set(missing):
        raise HTTPException(status_code=502, detail="La réponse IA omet un dîner")
    draft["generated"].update(by_day)
    _save_draft(monday, draft)
    return JSONResponse(content=_draft_payload(monday, draft))


@app.post("/meal-planning/{monday}/draft/confirm")
def confirm_week_draft(monday: str):
    _validate_date_str(monday)
    draft = _refresh_draft(monday)
    if _draft_conflicts(draft):
        raise HTTPException(status_code=409, detail="Résolvez les conflits avec le calendrier Mealie")
    locked = draft.get("locked", [])
    locked_days = {meal["jour"] for meal in locked if meal.get("repas") == "soir"}
    missing = [day for day in WEEK_DAYS if day not in locked_days and day not in draft["choices"] and day not in draft["generated"]]
    if missing:
        raise HTTPException(status_code=409, detail=f"Dîners manquants: {', '.join(missing)}")
    planning = locked + [draft["choices"].get(day) or draft["generated"][day] for day in WEEK_DAYS if day not in locked_days]
    _save_planning(monday, planning)
    ingredients = _dedupe_strings([item for meal in planning for item in _meal_ingredients(meal)])
    notes = {item.casefold(): ", ".join(meal.get("plats") or []) for meal in planning for item in _meal_ingredients(meal)}
    _save_shopping_list(monday, ingredients, notes=notes)
    _confirmed_path(monday).parent.mkdir(parents=True, exist_ok=True)
    _confirmed_path(monday).touch()
    _draft_path(monday).unlink(missing_ok=True)
    return JSONResponse(status_code=201, content={"monday": monday, "planning": planning})


@app.get("/meal-planning/{monday}", response_class=HTMLResponse)
def get_planning(request: Request, monday: str):
    _validate_date_str(monday)
    p = _path_for(monday)
    if p.exists() and not _sync_only_planning(monday):
        _sync_mealie_week(monday)
    if not p.exists() or _sync_only_planning(monday):
        if p.exists() and _wants_json(request):
            return JSONResponse(content=_load_planning(monday))
        # The draft is created by its API so opening the page is read-only.
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "monday": monday,
                "mealie_url": MEALIE_URL,
            },
        )

    if _wants_json(request):
        planning = _load_planning(monday)
        return JSONResponse(content=planning)

    # page planning (HTML) + JS qui refait un GET Accept: application/json
    return templates.TemplateResponse(
        request,
        "planning.html",
        {
            "monday": monday,
            "mealie_url": MEALIE_URL,
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
    _confirmed_path(monday).parent.mkdir(parents=True, exist_ok=True)
    _confirmed_path(monday).touch()
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
        request,
        "shopping_list.html",
        {
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
    mode = (body.get("mode") or "vegetarien").strip().lower()
    ingredients = body.get("ingredients") or body.get("ingredients_text") or ""
    required: Optional[List[str]] = None
    if isinstance(ingredients, list):
        required = [str(x).strip() for x in ingredients if str(x).strip()]
    elif isinstance(ingredients, str) and ingredients.strip():
        required = [x.strip() for x in ingredients.split(",") if x.strip()]
    if not day:
        raise HTTPException(status_code=400, detail="Missing jour/day")

    planning = _load_planning(monday)
    new_meal = _generate_meal(monday, day, repas, planning, mode=mode, required_ingredients=required)
    old_meal: Optional[Dict[str, Any]] = None

    replaced = False
    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            old_meal = m
            if m.get("mealie_plan_sync"):
                new_meal["mealie_plan_sync"] = True
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


@app.post("/meal-planning/{monday}/regenerate-mealie")
async def regenerate_mealie(request: Request, monday: str):
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
    recipe = _pop_mealie_recipe()
    if not recipe:
        raise HTTPException(status_code=503, detail="Aucune recette Mealie disponible")
    old_meal: Optional[Dict[str, Any]] = None
    new_meal: Dict[str, Any]

    replaced = False
    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            old_meal = copy.deepcopy(m)
            updated = copy.deepcopy(m)
            _append_mealie_dish(updated, recipe)
            new_meal = _normalize_meal(day, repas, updated)
            planning[i] = new_meal
            replaced = True
            break
    if not replaced:
        base = _normalize_meal(day, repas, {"jour": day, "repas": repas, "plats": [], "ingredients": [], "restes": []})
        _append_mealie_dish(base, recipe)
        new_meal = base
        planning.append(new_meal)
    _save_planning(monday, planning)
    try:
        _update_shopping_list_after_substitution(monday, old_meal, new_meal)
    except HTTPException:
        pass
    return JSONResponse(content={"meal": new_meal, "monday": monday})


def _add_mealie_recipe_to_planning(monday: str, day: str, repas: str, recipe: Dict[str, Any]) -> Dict[str, Any]:
    planning = _load_planning(monday)
    old_meal: Optional[Dict[str, Any]] = None
    new_meal: Dict[str, Any]

    replaced = False
    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            old_meal = copy.deepcopy(m)
            updated = copy.deepcopy(m)
            _append_mealie_dish(updated, recipe)
            new_meal = _normalize_meal(day, repas, updated)
            planning[i] = new_meal
            replaced = True
            break
    if not replaced:
        base = _normalize_meal(day, repas, {"jour": day, "repas": repas, "plats": [], "ingredients": [], "restes": []})
        _append_mealie_dish(base, recipe)
        new_meal = base
        planning.append(new_meal)
    _save_planning(monday, planning)
    try:
        _update_shopping_list_after_substitution(monday, old_meal, new_meal)
    except HTTPException:
        pass
    return new_meal


@app.post("/meal-planning/{monday}/add-mealie-dish")
async def add_mealie_dish(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    day = (body.get("jour") or body.get("day") or "").strip().lower()
    repas = (body.get("repas") or "soir").strip().lower()
    slug = (body.get("slug") or "").strip()
    name = (
        body.get("name")
        or body.get("recipe_name")
        or body.get("recette")
        or body.get("receipt")
        or ""
    )
    name = str(name).strip()
    if not day:
        raise HTTPException(status_code=400, detail="Missing jour/day")
    if not slug and not name:
        raise HTTPException(status_code=400, detail="Missing recipe slug or name")
    recipe = _lookup_mealie_recipe(slug=slug, name=name)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recette Mealie introuvable")
    new_meal = _add_mealie_recipe_to_planning(monday, day, repas, recipe)
    return JSONResponse(content={"meal": new_meal, "monday": monday})


@app.post("/meal-planning/{monday}/choose-mealie")
async def choose_mealie_recipe(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        body = await request.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from error
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    day = str(body.get("day") or "").strip().lower()
    repas = str(body.get("repas") or "").strip().lower()
    slug = str(body.get("slug") or "").strip()
    if not slug:
        raise HTTPException(status_code=400, detail="Missing recipe slug")
    _editable_planning_meal(monday, day, repas)
    recipe = next((item for item in _catalog_recipes() if item.get("slug") == slug), None)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recette Mealie introuvable")
    planned_date = date.fromisoformat(monday) + timedelta(days=WEEK_DAYS.index(day))
    grade = seasonality.recipe_score(recipe, planned_date.month, seasonality.load(SEASONALITY_FILE))["grade"]
    if grade not in {"A", "B", "C"}:
        raise HTTPException(status_code=400, detail="Cette recette n’a pas un season-score A, B ou C pour ce repas")
    new_meal = _add_mealie_recipe_to_planning(monday, day, repas, recipe)
    return JSONResponse(content={"meal": new_meal, "monday": monday})


@app.post("/meal-planning/{monday}/remove-dish")
async def remove_dish(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    day = (body.get("jour") or body.get("day") or "").strip().lower()
    repas = (body.get("repas") or "soir").strip().lower()
    dish = str(body.get("dish") or body.get("plat") or "").strip()
    if not day:
        raise HTTPException(status_code=400, detail="Missing jour/day")
    if not dish:
        raise HTTPException(status_code=400, detail="Missing dish")

    planning = _load_planning(monday)
    old_meal: Optional[Dict[str, Any]] = None
    new_meal: Optional[Dict[str, Any]] = None

    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            old_meal = copy.deepcopy(m)
            updated = copy.deepcopy(m)
            _remove_mealie_dish(updated, dish)
            new_meal = _normalize_meal(day, repas, updated)
            planning[i] = new_meal
            break
    if new_meal is None:
        raise HTTPException(status_code=404, detail="Meal not found")

    _save_planning(monday, planning)
    try:
        _update_shopping_list_after_substitution(monday, old_meal, new_meal)
    except HTTPException:
        pass
    return JSONResponse(content={"meal": new_meal, "monday": monday})


@app.post("/meal-planning/{monday}/generate-week")
def generate_week(monday: str):
    _validate_date_str(monday)
    return complete_week_draft(monday)


@app.get("/mealie/recipes")
def list_mealie_recipes():
    entries = _load_mealie_buffer()
    if not entries:
        _refresh_mealie_buffer()
        entries = _load_mealie_buffer()
    return JSONResponse(content=entries)


def _catalog_month(month: Optional[int]) -> int:
    value = month if month is not None else date.today().month
    if not 1 <= value <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    return value


def _catalog_recipes() -> List[Dict[str, Any]]:
    recipes = _load_mealie_buffer()
    if not recipes:
        _refresh_mealie_buffer()
        recipes = _load_mealie_buffer()
    return recipes


def _mealie_image_url(recipe: Dict[str, Any]) -> Optional[str]:
    if recipe.get("id") and recipe.get("image") and recipe.get("slug"):
        return f"/mealie/images/{quote(str(recipe['slug']), safe='')}"
    return None


def _scored_mealie_recipes(month: int, allowed_grades: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    mapping = seasonality.load(SEASONALITY_FILE)
    items = [
        {"slug": recipe.get("slug"), "name": recipe.get("name"), "image_url": _mealie_image_url(recipe),
         "seasonality": seasonality.recipe_score(recipe, month, mapping)}
        for recipe in _catalog_recipes()
    ]
    if allowed_grades is not None:
        items = [item for item in items if item["seasonality"]["grade"] in allowed_grades]
    items.sort(key=lambda item: (
        item["seasonality"]["score"] is None,
        -(item["seasonality"]["score"] or 0),
        str(item.get("name") or "").casefold(),
    ))
    return items


@app.get("/mealie/catalog", response_class=HTMLResponse)
def mealie_catalog(request: Request, month: Optional[int] = None):
    selected_month = _catalog_month(month)
    return templates.TemplateResponse(request, "mealie_catalog.html", {"month": selected_month, "chooser": None})


@app.get("/mealie/catalog/recipes")
def mealie_catalog_recipes(month: Optional[int] = None):
    selected_month = _catalog_month(month)
    return JSONResponse(content={"month": selected_month, "recipes": _scored_mealie_recipes(selected_month)})


@app.get("/mealie/catalog/recipes/{slug}")
def mealie_catalog_recipe(slug: str, month: Optional[int] = None):
    selected_month = _catalog_month(month)
    recipe = next((item for item in _catalog_recipes() if item.get("slug") == slug), None)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recette Mealie introuvable")
    mapping = seasonality.load(SEASONALITY_FILE)
    return JSONResponse(content={
        "slug": slug,
        "name": recipe.get("name"),
        "image_url": _mealie_image_url(recipe),
        "month": selected_month,
        "seasonality": seasonality.recipe_score(recipe, selected_month, mapping),
        "ingredients": seasonality.ingredient_scores(recipe, selected_month, mapping),
    })


def _editable_planning_meal(monday: str, day: str, repas: str) -> Dict[str, Any]:
    if day not in WEEK_DAYS:
        raise HTTPException(status_code=400, detail="Invalid day")
    planning = _load_planning(monday)
    meal = next((item for item in planning if item.get("jour") == day and item.get("repas") == repas), None)
    if meal is None:
        raise HTTPException(status_code=404, detail="Meal not found")
    if meal.get("mealie_plan_sync"):
        raise HTTPException(status_code=409, detail="Repas Mealie verrouillé")
    return meal


@app.get("/meal-planning/{monday}/choose-mealie", response_class=HTMLResponse)
def choose_mealie_page(request: Request, monday: str, day: str, repas: str):
    _validate_date_str(monday)
    _editable_planning_meal(monday, day, repas)
    planned_date = date.fromisoformat(monday) + timedelta(days=WEEK_DAYS.index(day))
    chooser = {"monday": monday, "day": day, "repas": repas, "date": planned_date.isoformat()}
    return templates.TemplateResponse(request, "mealie_catalog.html", {"month": planned_date.month, "chooser": chooser})


@app.get("/meal-planning/{monday}/choose-mealie/options")
def choose_mealie_options(monday: str, day: str, repas: str):
    _validate_date_str(monday)
    _editable_planning_meal(monday, day, repas)
    planned_date = date.fromisoformat(monday) + timedelta(days=WEEK_DAYS.index(day))
    return JSONResponse(content={"month": planned_date.month, "recipes": _scored_mealie_recipes(planned_date.month, {"A", "B", "C"})})


@app.get("/mealie/images/{slug}")
def mealie_image(slug: str):
    recipe = next((item for item in _load_mealie_buffer() if item.get("slug") == slug), None)
    if recipe is None or not _mealie_image_url(recipe) or not MEALIE_URL or not MEALIE_TOKEN:
        raise HTTPException(status_code=404, detail="Image Mealie introuvable")
    recipe_id = quote(str(recipe["id"]), safe="")
    try:
        upstream = requests.get(
            f"{MEALIE_URL}/api/media/recipes/{recipe_id}/images/min-original.webp",
            headers={"Authorization": f"Bearer {MEALIE_TOKEN}", "accept": "image/webp"},
            params={"version": str(recipe["image"])}, timeout=20,
        )
    except requests.RequestException as error:
        logger.warning("Failed to load Mealie image for %s: %s", slug, error)
        raise HTTPException(status_code=502, detail="Image Mealie indisponible") from error
    if upstream.status_code != 200 or not upstream.headers.get("content-type", "").startswith("image/"):
        raise HTTPException(status_code=404, detail="Image Mealie introuvable")
    return Response(content=upstream.content, media_type=upstream.headers["content-type"], headers={"Cache-Control": "private, max-age=3600"})


@app.post("/meal-planning/{monday}/inject-mealie")
async def inject_mealie(request: Request, monday: str):
    _validate_date_str(monday)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    day = (body.get("jour") or body.get("day") or "").strip().lower()
    repas = (body.get("repas") or "soir").strip().lower()
    slug = (body.get("slug") or "").strip()
    name = (
        body.get("name")
        or body.get("recipe_name")
        or body.get("recette")
        or body.get("receipt")
        or ""
    )
    name = str(name).strip()
    if not day:
        raise HTTPException(status_code=400, detail="Missing jour/day")
    if not slug and not name:
        raise HTTPException(status_code=400, detail="Missing recipe slug or name")

    recipe = _lookup_mealie_recipe(slug=slug, name=name)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recette Mealie introuvable")

    planning = _load_planning(monday)
    new_meal = _mealie_recipe_to_meal(recipe, day, repas)
    old_meal: Optional[Dict[str, Any]] = None

    replaced = False
    for i, m in enumerate(planning):
        if str(m.get("jour", "")).strip().lower() == day and str(m.get("repas", "")).strip().lower() == repas:
            old_meal = m
            if m.get("mealie_plan_sync"):
                new_meal["mealie_plan_sync"] = True
            planning[i] = new_meal
            replaced = True
            break
    if not replaced:
        planning.append(new_meal)
    _save_planning(monday, planning)
    try:
        _update_shopping_list_after_substitution(monday, old_meal, new_meal)
    except HTTPException:
        pass
    return JSONResponse(content={"meal": new_meal, "monday": monday})
def _coalesce_courses(meal: Dict[str, Any]) -> List[str]:
    raw = meal.get("courses")
    if not isinstance(raw, list):
        raw = meal.get("ingredients")
    if not isinstance(raw, list):
        return []
    seen = set()
    out: List[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        v = x.strip()
        if not v:
            continue
        if v.lower() in seen:
            continue
        seen.add(v.lower())
        out.append(v)
    return out


def _parse_duration_minutes(raw: Optional[str]) -> Optional[int]:
    if not raw or not isinstance(raw, str):
        return None
    txt = raw.lower()
    iso = re.fullmatch(r"p(?:(\d+)d)?(?:t(?:(\d+)h)?(?:(\d+)m)?)?", txt.strip())
    if iso:
        return sum(int(part or 0) * unit for part, unit in zip(iso.groups(), (1440, 60, 1))) or None
    total = 0
    m = re.findall(r"(\d+)\s*(heure|heures|h)", txt)
    for val, _ in m:
        total += int(val) * 60
    m2 = re.findall(r"(\d+)\s*(minute|minutes|min)", txt)
    for val, _ in m2:
        total += int(val)
    return total or None


def _load_mealie_buffer() -> List[Dict[str, Any]]:
    if not MEALIE_BUFFER_FILE.exists():
        return []
    try:
        data = json.loads(MEALIE_BUFFER_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Mealie buffer corrupted, resetting")
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _save_mealie_buffer(entries: List[Dict[str, Any]]) -> None:
    MEALIE_BUFFER_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_mealie_recipe_detail(slug: str) -> Optional[Dict[str, Any]]:
    if not MEALIE_URL or not MEALIE_TOKEN:
        return None
    url = f"{MEALIE_URL}/api/recipes/{slug}"
    r = requests.get(url, headers={"Authorization": f"Bearer {MEALIE_TOKEN}", "accept": "application/json"}, timeout=30)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _refresh_mealie_buffer() -> None:
    if not MEALIE_URL or not MEALIE_TOKEN:
        logger.info("MEALIE_URL or MEALIE_TOKEN not set, skipping Mealie buffer refresh")
        return
    per_page = 50
    page = 1
    dedup: Dict[str, Dict[str, Any]] = {}
    headers = {"Authorization": f"Bearer {MEALIE_TOKEN}", "accept": "application/json"}
    workers_env = os.getenv("MEALIE_FETCH_WORKERS", "8")
    try:
        workers = max(1, int(workers_env))
    except ValueError:
        workers = 8

    def build_entry(name: str, slug: str, total_time: Optional[str], recipe_id: Any, image_version: Any) -> Optional[Dict[str, Any]]:
        detail = _fetch_mealie_recipe_detail(slug) or {}
        ing = []
        ingredient_refs = []
        for rec in detail.get("recipeIngredient") or []:
            if not isinstance(rec, dict):
                continue
            note = rec.get("display") or rec.get("note") or ""
            note = str(note).strip()
            if note:
                ing.append(note)
                food = rec.get("food") if isinstance(rec.get("food"), dict) else {}
                ingredient_refs.append({"display": note, "food_id": str(food.get("id") or ""), "food_name": str(food.get("name") or "")})
        duree = _parse_duration_minutes(detail.get("totalTime") or total_time)
        return {
            "name": name,
            "slug": slug,
            "id": str(detail.get("id") or recipe_id or ""),
            "image": detail.get("image") or image_version,
            "ingredients": ing,
            "ingredient_refs": ingredient_refs,
            "duree_preparation_minutes": duree,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            url = f"{MEALIE_URL}/api/recipes?orderDirection=desc&page={page}&perPage={per_page}&requireAllCategories=false&requireAllTags=false&requireAllTools=false&requireAllFoods=false"
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                logger.warning("Failed to fetch Mealie page %s: %s", page, r.text)
                break
            payload = r.json()
            items = payload.get("items") or []
            if not items:
                break
            seen_in_page = set()
            futures = []
            for it in items:
                name = str(it.get("name") or "").strip()
                slug = str(it.get("slug") or "").strip()
                if not name or not slug:
                    continue
                key = name.lower()
                if key in dedup or key in seen_in_page:
                    continue
                seen_in_page.add(key)
                futures.append(executor.submit(build_entry, name, slug, it.get("totalTime"), it.get("id"), it.get("image")))
            for fut in as_completed(futures):
                try:
                    entry = fut.result()
                except Exception:
                    continue
                if not entry:
                    continue
                key = str(entry.get("name") or "").strip().lower()
                if key and key not in dedup:
                    dedup[key] = entry
            total_pages = payload.get("total_pages") or payload.get("totalPages") or page
            if page >= total_pages:
                break
            page += 1
    entries = list(dedup.values())
    _save_mealie_buffer(entries)
    logger.info("Mealie buffer refreshed with %s recipes", len(entries))


def _pop_mealie_recipe() -> Optional[Dict[str, Any]]:
    buf = _load_mealie_buffer()
    if not buf:
        _refresh_mealie_buffer()
        buf = _load_mealie_buffer()
    if not buf:
        return None
    recipe = buf.pop(0)
    _save_mealie_buffer(buf)
    return recipe


def _lookup_mealie_recipe(slug: str = "", name: str = "") -> Optional[Dict[str, Any]]:
    def match(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if slug:
            for it in entries:
                if str(it.get("slug") or "").strip() == slug:
                    return it
        if name:
            target = name.strip().lower()
            for it in entries:
                if str(it.get("name") or "").strip().lower() == target:
                    return it
        return None

    entries = _load_mealie_buffer()
    found = match(entries)
    if found:
        return found
    _refresh_mealie_buffer()
    entries = _load_mealie_buffer()
    return match(entries)


MEALIE_MEAL_TYPES = {
    "breakfast": "petit-dejeuner",
    "lunch": "midi",
    "dinner": "soir",
    "side": "accompagnement",
    "snack": "collation",
    "drink": "boisson",
    "dessert": "dessert",
}


def _fetch_mealie_plan(monday: str) -> Optional[List[Dict[str, Any]]]:
    """Return None when Mealie is not configured, distinct from an empty plan."""
    if not MEALIE_URL or not MEALIE_TOKEN:
        return None
    start = date.fromisoformat(monday)
    end = start + timedelta(days=6)
    entries: List[Dict[str, Any]] = []
    page = 1
    while True:
        response = requests.get(
            f"{MEALIE_URL}/api/households/mealplans",
            headers={"Authorization": f"Bearer {MEALIE_TOKEN}", "accept": "application/json"},
            params={"start_date": start.isoformat(), "end_date": end.isoformat(), "page": page, "perPage": 100},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("Invalid Mealie meal plan response")
        entries.extend(item for item in payload["items"] if isinstance(item, dict))
        total_pages = payload.get("total_pages") or payload.get("totalPages") or page
        if page >= int(total_pages):
            return entries
        page += 1


def _mealie_planned_meals(monday: str, entries: List[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    start = date.fromisoformat(monday)
    cached = {str(r.get("slug") or ""): r for r in _load_mealie_buffer()}
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for entry in entries:
        recipe = entry.get("recipe")
        if not isinstance(recipe, dict):
            continue
        try:
            planned_date = date.fromisoformat(str(entry.get("date")))
        except ValueError:
            continue
        if not start <= planned_date <= start + timedelta(days=6):
            continue
        entry_type = str(entry.get("entryType") or "").strip().lower()
        if not entry_type:
            continue
        slug = str(recipe.get("slug") or "").strip()
        name = str(recipe.get("name") or "").strip()
        if not slug or not name:
            continue
        if slug not in cached:
            try:
                detail = _fetch_mealie_recipe_detail(slug) or {}
            except requests.RequestException as error:
                logger.warning("Failed to fetch Mealie recipe %s: %s", slug, error)
                detail = {}
            ingredients = []
            ingredient_refs = []
            for ingredient in detail.get("recipeIngredient") or []:
                if isinstance(ingredient, dict):
                    display = str(ingredient.get("display") or ingredient.get("note") or "").strip()
                    if display:
                        ingredients.append(display)
                        food = ingredient.get("food") if isinstance(ingredient.get("food"), dict) else {}
                        ingredient_refs.append({"display": display, "food_id": str(food.get("id") or ""), "food_name": str(food.get("name") or "")})
            cached[slug] = {
                "slug": slug,
                "name": name,
                "id": str(detail.get("id") or recipe.get("id") or ""),
                "image": detail.get("image") or recipe.get("image"),
                "ingredients": ingredients,
                "ingredient_refs": ingredient_refs,
                "duree_preparation_minutes": _parse_duration_minutes(detail.get("totalTime") or recipe.get("totalTime")),
            }
        key = (["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"][planned_date.weekday()], MEALIE_MEAL_TYPES.get(entry_type, entry_type))
        meal = grouped.setdefault(key, _normalize_meal(*key, {"plats": [], "ingredients": [], "restes": []}))
        current_recipe = dict(cached[slug], name=name)
        _append_mealie_dish(meal, current_recipe)
        meal["restes"] = []
        duration = current_recipe.get("duree_preparation_minutes")
        if isinstance(duration, (int, float)):
            meal["duree_preparation_minutes"] = max(meal.get("duree_preparation_minutes") or 0, int(duration))
        meal["mealie_plan_sync"] = True
    return grouped


def _reconcile_shopping_list(monday: str, before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> None:
    try:
        payload = _load_shopping_list(monday)
    except HTTPException as error:
        if error.status_code == 404:
            return
        raise
    def ingredients(plan: List[Dict[str, Any]]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for meal in plan:
            for ingredient in _meal_ingredients(meal):
                result.setdefault(ingredient.strip().lower(), ingredient)
        return result
    old = ingredients(before)
    new = ingredients(after)
    bought = payload["bought"]
    bought_keys = {item.strip().lower() for item in bought}
    to_buy = [item for item in payload["to_buy"] if item.strip().lower() not in (old.keys() - new.keys())]
    existing = {item.strip().lower() for item in to_buy} | bought_keys
    for key, ingredient in new.items():
        if key not in old and key not in existing:
            to_buy.append(ingredient)
            existing.add(key)
    notes = dict(payload["notes"])
    for key in old.keys() - new.keys():
        notes.pop(key, None)
    for meal in after:
        title = ", ".join(meal.get("plats") or [])
        for ingredient in _meal_ingredients(meal):
            notes[ingredient.strip().lower()] = title
    _save_shopping_list(monday, to_buy, bought, notes)


def _sync_mealie_week(monday: str) -> None:
    if not _path_for(monday).exists():
        return
    try:
        entries = _fetch_mealie_plan(monday)
        if entries is None:
            return
        imported = _mealie_planned_meals(monday, entries)
    except (requests.RequestException, ValueError, TypeError) as error:
        logger.warning("Failed to import Mealie plan for %s: %s", monday, error)
        return
    try:
        before = _load_planning(monday)
    except HTTPException as error:
        if error.status_code != 404:
            raise
        before = []
    if not before and not _path_for(monday).exists() and not imported:
        return
    after = [meal for meal in before if not meal.get("mealie_plan_sync") and (meal.get("jour"), meal.get("repas")) not in imported]
    after.extend(imported.values())
    if after != before:
        _save_planning(monday, after)
        _reconcile_shopping_list(monday, before, after)
