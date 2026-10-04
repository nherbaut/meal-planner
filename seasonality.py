"""Monthly Mealie ingredient seasonality. The CLI is the only AI entry point."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

MONTHS = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")


def line_key(value: str) -> str:
    value = " ".join(value.casefold().split())
    # Quantities vary between recipes; identify the ingredient once.
    value = re.sub(r"^[\d\s.,/½¼¾¹²³]+\s*(?:(?:kg|g|ml|cl|l|cs|cc|càs|càc|pièces?|gousses?|tranches?|cuillères?)\b\s*)?", "", value, flags=re.I)
    return value.strip(" ,.-")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "region": "Sud-Ouest de la France", "foods": {}, "lines": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("foods"), dict) or not isinstance(data.get("lines"), dict):
        raise ValueError(f"Invalid seasonality file: {path}")
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def score_grade(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.5:
        return "D"
    return "E"


def ingredient_scores(recipe: dict[str, Any], month: int, mapping: dict[str, Any]) -> list[dict[str, Any]]:
    refs = recipe.get("ingredient_refs") or [{"display": x} for x in recipe.get("ingredients") or []]
    result = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        display = str(ref.get("display") or "").strip()
        food_id = str(ref.get("food_id") or "")
        neutral = False
        if not food_id:
            line = mapping.get("lines", {}).get(line_key(display), {})
            neutral = line.get("neutral") is True
            food_id = str(line.get("food_id") or "")
        food = mapping.get("foods", {}).get(food_id, {})
        neutral = neutral or food.get("neutral") is True
        value = food.get("months", {}).get(f"{month:02d}") if isinstance(food.get("months"), dict) else None
        result.append({
            "name": str(food.get("name") or ref.get("food_name") or display),
            "display": display,
            "score": value if not neutral and type(value) is int and 0 <= value <= 2 else None,
            "status": "neutral" if neutral else "scored" if type(value) is int and 0 <= value <= 2 else "unknown",
        })
    return result


def recipe_score(recipe: dict[str, Any], month: int, mapping: dict[str, Any]) -> dict[str, Any]:
    ingredients = ingredient_scores(recipe, month, mapping)
    relevant = sum(item["status"] != "neutral" for item in ingredients)
    evaluated = sum(item["status"] == "scored" for item in ingredients)
    points = sum(item["score"] for item in ingredients if item["status"] == "scored")
    raw_score = points / (2 * relevant) if evaluated else None
    return {
        "score": round(raw_score, 3) if raw_score is not None else None,
        "grade": score_grade(raw_score),
        "coverage": round(evaluated / relevant, 3) if relevant else None,
        "evaluated": evaluated,
        "relevant": relevant,
    }


def _ask_ai(items: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for new ingredients")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "max_completion_tokens": 6000, "messages": [
            {"role": "system", "content": "Réponds uniquement par un tableau JSON valide. Aucun texte supplémentaire."},
            {"role": "user", "content": instruction + "\n" + json.dumps(items, ensure_ascii=False)},
        ]}, timeout=180,
    )
    response.raise_for_status()
    result = json.loads(response.json()["choices"][0]["message"]["content"])
    if not isinstance(result, list):
        raise ValueError("AI response must be a JSON array")
    return result


def _batches(items: list[Any], size: int = 12):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def update_from_recipes(path: Path, recipes: list[dict[str, Any]]) -> tuple[int, int]:
    data = load(path)
    if not path.exists():
        save(path, data)
    all_foods: dict[str, str] = {}
    all_lines: dict[str, str] = {}
    for recipe in recipes:
        refs = recipe.get("ingredient_refs") or [{"display": x} for x in recipe.get("ingredients") or []]
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            food_id = str(ref.get("food_id") or "")
            display = str(ref.get("display") or "").strip()
            if food_id:
                all_foods[food_id] = str(ref.get("food_name") or display)
            elif display:
                all_lines[line_key(display)] = display
    new_foods = [(food_id, name) for food_id, name in all_foods.items() if food_id not in data["foods"]]
    for batch in _batches(new_foods):
        pending = dict(batch)
        for _ in range(3):
            if not pending:
                break
            result = _ask_ai([{"id": key, "name": name} for key, name in pending.items()],
                "Pour chaque ingrédient, estime la saisonnalité du produit frais dans le Sud-Ouest de la France. "
                "Rends exactement un objet par id: {id, neutral: bool, months: {01:0..2,...,12:0..2}}. "
                "0=hors saison, 1=disponible localement, 2=pleine saison. "
                "Pour sel, épices, conserves, produits secs, viande, etc., neutral=true et months={}. "
                "Ne considère pas la disponibilité par importation comme une saison locale.")
            by_id = {str(item.get("id")): item for item in result if isinstance(item, dict)}
            for key, name in list(pending.items()):
                item = by_id.get(key)
                if item is None or type(item.get("neutral")) is not bool:
                    continue
                neutral = item["neutral"]
                months = item.get("months") or {}
                if not neutral and (not isinstance(months, dict) or any(type(months.get(m)) is not int or months[m] not in (0, 1, 2) for m in MONTHS)):
                    continue
                data["foods"][key] = {"name": name, "neutral": neutral, "months": {} if neutral else {m: months[m] for m in MONTHS}}
                del pending[key]
            save(path, data)
        if pending:
            raise ValueError(f"Missing seasonality for: {', '.join(pending.values())}")
    new_lines = [(key, display) for key, display in all_lines.items() if key not in data["lines"]]
    names = {key: value.get("name", "") for key, value in data["foods"].items()}
    for batch in _batches(new_lines):
        prompts = []
        for key, display in batch:
            cleaned = re.sub(r"^[\d\s.,/½¼¾¹²³]+\s*(?:g|kg|ml|l|cs|cc|pièces?|gousses?)?\s*", "", display, flags=re.I)
            candidates = difflib.get_close_matches(cleaned.casefold(), [n.casefold() for n in names.values()], n=10, cutoff=0.2)
            possible = [{"id": food_id, "name": name} for food_id, name in names.items() if name.casefold() in candidates]
            prompts.append({"key": key, "display": display, "candidates": possible})
        pending = {prompt["key"]: prompt for prompt in prompts}
        for _ in range(3):
            if not pending:
                break
            result = _ask_ai(list(pending.values()),
                "Pour chaque ligne de recette, identifie l'ingrédient principal parmi ses candidats. "
                "Retourne exactement {key, food_id, neutral} par ligne. food_id doit être un id candidat ou null. "
                "neutral=true uniquement si la saison n'a pas de sens (sel, épices, huile, produit sec, etc.). "
                "Si aucun candidat ne convient à un produit saisonnier, food_id=null et neutral=false.")
            by_key = {str(item.get("key")): item for item in result if isinstance(item, dict)}
            for key, prompt in list(pending.items()):
                item = by_key.get(key)
                if item is None or type(item.get("neutral")) is not bool:
                    continue
                allowed = {candidate["id"] for candidate in prompt["candidates"]}
                food_id = item.get("food_id") or None
                if food_id is not None and str(food_id) not in allowed:
                    continue
                data["lines"][key] = {"display": prompt["display"], "food_id": str(food_id) if food_id else None, "neutral": item["neutral"]}
                del pending[key]
            save(path, data)
        if pending:
            raise ValueError(f"Missing ingredient matches for: {', '.join(pending)}")
    return len(new_foods), len(new_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update the reviewable Mealie seasonality JSON")
    parser.add_argument("--no-refresh", action="store_true", help="Use the existing Mealie recipe cache")
    args = parser.parse_args()
    import app
    if not args.no_refresh:
        app._refresh_mealie_buffer()
    foods, lines = update_from_recipes(app.SEASONALITY_FILE, app._load_mealie_buffer())
    print(f"{foods} nouveaux ingrédients et {lines} nouvelles lignes classés : {app.SEASONALITY_FILE}")


if __name__ == "__main__":
    main()
