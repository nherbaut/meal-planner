import importlib
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Request

app_module = importlib.import_module("app")
seasonality = importlib.import_module("seasonality")
MONDAY = "2026-10-05"


def calendar_entry(day="2026-10-07", slug="calendar"):
    return {"date": day, "entryType": "dinner", "recipe": {"name": "Repas calendrier", "slug": slug}}


class DraftTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, value in (("DATA_DIR", root), ("MEALIE_BUFFER_FILE", root / "mealie_buffer.json"),
                            ("SEASONALITY_FILE", root / "seasonality.json")):
            patcher = patch.object(app_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app_module._save_mealie_buffer([
            {"name": "Soupe de courge", "slug": "soupe", "ingredients": ["courge"],
             "ingredient_refs": [{"display": "courge", "food_id": "courge", "food_name": "courge"}]},
            {"name": "Salade de tomate", "slug": "salade", "ingredients": ["tomate"],
             "ingredient_refs": [{"display": "tomate", "food_id": "tomate", "food_name": "tomate"}]},
            {"name": "Repas calendrier", "slug": "calendar", "ingredients": ["riz"]},
        ])
        seasonality.save(app_module.SEASONALITY_FILE, {"version": 1, "region": "Sud-Ouest de la France", "foods": {
            "courge": {"name": "courge", "neutral": False, "months": {"10": 2}},
            "tomate": {"name": "tomate", "neutral": False, "months": {"10": 0}},
        }, "lines": {}})


    def choose(self, day, slug):
        class JsonRequest:
            async def json(self):
                return {"day": day, "slug": slug}
        return asyncio.run(app_module.choose_draft_recipe(JsonRequest(), MONDAY))

    def draft(self):
        return json.loads(app_module.get_week_draft(MONDAY).body)

    def test_open_draft_choice_reload_complete_and_confirm(self):
        with patch.object(app_module, "_fetch_mealie_plan", return_value=[calendar_entry()]), \
             patch.object(app_module, "_call_openai") as ai:
            page = app_module.get_planning(Request({"type": "http", "headers": []}), MONDAY)
            self.assertEqual(page.status_code, 200)
            self.assertEqual(ai.call_count, 0)
            draft = self.draft()
            self.assertEqual(draft["locked"][0]["jour"], "mercredi")
            self.assertEqual(draft["suggestions"]["lundi"][0]["slug"], "soupe")
            response = self.choose("lundi", "soupe")
            self.assertEqual(response.status_code, 200)
            self.assertIn("lundi", self.draft()["choices"])
            with self.assertRaises(HTTPException) as error:
                self.choose("mercredi", "salade")
            self.assertEqual(error.exception.status_code, 409)
            missing = ["mardi", "jeudi", "vendredi", "samedi", "dimanche"]
            ai.return_value = json.dumps([{"jour": day, "plats": [f"Repas {day}"], "ingredients": ["lentilles"],
                                           "duree_preparation_minutes": 30, "restes": []} for day in missing])
            generated = app_module.complete_week_draft(MONDAY)
            self.assertEqual(generated.status_code, 200)
            self.assertEqual(set(json.loads(generated.body)["generated"]), set(missing))
            self.assertEqual(ai.call_count, 1)
            self.assertEqual(app_module.confirm_week_draft(MONDAY).status_code, 201)
        saved = app_module._load_planning(MONDAY)
        self.assertEqual(len(saved), 7)
        self.assertEqual(next(meal for meal in saved if meal["jour"] == "mercredi")["plats"], ["Repas calendrier"])
        self.assertIn("courge", app_module._load_shopping_list(MONDAY)["to_buy"])
        self.assertFalse(app_module._draft_path(MONDAY).exists())

    def test_ai_failure_keeps_draft_and_calendar_conflict_blocks_confirmation(self):
        with patch.object(app_module, "_fetch_mealie_plan", return_value=[]):
            self.choose("mercredi", "soupe")
            with patch.object(app_module, "_call_openai", side_effect=ValueError("offline")):
                with self.assertRaises(HTTPException) as error:
                    app_module.complete_week_draft(MONDAY)
                self.assertEqual(error.exception.status_code, 502)
            self.assertIn("mercredi", self.draft()["choices"])
        with patch.object(app_module, "_fetch_mealie_plan", return_value=[calendar_entry()]):
            draft = self.draft()
            self.assertEqual(draft["conflicts"], ["mercredi"])
            with self.assertRaises(HTTPException) as error:
                app_module.confirm_week_draft(MONDAY)
            self.assertEqual(error.exception.status_code, 409)
            self.choose("mercredi", None)
            self.assertEqual(self.draft()["conflicts"], [])

    def test_seasonality_mapping_is_incremental_and_manual_edits_survive(self):
        path = app_module.SEASONALITY_FILE
        recipe = {"ingredient_refs": [{"display": "2 courges", "food_id": "id1", "food_name": "courge"},
                                      {"display": "100 g sel"}]}
        with patch.object(seasonality, "_ask_ai", side_effect=[
            [{"id": "id1", "neutral": False, "months": {m: 2 for m in seasonality.MONTHS}}],
            [{"key": "sel", "food_id": None, "neutral": True}],
        ]) as ai:
            self.assertEqual(seasonality.update_from_recipes(path, [recipe]), (1, 1))
            self.assertEqual(ai.call_count, 2)
        data = seasonality.load(path)
        data["foods"]["id1"]["months"]["10"] = 0
        seasonality.save(path, data)
        with patch.object(seasonality, "_ask_ai", side_effect=AssertionError("AI called twice")):
            self.assertEqual(seasonality.update_from_recipes(path, [recipe]), (0, 0))
        self.assertEqual(seasonality.load(path)["foods"]["id1"]["months"]["10"], 0)
        self.assertEqual(seasonality.line_key("200 g sel"), "sel")

    def test_incomplete_ai_batch_retries_only_missing_ingredient(self):
        path = app_module.SEASONALITY_FILE
        recipes = [{"ingredient_refs": [
            {"display": "courge", "food_id": "a", "food_name": "courge"},
            {"display": "tomate", "food_id": "b", "food_name": "tomate"},
        ]}]
        first = [{"id": "a", "neutral": False, "months": {m: 2 for m in seasonality.MONTHS}}]
        second = [{"id": "b", "neutral": False, "months": {m: 1 for m in seasonality.MONTHS}}]
        with patch.object(seasonality, "_ask_ai", side_effect=[first, second]) as ask:
            self.assertEqual(seasonality.update_from_recipes(path, recipes), (2, 0))
        self.assertEqual([item["id"] for item in ask.call_args_list[1].args[0]], ["b"])
        self.assertIn("a", seasonality.load(path)["foods"])
        self.assertIn("b", seasonality.load(path)["foods"])

    def test_recipe_score_uses_meal_month_and_unknowns_reduce_coverage(self):
        recipe = {"ingredient_refs": [{"food_id": "courge"}, {"food_id": "missing"}]}
        mapping = seasonality.load(app_module.SEASONALITY_FILE)
        october = seasonality.recipe_score(recipe, 10, mapping)
        self.assertEqual(october["score"], 0.5)
        self.assertEqual(october["coverage"], 0.5)
        self.assertEqual(october["grade"], "D")
        self.assertIsNone(seasonality.recipe_score(recipe, 9, mapping)["score"])

    def test_grade_boundaries_and_ingredient_details(self):
        for value, expected in ((1, "A"), (0.9, "A"), (0.899, "B"), (0.8, "B"),
                                (0.799, "C"), (0.7, "C"), (0.699, "D"),
                                (0.5, "D"), (0.499, "E"), (0, "E"), (None, None)):
            self.assertEqual(seasonality.score_grade(value), expected)
        detail = json.loads(app_module.get_recipe_seasonality(MONDAY, "soupe", "lundi").body)
        self.assertEqual(detail["seasonality"]["grade"], "A")
        self.assertEqual(detail["ingredients"][0]["score"], 2)
        self.assertEqual(detail["ingredients"][0]["status"], "scored")

    def test_legacy_calendar_only_week_can_be_confirmed(self):
        days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        old = [app_module._normalize_meal(day, "soir", {
            "plats": ["Repas calendrier"], "ingredients": ["riz"], "restes": [], "mealie_plan_sync": True,
        }) for day in days]
        app_module._save_planning(MONDAY, old)
        with patch.object(app_module, "_fetch_mealie_plan", return_value=[calendar_entry(
                (app_module.date.fromisoformat(MONDAY) + app_module.timedelta(days=i)).isoformat()) for i in range(7)]), \
             patch.object(app_module, "_generate_week", side_effect=AssertionError("AI must not run")):
            page = app_module.get_planning(Request({"type": "http", "headers": []}), MONDAY)
            self.assertIn("Composer la semaine", page.body.decode())
            self.assertEqual(app_module.confirm_week_draft(MONDAY).status_code, 201)
            self.assertTrue(app_module._confirmed_path(MONDAY).exists())
            self.assertIn("Menu de la semaine", app_module.get_planning(Request({"type": "http", "headers": []}), MONDAY).body.decode())


if __name__ == "__main__":
    unittest.main()
