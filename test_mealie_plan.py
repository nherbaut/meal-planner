import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

import requests
from fastapi import Request


meal_planning = importlib.import_module("app")
MONDAY = "2026-10-05"


def entry(day, kind, name, slug):
    return {
        "date": day,
        "entryType": kind,
        "recipe": {"name": name, "slug": slug, "totalTime": "PT30M"},
    }


def meal(day, kind, name, ingredient):
    return meal_planning._normalize_meal(day, kind, {
        "plats": [name], "ingredients": [ingredient], "restes": [],
    })


class MealieWeekSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_patch = patch.object(meal_planning, "DATA_DIR", Path(self.tmp.name))
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.buffer_patch = patch.object(meal_planning, "MEALIE_BUFFER_FILE", Path(self.tmp.name) / "buffer.json")
        self.buffer_patch.start()
        self.addCleanup(self.buffer_patch.stop)
        meal_planning._save_mealie_buffer([
            {"name": "Mealie dinner", "slug": "mealie-dinner", "ingredients": ["carotte"], "duree_preparation_minutes": 25},
            {"name": "Mealie lunch", "slug": "mealie-lunch", "ingredients": ["riz"], "duree_preparation_minutes": 15},
        ])

    def test_import_replaces_slots_updates_list_and_removes_deleted_entries(self):
        original = [meal("lundi", "soir", "Local dinner", "pomme"), meal("mardi", "soir", "Local Tuesday", "poire")]
        meal_planning._save_planning(MONDAY, original)
        meal_planning._save_shopping_list(MONDAY, ["pomme", "poire"], ["sel"])
        planned = [entry(MONDAY, "dinner", "Mealie dinner", "mealie-dinner"),
                   entry(MONDAY, "lunch", "Mealie lunch", "mealie-lunch")]
        with patch.object(meal_planning, "_fetch_mealie_plan", return_value=planned):
            meal_planning._sync_mealie_week(MONDAY)
            first = meal_planning._load_planning(MONDAY)
            meal_planning._sync_mealie_week(MONDAY)
            self.assertEqual(first, meal_planning._load_planning(MONDAY))
        self.assertEqual({(m["jour"], m["repas"]) for m in first},
                         {("lundi", "soir"), ("lundi", "midi"), ("mardi", "soir")})
        self.assertEqual(next(m for m in first if m["repas"] == "soir" and m["jour"] == "lundi")["plats"], ["Mealie dinner"])
        shopping = meal_planning._load_shopping_list(MONDAY)
        self.assertEqual(set(shopping["to_buy"]), {"carotte", "riz", "poire"})
        self.assertEqual(shopping["bought"], ["sel"])
        with patch.object(meal_planning, "_fetch_mealie_plan", return_value=[]):
            meal_planning._sync_mealie_week(MONDAY)
        self.assertEqual(meal_planning._load_planning(MONDAY), [original[1]])
        self.assertEqual(meal_planning._load_shopping_list(MONDAY)["to_buy"], ["poire"])

    def test_opening_ai_week_replaces_wednesday_dinner(self):
        days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        meal_planning._save_planning(MONDAY, [meal(day, "soir", "AI dinner", "pomme") for day in days])
        planned = [entry("2026-10-07", "dinner", "Mealie dinner", "mealie-dinner")]
        with patch.object(meal_planning, "_fetch_mealie_plan", return_value=planned):
            response = meal_planning.get_planning(self.json_request(), MONDAY)
        week = json.loads(response.body)
        self.assertEqual(len(week), 7)
        self.assertEqual(next(item for item in week if item["jour"] == "mercredi")["plats"], ["Mealie dinner"])

    def test_new_week_does_not_generate_or_save_on_open(self):
        planned = [entry(MONDAY, "breakfast", "Mealie lunch", "mealie-lunch")]
        with patch.object(meal_planning, "_fetch_mealie_plan", return_value=planned), \
             patch.object(meal_planning, "_generate_week", side_effect=AssertionError("AI must not run")):
            response = meal_planning.get_planning(self.json_request(), MONDAY)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Composer la semaine", response.body.decode())
        self.assertFalse(meal_planning._path_for(MONDAY).exists())

    def test_iso_duration_is_parsed(self):
        self.assertEqual(meal_planning._parse_duration_minutes("PT1H30M"), 90)

    def test_fetches_full_week_and_all_pages(self):
        first = Mock()
        first.json.return_value = {"items": [entry(MONDAY, "dinner", "First", "first")], "totalPages": 2}
        second = Mock()
        second.json.return_value = {"items": [entry("2026-10-11", "lunch", "Second", "second")], "totalPages": 2}
        with patch.object(meal_planning, "MEALIE_URL", "https://mealie.example"), \
             patch.object(meal_planning, "MEALIE_TOKEN", "token"), \
             patch.object(meal_planning.requests, "get", side_effect=[first, second]) as get:
            result = meal_planning._fetch_mealie_plan(MONDAY)
        self.assertEqual(len(result), 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"], {
            "start_date": MONDAY, "end_date": "2026-10-11", "page": 1, "perPage": 100,
        })
        self.assertEqual(get.call_args_list[1].kwargs["params"]["page"], 2)

    def test_failed_mealie_request_preserves_saved_week(self):
        original = [meal("lundi", "soir", "Local dinner", "pomme")]
        meal_planning._save_planning(MONDAY, original)
        with patch.object(meal_planning, "_fetch_mealie_plan", side_effect=requests.ConnectionError("offline")):
            response = meal_planning.get_planning(self.json_request(), MONDAY)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), original)

    def test_all_html_routes_render_with_request_first_signature(self):
        request = Request({"type": "http", "headers": []})
        with patch.object(meal_planning, "_fetch_mealie_plan", return_value=[]):
            self.assertIn("Compléter avec l’IA", meal_planning.get_planning(request, MONDAY).body.decode())
            meal_planning._save_planning(MONDAY, [meal("lundi", "soir", "Local dinner", "pomme")])
            self.assertIn("Menu de la semaine", meal_planning.get_planning(request, MONDAY).body.decode())
        self.assertIn("Mes semaines", meal_planning.list_plannings(request).body.decode())
        meal_planning._save_shopping_list(MONDAY, ["pomme"])
        self.assertIn("pomme", meal_planning.shopping_list(request, MONDAY).body.decode())

    @staticmethod
    def json_request():
        return Request({"type": "http", "headers": [(b"accept", b"application/json")]})

    def test_multiple_recipes_share_one_slot(self):
        meal_planning._save_mealie_buffer([
            {"name": "First", "slug": "first", "ingredients": ["riz"], "duree_preparation_minutes": 10},
            {"name": "Second", "slug": "second", "ingredients": ["carotte"], "duree_preparation_minutes": 20},
        ])
        grouped = meal_planning._mealie_planned_meals(MONDAY, [
            entry(MONDAY, "dinner", "First", "first"), entry(MONDAY, "dinner", "Second", "second")
        ])
        self.assertEqual(grouped[("lundi", "soir")]["plats"], ["First", "Second"])
        self.assertEqual(grouped[("lundi", "soir")]["ingredients"], ["riz", "carotte"])
        self.assertEqual(grouped[("lundi", "soir")]["duree_preparation_minutes"], 20)


if __name__ == "__main__":
    unittest.main()
