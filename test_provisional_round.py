import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from provisional_round import build_inputs, write_inputs, current_inputs, tee_contract, ENV


class ProvisionalTests(unittest.TestCase):
    def build(self, n=71, cut=65, totals=None, official=None):
        stats = pd.DataFrame({"player_name": [f"p{i:03}" for i in range(n)],
                              "position": range(1, n + 1), "course": "BD",
                              "round": -3, "total": totals or list(range(n)),
                              "thru": [12] + [18] * (n - 1)})
        return build_inputs(stats, official, event_id=554, tourney="utah",
                            course_id=930, par=71, cut=cut, date="2026-10-03")

    def test_cut_includes_ties_preserves_original_holes_and_scores(self):
        data = self.build(totals=list(range(64)) + [64] * 4 + [65] * 3)
        self.assertEqual(len(data["field"]), 68)
        self.assertEqual(data["stats"][0]["thru"], 12)
        self.assertEqual(data["stats"][0]["round"], -3)
        self.assertEqual(data["standings"][0]["assumed_r2_strokes"], 68)

    def test_halves_order_window_and_leftovers(self):
        data = self.build(n=65)
        field = pd.DataFrame(data["field"]).set_index("player_name")
        self.assertEqual(field.loc["p000", "starting_tee"], 1)
        self.assertEqual(field.loc["p064", "starting_tee"], 10)
        self.assertEqual(field.loc["p000", "r3_teetime"], "2026-10-03 11:30")
        self.assertEqual(field.loc["p064", "r3_teetime"], "2026-10-03 11:30")
        self.assertTrue((field.groupby(["r3_teetime", "starting_tee"]).size() <= 3).all())
        self.assertEqual(field.loc["p032", "starting_tee"], 1)
        self.assertEqual(field.loc["p033", "starting_tee"], 10)

    def test_ties_and_leftovers_are_deterministic(self):
        self.assertEqual(self.build(n=7, totals=[0]*7)["field"],
                         self.build(n=7, totals=[0]*7)["field"])

    def test_official_time_preferred(self):
        official = pd.DataFrame([{"player_name": "p000", "r3_teetime": "2026-10-03 10:55",
                                  "starting_tee": 10, "course": "BD"}])
        row = next(r for r in self.build(official=official)["field"] if r["player_name"] == "p000")
        self.assertEqual(row["r3_teetime"], "2026-10-03 10:55")
        self.assertEqual(row["tee_time_source"], "official")

    def test_manifest_hash_identity_and_group_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_inputs(self.build(), Path(directory) / "input.json")
            with patch.dict(os.environ, {ENV: str(path)}):
                self.assertEqual(len(tee_contract(554, 3)["tee_time_names"]), 65)
                with self.assertRaises(ValueError): current_inputs(event_id=555)
                with self.assertRaises(ValueError): current_inputs(target_round=2)
                data = json.loads(path.read_text()); data["cut_total"] = 0
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): current_inputs()

    def test_normal_mode_has_no_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(current_inputs())
            self.assertIsNone(tee_contract(554, 3))

    def test_missing_score_cannot_be_assumed(self):
        with self.assertRaises(ValueError): self.build(n=3, totals=[0, None, 1])

    def test_no_current_odds_keeps_pricing_schema(self):
        import odds_loader
        with patch.object(odds_loader, "_fetch_scraped_json", return_value=None), \
                patch.object(odds_loader, "_fetch_datagolf_api", return_value=pd.DataFrame()):
            odds = odds_loader.load_matchup_odds("round_matchups", api_key="test", round=3)
        self.assertTrue(odds.empty)
        self.assertTrue({"Player 1", "Player 2", "Bookmaker", "P1 Odds", "P2 Odds"} <= set(odds))
        config = {"tourney": "utah", "std_dev": 2.8, "cut_line": 65,
                  "use_10_shot_rule": False, "simulations": 100000, "event_id": 554}
        with patch("sheet_config.load_config", return_value=config):
            import round_sim
        priced = round_sim.calculate_edges(round_sim.price_matchups(odds, {}))
        combined, sharp = round_sim.build_matchup_outputs(priced, 3, {}, {})
        self.assertTrue(combined.empty and sharp.empty)


if __name__ == "__main__": unittest.main()
