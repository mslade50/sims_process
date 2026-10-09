import sqlite3
import tempfile
import unittest
import ast
import os
import types
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import numpy as np

from late_field_replacements import (
    replacement_ema20, replacement_category_distributions,
    extend_category_distributions, validated_late_players,
    historical_database_path,
)

CATS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


def distributions(player):
    return pd.DataFrame([{"player_name": player, "category_clean": cat,
                         "mean": 0., "std": 1., "skew": 0., "n_eff": 50.} for cat in CATS])


class LateFieldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "history.db"
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE player_rounds (player_name TEXT, tour TEXT, year INTEGER, "
                "event_id INTEGER, round_num INTEGER, round_date TEXT, sg_total_adj REAL, "
                "sg_ott_adj REAL, sg_app_adj REAL, sg_arg_adj REAL, sg_putt_adj REAL)")
            for i, value in enumerate([1., 2., 3., 999.]):
                connection.execute("INSERT INTO player_rounds VALUES (?, 'pga', 2026, 554, ?, ?, ?, ?, ?, ?, ?)",
                    ["Hardy, Nick", i+1, f"2026-10-0{i+1}", value, *([value/4]*4)])

    def test_ema20_excludes_event_and_future_rounds_and_leaves_db_unchanged(self):
        before = self.db.read_bytes()
        result = replacement_ema20("hardy, nick", "2026-10-04", db_path=self.db)
        alpha = 2/21
        expected = (1*(1-alpha) + 2*alpha)*(1-alpha) + 3*alpha
        self.assertAlmostEqual(result["my_pred"], expected)
        self.assertEqual(result["fallback_history_rounds"], 3)
        self.assertEqual(self.db.read_bytes(), before)

    def test_alias_and_no_history(self):
        result = replacement_ema20("nickname", "2026-10-04", db_path=self.db,
            aliases={"hardy, nick": "nickname"})
        self.assertEqual(result["fallback_history_rounds"], 3)
        with self.assertRaisesRegex(ValueError, "No pre-event"):
            replacement_ema20("unknown", "2026-10-04", db_path=self.db)

    def test_service_account_skips_inaccessible_profiles_and_fetches_own_snapshot(self):
        with patch.dict(os.environ,{"DG_HISTORICAL_DB":"","LOCALAPPDATA":str(self.root)}), \
             patch("late_field_replacements.Path.home",return_value=self.root), \
             patch("late_field_replacements.Path.glob",side_effect=PermissionError("profile denied")), \
             patch("dgdata_fetch.fetch_snapshot",return_value=self.db) as fetch:
            self.assertEqual(historical_database_path(),self.db)
        fetch.assert_called_once_with("dg_historical",out=str(
            self.root/"etr-golf/cache/late-replacements/dg_historical.db"))

    def test_category_history_excludes_current_results(self):
        result = replacement_category_distributions("hardy, nick", "2026-10-04", CATS,
            distributions("locked"), db_path=self.db)
        self.assertTrue((result["std"] > 0).all())
        self.assertTrue((result["mean"] < 1).all())
        self.assertEqual(result.category_clean.tolist(), CATS)

    def test_sparse_category_variance_uses_frozen_prior(self):
        result = replacement_category_distributions("hardy, nick", "2026-10-02", CATS,
            distributions("locked"), db_path=self.db)
        np.testing.assert_array_equal(result["std"], np.ones(4))

    def test_only_validated_late_players_can_extend_distributions(self):
        frozen = distributions("locked")
        catalog = distributions("hardy, nick")
        extended, roster, additions = extend_category_distributions(
            frozen, catalog, ["locked", "hardy, nick"], CATS, {"hardy, nick"})
        pd.testing.assert_frame_equal(extended.iloc[:4], frozen, check_dtype=False)
        self.assertEqual(roster, ["locked"])
        self.assertEqual(additions, ["hardy, nick"])
        with self.assertRaisesRegex(ValueError, "not validated"):
            extend_category_distributions(frozen, catalog, ["locked", "hardy, nick"], CATS, set())

    def test_late_player_markers_must_match_event(self):
        pd.DataFrame([{"player_name": "hardy, nick", "fallback_source": "historical_sg_ema20",
            "fallback_event_id": 554, "fallback_tourney": "utah", "fallback_cutoff": "2026-10-01"}]).to_csv(
                self.root/"r1_live_model.csv", index=False)
        self.assertEqual(validated_late_players(554,"utah",root=self.root), {"hardy, nick":"2026-10-01"})
        self.assertFalse(validated_late_players(557,"biltmore",root=self.root))

    def test_live_merge_uses_replacement_weather_and_preserves_locked_players(self):
        from r1_prediction_artifact import load_matching_r1_predictions, build_r1_prediction_manifest
        import json

        source = Path(__file__).with_name("live_stats_engine.py").read_text(encoding="utf-8")
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=="_merge_r1")
        namespace = {"pd":pd,"np":np,"os":os, "event_ids":[554],"tourney":"utah",
            "load_matching_r1_predictions":load_matching_r1_predictions,
            "replacement_ema20":lambda player,cutoff,**kw: replacement_ema20(player,cutoff,db_path=self.db),
            "name_replacements":{},"wind_override":0,"baseline_wind":0,"course_id":930,
            "dew_calculation":1., "WIND_ARRAYS":{1:[2.]},"DEW_ARRAYS":{1:[8.]},
            "compute_wind_factor":lambda *args: .1,
            "calculate_average_wind":lambda tt,array: array[0],"clean_names":lambda df:df}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"live_merge_test","exec"),namespace)
        players = [f"player {i}" for i in range(12)]
        locked = pd.DataFrame({"player_name":players,"my_pred":.5,"wind_adj1":.3,"dew_adj1":0.,"dew_r1":10.})
        locked.to_csv(self.root/"model_predictions_r1.csv",index=False)
        (self.root/"model_predictions_r1.meta.json").write_text(json.dumps(build_r1_prediction_manifest(
            locked,{"event_id":554,"tourney":"utah","field":players})),encoding="utf-8")
        active = pd.DataFrame({"player_name":players[:-1]+["hardy, nick"],"r1_teetime":"2026-10-04 10:00"})
        fake_inputs=types.ModuleType("sim_inputs");fake_inputs.feed_name_aliases={}
        previous=os.getcwd()
        try:
            os.chdir(self.root)
            with patch.dict("sys.modules",{"sim_inputs":fake_inputs}):
                result=namespace["_merge_r1"](active)
        finally:
            os.chdir(previous)
        self.assertTrue(result.iloc[:-1].pred.eq(.5).all())
        hardy=result.iloc[-1]
        self.assertAlmostEqual(hardy.wind_adj1,.2)
        self.assertAlmostEqual(hardy.dew_adj1,-2.)
        self.assertEqual(hardy.fallback_event_id, "554")


if __name__ == "__main__":
    unittest.main()
