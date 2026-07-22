import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

from publish_sim_fairs import _manifest_keys
from reprice_overflow import _frame, import_overflow, select_new_entries


def entry(player, book="pinnacle"):
    return {
        "event_id": "100", "event_name": "The Open",
        "player_name": player, "market_type": "win", "sportsbook": book,
        "decimal_odds": 241.0, "american_odds": 24000,
        "fair_prob": 0.00466, "fair_american": 21359,
        "edge_pp": 0.045, "edge_ev": 0.123,
        "kelly_stake": 1.28, "discovered_at": "2026-07-15 22:34:00 UTC",
        "key": f"100|{player}|win",
    }


class OverflowImportTests(unittest.TestCase):
    def test_initial_manifest_is_player_market_only(self):
        keys = _manifest_keys("100", [
            {"player_name": "Ryan Fox", "market_type": "winner", "sportsbook": "pinnacle"},
            {"player_name": "ryan fox", "market_type": "win", "sportsbook": "betcris"},
        ])
        self.assertEqual(keys, [{"player_name": "ryan fox", "market_type": "win"}])

    def test_select_dedups_on_player_market_not_book(self):
        entries = [entry("ryan fox", "pinnacle"), entry("ryan fox", "betcris"), entry("rory mcilroy")]
        selected = select_new_entries(entries, ["100"], {("100", "ryan fox", "win")})
        self.assertEqual([item["player_name"] for item in selected], ["rory mcilroy"])

    def test_frame_populates_storage_probability_column(self):
        frame = _frame([entry("ryan fox")])
        self.assertEqual(frame.loc[0, "market_type"], "win")
        self.assertAlmostEqual(frame.loc[0, "simulated_win_prob"], 0.00466)
        self.assertEqual(frame.loc[0, "stake"], 1.28)

    def test_import_is_idempotent_against_sheet_keys(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            path = os.path.join(tmp, "overflow.json")
            payload = {"schema_version": 1, "events": {"100": {"bets": [entry("ryan fox"), entry("rory mcilroy")]}}}
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            store = Mock()
            fake = types.ModuleType("sheets_storage")
            fake.load_finish_position_keys = lambda spreadsheet, event_id: [
                {"player_name": "ryan fox", "market_type": "win"}
            ]
            fake.store_finish_positions = store
            prior = sys.modules.get("sheets_storage")
            sys.modules["sheets_storage"] = fake
            try:
                count = import_overflow(object(), [("100", "The Open", 2026)], path=path)
            finally:
                if prior is None:
                    sys.modules.pop("sheets_storage", None)
                else:
                    sys.modules["sheets_storage"] = prior
            self.assertEqual(count, 1)
            frame = store.call_args.args[0]
            self.assertEqual(frame.loc[0, "player_name"], "rory mcilroy")


if __name__ == "__main__":
    unittest.main()
