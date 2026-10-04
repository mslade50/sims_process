import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import official_round_history as history
os.environ.setdefault('COEFFS_FROM_CACHE', '1')
import sim_inputs


class OfficialHistoryTests(unittest.TestCase):
    def stats(self, **changes):
        values = dict(player_name=['alpha', 'beta'], position=['1', 'CUT'],
                      thru=[18, 18], round=[-4, 2], sg_total=[3., -3.],
                      event_name=['Utah'] * 2)
        values.update(changes)
        return pd.DataFrame(values)

    def test_completion_includes_cut_players(self):
        with self.assertRaisesRegex(ValueError, '18 holes'):
            history.validate_completed_stats(self.stats(thru=[18, 17]), 'Utah', 71)

    def test_rejects_wrong_event_duplicates_missing_scores_and_nonfinite(self):
        for changes in (dict(event_name=['Other'] * 2),
                        dict(player_name=['alpha'] * 2), dict(round=[-4, None]),
                        dict(sg_total=[3, float('inf')]), dict(round=[-4, 1.5])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                history.validate_completed_stats(self.stats(**changes), 'Utah', 71)

    def test_withdrawals_are_excluded_but_full_results_retained(self):
        result = history.validate_completed_stats(
            self.stats(position=['1', 'WD'], thru=[18, None], round=[-4, None]),
            'Utah', 71)
        self.assertEqual(result.player_name.tolist(), ['alpha'])

    def test_official_score_uses_par_plus_score_not_field_relative_sg(self):
        self.assertEqual(history.completed_strokes(dict(round=-4, thru=18), 71), 67)
        self.assertIsNone(history.completed_strokes(dict(round=-4, thru=17), 71))

    def test_history_uses_each_round_forecast_instead_of_live_r4_rows(self):
        config = dict(round_num=1, wind=[9] * 15, dew=[50] * 15,
                      expected_score_1=69.2, wind_r2=[3] * 15,
                      dew_r2=[40] * 15, expected_score_r2=[69.6])
        result = history.historical_config(config)
        self.assertNotIn('wind', result)
        self.assertNotIn('expected_score_1', result)
        self.assertEqual(result['wind_r2'], [3] * 15)
        self.assertEqual(result['expected_score_r2'], [69.6])
        self.assertIn('wind', config)

    def test_dry_run_validates_without_replacing_or_archiving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'r2_live_model.csv'
            self.stats(assumed_r2_strokes=[67, 73]).to_csv(path, index=False)
            before = path.read_bytes()
            field = pd.DataFrame()
            field.attrs.update(event_id=554, event_name='Utah', course_ids=[930])
            with patch('api_utils.fetch_field_updates', return_value=field), patch(
                'api_utils.fetch_live_stats', return_value=self.stats()), patch.dict(
                    os.environ, {'PROVISIONAL_ROUND_INPUT': ''}):
                history.reconcile(root, 3, 554, 930, 'utah', 71, 'key',
                                  lambda *a: self.fail('ran'), dry_run=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / 'sheet_backups').exists())

    def test_rebuilds_chain_archives_provisional_and_checks_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stats(assumed_r2_strokes=[67, 73]).to_csv(root / 'r2_live_model.csv', index=False)
            field = pd.DataFrame({'player_name': ['alpha', 'beta']})
            field.attrs.update(event_id=554, event_name='Utah', course_ids=[930])
            calls = []
            def run(command, label):
                rnd = int(command[command.index('--round') + 1])
                calls.append(rnd)
                self.stats().to_csv(root / f'r{rnd}_live_model.csv', index=False)
                pd.DataFrame({'player_name': ['alpha'], 'my_pred': [0.]}).to_csv(
                    root / f'model_predictions_r{rnd + 1}.csv', index=False)
            with patch('api_utils.fetch_field_updates', return_value=field), patch(
                'api_utils.fetch_live_stats', return_value=self.stats()), patch.dict(
                    os.environ, {}, clear=True):
                history.reconcile(root, 3, 554, 930, 'utah', 71, 'key', run)
            self.assertEqual(calls, [1, 2])
            self.assertNotIn('assumed_r2_strokes', pd.read_csv(root / 'r2_live_model.csv'))
            self.assertEqual(len(list(root.glob('sheet_backups/official_history/*/receipt.json'))), 1)

    def test_rejects_partial_official_results_before_any_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stats(provisional_assumptions=['assumed'] * 2).to_csv(root / 'r2_live_model.csv', index=False)
            field = pd.DataFrame()
            field.attrs.update(event_id=554, event_name='Utah', course_ids=[930])
            with patch('api_utils.fetch_field_updates', return_value=field), patch(
                'api_utils.fetch_live_stats', return_value=self.stats(thru=[18, 17])), patch.dict(
                    os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, '18 holes'):
                    history.reconcile(root, 3, 554, 930, 'utah', 71, 'key', lambda *a: self.fail('ran'))

    def test_ordinary_history_does_not_fetch_or_rebuild(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            'api_utils.fetch_field_updates', side_effect=AssertionError('fetched')):
            history.reconcile(Path(directory), 3, 554, 930, 'utah', 71, 'key', None)

    def test_wrong_event_or_course_and_inherited_override_do_not_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stats(assumed_r2_strokes=[67, 73]).to_csv(root / 'r2_live_model.csv', index=False)
            for metadata in (dict(event_id=999, course_ids=[930]),
                             dict(event_id=554, course_ids=[999])):
                field = pd.DataFrame()
                field.attrs.update(metadata)
                with patch('api_utils.fetch_field_updates', return_value=field), patch.dict(
                    os.environ, {'PROVISIONAL_ROUND_INPUT': ''}):
                    with self.assertRaisesRegex(ValueError, 'identity'):
                        history.reconcile(root, 3, 554, 930, 'utah', 71, 'key', None)
            with patch.dict(os.environ, {'PROVISIONAL_ROUND_INPUT': 'assumptions.json'}):
                with self.assertRaisesRegex(ValueError, 'override'):
                    history.reconcile(root, 3, 554, 930, 'utah', 71, 'key', None)

    def test_sim_loader_uses_official_scores_and_preserves_provisional_guard(self):
        import ast
        import numpy as np
        from sim_health_gate import SimulationHealthError
        source = (Path(__file__).parent / 'round_sim.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'load_known_rounds')
        scope = dict(os=os, pd=pd, np=np, name_replacements={}, PAR=71, CUT_LINE=65,
                     USE_10_SHOT_RULE=False, tourney='utah',
                     SimulationHealthError=SimulationHealthError)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'round_sim.py', 'exec'), scope)
        with patch('provisional_round.current_inputs', return_value=None), patch(
            'os.path.exists', return_value=True), patch('pandas.read_csv', return_value=self.stats()):
            result = scope['load_known_rounds'](1, {}, 71)
            self.assertEqual(result['strokes'][1].tolist(), [67, 73])
        with patch('provisional_round.current_inputs', return_value=None), patch(
            'os.path.exists', return_value=True), patch('pandas.read_csv', return_value=self.stats(
                assumed_r2_strokes=[67, 73])):
            with self.assertRaisesRegex(SimulationHealthError, 'official rebuild'):
                scope['load_known_rounds'](1, {}, 71)


if __name__ == '__main__':
    unittest.main()
