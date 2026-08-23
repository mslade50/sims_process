"""Tests for the live matchup tape: publish_sim_fairs._build_live_matchup_tape
(builder) and kalshi_maker._load_matchup_tape_file (consumer). Fully offline —
no GitHub, no Kalshi. Run: python test_matchup_tape.py"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ["MAKER_SHADOW"] = "1"

import publish_sim_fairs as psf
import kalshi_maker as km

_p = _f = 0


def eq(name, got, want):
    global _p, _f
    if got == want:
        _p += 1
    else:
        _f += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


tmp = Path(tempfile.mkdtemp(prefix="matchup_tape_"))
psf.PROJECT_ROOT = tmp

# Live sim outputs: 3 players x 40 integer draws, live (non-alphabetical) order.
names = ["Scheffler, Scottie", "aberg, ludvig", "MacIntyre, Robert"]
rng = np.random.default_rng(7)
scores = rng.integers(260, 290, size=(3, 40)).astype(float)
np.save(tmp / "final_scores_live_testtourney.npy", scores)
(tmp / "player_names_live_testtourney.json").write_text(json.dumps(names))

repl = {"macintyre, robert": "mcintyre, robert"}  # name_replacements pass-through

# ── builder ──
tbl = psf._build_live_matchup_tape("testtourney", 99, repl, max_draws=None)
eq("builder rows", tbl.num_rows, 3)
eq("builder draws", tbl.num_columns - 1, 40)  # +1 = index column
md = {k.decode(): v.decode() for k, v in tbl.schema.metadata.items()
      if isinstance(k, bytes) and k in (b"tourney", b"event_id", b"source")}
eq("builder meta tourney", md.get("tourney"), "testtourney")
eq("builder meta event", md.get("event_id"), "99")
eq("builder meta source", md.get("source"), "final_scores_live")
idx = tbl.to_pandas().index.tolist()
eq("names normalized + replaced", idx,
   ["scheffler, scottie", "aberg, ludvig", "mcintyre, robert"])
eq("integral scores compress to int16", str(tbl.to_pandas().dtypes.iloc[0]), "int16")

# downsample keeps per-player rows, cuts draws
small = psf._build_live_matchup_tape("testtourney", 99, repl, max_draws=10)
eq("downsampled draws", small.num_columns - 1, 10)

# missing files -> None
eq("no live files -> None",
   psf._build_live_matchup_tape("othertourney", 99, repl), None)

# mismatched names/rows -> None
(tmp / "player_names_live_testtourney.json").write_text(json.dumps(names[:2]))
eq("name/row mismatch -> None",
   psf._build_live_matchup_tape("testtourney", 99, repl), None)
(tmp / "player_names_live_testtourney.json").write_text(json.dumps(names))

# ── consumer ──
tape_path = tmp / "tape.parquet"
pq.write_table(tbl, tape_path)

got_scores, got_names, got_md = km._load_matchup_tape_file(str(tape_path), "testtourney")
eq("consumer shape", got_scores.shape, (3, 40))
eq("consumer names ride the index", got_names,
   ["scheffler, scottie", "aberg, ludvig", "mcintyre, robert"])
eq("consumer meta sim_run_at present", "sim_run_at" in got_md, True)
# orientation round-trip: player 0's draws survive intact
eq("consumer values intact", got_scores[0].tolist(),
   np.round(scores[0]).astype("int16").tolist())

# tourney mismatch fails closed
s2, n2, _ = km._load_matchup_tape_file(str(tape_path), "someothertourney")
eq("consumer tourney mismatch -> None", (s2, n2), (None, None))

# float (non-integral) scores stay float32
np.save(tmp / "final_scores_live_testtourney.npy", scores + 0.25)
ftbl = psf._build_live_matchup_tape("testtourney", 99, repl, max_draws=None)
eq("non-integral stays float32", str(ftbl.to_pandas().dtypes.iloc[0]), "float32")

# ── made-cut mask builder (pairs with the pre-event tape) ──
tdir = tmp / "testtourney"
tdir.mkdir()
fs_pre = rng.integers(260, 290, size=(3, 40))
mask = rng.integers(0, 2, size=(3, 40)).astype(bool)
np.save(tdir / "final_scores.npy", fs_pre)
np.save(tdir / "made_cut.npy", mask)
(tdir / "player_names.json").write_text(json.dumps(names))

# A live outright payload's portfolio tape must use the paired live outputs, not
# the pre-event files above.  The values are intentionally very different.
live_scores = rng.integers(310, 330, size=(3, 40))
live_mask = rng.integers(0, 2, size=(3, 40)).astype(bool)
np.save(tmp / "final_scores_live_testtourney.npy", live_scores)
np.save(tmp / "made_cut_live_testtourney.npy", live_mask)
ltbl = psf._build_tournament_samples(
    "testtourney", 99, "2026-08-23 10:00:00 UTC", repl,
    max_draws=None, use_live=True,
)
eq("live tournament tape values", ltbl.to_pandas().iloc[0].tolist(),
   live_scores[0].astype("int16").tolist())
ltmd = {k.decode(): v.decode() for k, v in ltbl.schema.metadata.items()
        if isinstance(k, bytes) and k in (b"tourney", b"source")}
eq("live tournament tape source", ltmd.get("source"), "final_scores_live")
lm = psf._build_made_cut_mask(
    "testtourney", 99, repl, max_draws=None, use_live=True,
)
eq("live mask values", lm.to_pandas().iloc[0].tolist(),
   live_mask[0].astype("int8").tolist())
lmmd = {k.decode(): v.decode() for k, v in lm.schema.metadata.items()
        if isinstance(k, bytes) and k in (b"tourney", b"source")}
eq("live mask source", lmmd.get("source"), "made_cut_live")

# Requesting live data is fail-closed: an available pre-event pair is not a
# substitute when the live pair is absent.
preonly = tmp / "preonly"
preonly.mkdir()
np.save(preonly / "final_scores.npy", fs_pre)
(preonly / "player_names.json").write_text(json.dumps(names))
eq("live tape does not fall back to pre-event pair",
   psf._build_tournament_samples(
       "preonly", 100, "2026-08-23 10:00:00 UTC", repl, use_live=True,
   ), None)

mtbl = psf._build_made_cut_mask("testtourney", 99, repl, max_draws=None)
eq("mask rows/draws", (mtbl.num_rows, mtbl.num_columns - 1), (3, 40))
mdf = mtbl.to_pandas()
eq("mask values round-trip", mdf.iloc[0].tolist(), mask[0].astype("int8").tolist())
mmd = {k.decode(): v.decode() for k, v in mtbl.schema.metadata.items()
       if isinstance(k, bytes) and k in (b"tourney", b"source")}
eq("mask meta", (mmd.get("tourney"), mmd.get("source")), ("testtourney", "made_cut"))
# downsample stride matches the tape's (same linspace)
m10 = psf._build_made_cut_mask("testtourney", 99, repl, max_draws=10).to_pandas()
sidx = np.linspace(0, 39, 10).round().astype(int)
eq("mask downsample stride matches tape", m10.iloc[0].tolist(),
   mask[0, sidx].astype("int8").tolist())
# shape mismatch vs final_scores -> refused
np.save(tdir / "made_cut.npy", mask[:, :30])
eq("mask/tape shape mismatch -> None",
   psf._build_made_cut_mask("testtourney", 99, repl), None)

print(f"\n{_p} passed, {_f} failed")
shutil.rmtree(tmp, ignore_errors=True)
raise SystemExit(1 if _f else 0)
