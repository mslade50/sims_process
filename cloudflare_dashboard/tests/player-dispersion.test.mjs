import assert from 'node:assert/strict';
import test from 'node:test';
import {DEFAULT_APP_BANDS, DEFAULT_TEE_BANDS, bandIndex, bandLabel, bandStats, compare, dispersionKey, headline, laterality, parseIndex, parsePayload,
  parseReference, quantile, referenceBand, roundCount, selectPoints, tourSd, viewWindow, windowSpan} from '../app/player-dispersion-rules.ts';

const index = {schema_version: 'player_dispersion.index.v1', generation: 'g', prefix: 'golfprice/player_profiles/dispersion-abc/', reference_key: 'golfprice/player_profiles/dispersion-abc/reference.json', as_of: '2026-10-10', count: 2, player_ids: [7, 9]};
// half-yard integers: tee shots (forward 600=300 yd, lateral 20=10 yd right), approach shots
const payload = {
  schema_version: 'player_dispersion.v1', dg_id: 7, as_of: '2026-10-10',
  rounds: [['2026-09-01', 4, 'E'], ['2026-08-31', 3, 'E'], ['2026-08-30', 2, 'E']],
  tee: {r: [0, 0, 1, 2], f: [600, 560, 640, 500], l: [20, -10, 0, 90], z: 'FRF?'},
  app: {r: [0, 0, 1, 2], s: [240, 300, 400, 420], l: [10, -6, 0, 40], a: [-8, 4, 0, 30], z: 'GRGG', o: 'FRFT'},
};
const ref = {schema_version: 'player_dispersion.v1', as_of: '2026-10-10', max_rounds: 100, n_players: 900, min_player_shots: 30, tee_bands: DEFAULT_TEE_BANDS, app_bands: DEFAULT_APP_BANDS,
  tee: {'2': {n_shots: 1000, n_players: 400, lat: {n: 1000, mean: 0, sd: 19, q10: -24, q90: 23, player_sd: {n_players: 300, p10: 15, p50: 18, p90: 23}}, fairway_rate: .62}},
  app: {all: {'1': {n_shots: 1000, n_players: 400, lat: {n: 1000, mean: 0, sd: 7, q10: -8, q90: 7, player_sd: {n_players: 300, p10: 5, p50: 6.5, p90: 8}}, green_rate: .72}}}};

test('index and payload parsing are strict about identity, schema and array shape', () => {
  assert.ok(parseIndex(index));
  assert.equal(parseIndex({...index, prefix: 'golfprice/other/'}), null);
  assert.equal(parseIndex({...index, reference_key: 'elsewhere/reference.json'}), null);
  assert.equal(parseIndex({...index, schema_version: 'x'}), null);
  assert.equal(dispersionKey(parseIndex(index), 7), 'golfprice/player_profiles/dispersion-abc/7.json');
  assert.equal(dispersionKey(parseIndex(index), 8), null);
  assert.equal(dispersionKey(parseIndex(index), null), null);
  assert.ok(parsePayload(payload, 7));
  assert.equal(parsePayload(payload, 8), null);
  assert.equal(parsePayload({...payload, tee: {...payload.tee, l: [1]}}, 7), null);
  assert.equal(parsePayload({...payload, tee: {...payload.tee, z: 'F'}}, 7), null);
  assert.equal(parsePayload({...payload, schema_version: 'v0'}, 7), null);
  assert.equal(parsePayload({...payload, tee: undefined, app: undefined}, 7), null);
  assert.ok(parseReference(ref)); assert.equal(parseReference({}), null);
});

test('half-yard decoding, round window and band assignment', () => {
  const tee = selectPoints(payload, 'tee', 20, DEFAULT_TEE_BANDS);
  assert.equal(tee.length, 4);
  assert.deepEqual([tee[0].dist, tee[0].lat, tee[0].band], [300, 10, 2]);
  assert.equal(tee[1].lat, -5);
  assert.equal(selectPoints(payload, 'tee', 1, DEFAULT_TEE_BANDS).length, 2, 'only the newest round');
  assert.equal(selectPoints(payload, 'tee', 2, DEFAULT_TEE_BANDS).length, 3);
  assert.equal(roundCount(payload, 'tee'), 3);
  assert.deepEqual(windowSpan(payload, 2), {first: '2026-08-31', last: '2026-09-01', n: 2});
  assert.equal(windowSpan({...payload, rounds: []}, 20), null);
  assert.equal(bandIndex(260, DEFAULT_TEE_BANDS), 1); assert.equal(bandIndex(259.5, DEFAULT_TEE_BANDS), 0); assert.equal(bandIndex(40, DEFAULT_APP_BANDS), -1);
});

test('approach: forward position is start yardage plus long/short; lie filter works', () => {
  const app = selectPoints(payload, 'app', 20, DEFAULT_APP_BANDS);
  assert.equal(app.length, 4);
  assert.deepEqual([app[0].dist, app[0].along, app[0].y, app[0].lat], [120, -4, 116, 5]);
  assert.equal(selectPoints(payload, 'app', 20, DEFAULT_APP_BANDS, 'F').length, 2);
  assert.equal(selectPoints(payload, 'app', 20, DEFAULT_APP_BANDS, 'T').length, 1);
  assert.equal(selectPoints(payload, 'app', 20, DEFAULT_APP_BANDS, 'S').length, 0);
});

test('band statistics: width is the 10th-90th percentile; hit rate ignores unknown-era lies', () => {
  const pts = selectPoints(payload, 'tee', 20, DEFAULT_TEE_BANDS);
  const stats = bandStats(pts, 'tee', DEFAULT_TEE_BANDS);
  const b2 = stats.find((s) => s.band === 2);
  assert.equal(b2.n, 1); assert.equal(b2.meanLat, 10); assert.equal(b2.rate, null, 'fewer than 10 known lies');
  assert.equal(quantile([1, 2, 3, 4, 5], .5), 3);
  const many = Array.from({length: 40}, (_, i) => ({lat: i - 20, y: 300, dist: 300, along: 0, band: 2, round: 0, lie: i % 2 ? 'F' : 'R', start: 'T'}));
  const s = bandStats(many, 'tee', DEFAULT_TEE_BANDS)[0];
  assert.equal(s.n, 40); assert.equal(s.rate, .5); assert.ok(Math.abs(s.width - (quantile(many.map((p) => p.lat).sort((a, b) => a - b), .9) - quantile(many.map((p) => p.lat).sort((a, b) => a - b), .1))) < 1e-9);
  const withUnknown = [...many, ...Array.from({length: 10}, () => ({...many[0], lie: '?'}))];
  assert.equal(bandStats(withUnknown, 'tee', DEFAULT_TEE_BANDS)[0].rateN, 40);
});

test('PGA comparison uses the median player SD, needs 30 shots and words the result plainly', () => {
  assert.equal(tourSd(referenceBand(parseReference(ref), 'tee', 2)), 18);
  assert.equal(referenceBand(parseReference(ref), 'tee', 4), null);
  assert.equal(referenceBand(parseReference(ref), 'app', 1, 'all').green_rate, .72);
  const stat = (sd, n) => ({band: 2, n, meanLat: 0, sdLat: sd, q10: -1, q90: 1, width: 2, meanY: 300, meanAlong: 0, rate: null, rateN: 0});
  const r = referenceBand(parseReference(ref), 'tee', 2);
  assert.equal(compare(stat(13.5, 100), r).tone, 'narrower'); assert.match(compare(stat(13.5, 100), r).text, /25% tighter/);
  assert.match(compare(stat(27, 100), r).text, /50% wider/);
  assert.equal(compare(stat(18.3, 100), r).tone, 'similar');
  assert.equal(compare(stat(13.5, 10), r).tone, 'unknown');
  assert.equal(compare(stat(13.5, 100), null).tone, 'unknown');
  assert.match(headline('tee', [stat(13.5, 100)], DEFAULT_TEE_BANDS, parseReference(ref)), /25% tighter around the middle of the fairway/);
  assert.match(headline('tee', [stat(13.5, 5)], DEFAULT_TEE_BANDS, parseReference(ref)), /Not enough shots/);
});

test('plain labels and chart window', () => {
  assert.equal(bandLabel('tee', DEFAULT_TEE_BANDS, 0), 'Under 260'); assert.equal(bandLabel('tee', DEFAULT_TEE_BANDS, 4), '330+');
  assert.equal(bandLabel('app', DEFAULT_APP_BANDS, 3), '200+'); assert.equal(bandLabel('app', DEFAULT_APP_BANDS, 0), '50–100');
  assert.equal(laterality(-3.2), '3 yd left'); assert.equal(laterality(2.6), '3 yd right'); assert.equal(laterality(0.1), 'on the line');
  const pts = selectPoints(payload, 'tee', 20, DEFAULT_TEE_BANDS);
  const v = viewWindow(pts, 'tee');
  assert.ok(v.xMax >= 40 && v.xMax <= 70 && v.yLo < v.yHi);
  assert.deepEqual(viewWindow([], 'app').yHi > 0, true);
});
