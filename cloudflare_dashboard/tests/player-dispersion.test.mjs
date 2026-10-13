import assert from 'node:assert/strict';
import test from 'node:test';
import {DEFAULT_APP_BANDS, DEFAULT_TEE_BANDS, bandIndex, bandLabel, bandStats, compare, dispersionKey, headline, laterality, parseIndex, parsePayload,
  parseReference, quantile, referenceBand, roundCount, selectPoints, tourSd, viewWindow, windowSpan, fmtDist, sideRange, depthRange, greenWindow, fairwayGuide, fmtFt,
  availableDims, hasEventIds, sliceKeys, sliceLabel, sliceRounds, sliceSpan, bucketCounts, displayName, inBucket, situationReadout, situationStat, SIT_HELP} from '../app/player-dispersion-rules.ts';

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

test('green view: units, ranges, scale and median proximity', () => {
  assert.equal(fmtDist(3), '9 ft'); assert.equal(fmtDist(-12.4), '12 yd'); assert.equal(fmtDist(null), '—'); assert.equal(fmtFt(7.4), '22 ft');
  assert.equal(sideRange(-9, 8), '27 ft left to 24 ft right'); assert.equal(sideRange(-19, 18), '19 yd left to 18 yd right'); assert.equal(sideRange(-0.2, 4), 'the line to 12 ft right');
  assert.equal(depthRange(-12, 6), '12 yd short to 18 ft long');
  const pts = selectPoints(payload, 'app', 20, DEFAULT_APP_BANDS);
  const stats = bandStats(pts, 'app', DEFAULT_APP_BANDS);
  const s0 = stats.find((s) => s.band === 1);
  assert.equal(s0.n, 1); assert.ok(Math.abs(s0.proxMed - Math.hypot(5, -4)) < 1e-9);
  assert.equal(s0.alongQ10 <= s0.alongQ90, true);
  const w = greenWindow(pts);
  assert.ok([15, 20, 30, 40, 60, 90].includes(w.radius) && w.rings.every((r) => r < w.radius));
  assert.equal(greenWindow([]).radius, 20);
});

test('fairway guide is one strip at the median published field extent', () => {
  const r = parseReference({...ref, tee: {'2': {...ref.tee['2'], fairway_lat: {n: 900, q02: -14, q98: 15}}}});
  const g = fairwayGuide(r, DEFAULT_TEE_BANDS, 240, 340);
  assert.deepEqual(g, [{y0: 240, y1: 340, left: -14, right: 15}]);
  assert.deepEqual(fairwayGuide(null, DEFAULT_TEE_BANDS, 240, 340), []);
  assert.deepEqual(fairwayGuide(parseReference(ref), DEFAULT_TEE_BANDS, 240, 340), []);
});

// ---- v2 situations -------------------------------------------------------------------------------------------------------------
const spec = (kind) => ({
  tee: {dims: [
    {key: 'dog', label: 'Dogleg', field: 'd', buckets: [{key: 'L', label: 'Dogleg left', codes: ['L']}, {key: 'S', label: 'Straight', codes: ['S']}, {key: 'R', label: 'Dogleg right', codes: ['R']}]},
    {key: 'wat', label: 'Water / OB', field: 'w', buckets: [{key: '1', label: 'Left only', codes: ['1']}, {key: '0', label: 'None', codes: ['0']}]}],
    stats: {dog: {L: {all: {n: 5000, n_players: 300, lat: {mean: -1, sd: 18, q10: -25, q90: 21, player_sd_p50: 17.5}, fwd: {mean: 290, sd: 20, q10: 260, q90: 320}},
      bands: {'2': {n: 900, n_players: 200, lat: {mean: -2, sd: 18, q10: -26, q90: 20, player_sd_p50: 17}}, '3': {n: 900, n_players: 200, lat: {mean: 0, sd: 18, q10: -24, q90: 22}}}}}, wat: {'1': {all: {n: 900, n_players: 100, lat: {mean: 1.5, sd: 18, q10: -20, q90: 24}}, bands: {}}}}},
  app: {dims: [
    {key: 'pin', label: 'Pin side', field: 'p', buckets: [{key: 'L', label: 'Left', codes: ['0', '3', '6']}, {key: 'C', label: 'Center', codes: ['1', '4', '7']}, {key: 'R', label: 'Right', codes: ['2', '5', '8']}]},
    {key: 'dep', label: 'Pin depth', field: 'p', buckets: [{key: 'F', label: 'Front', codes: ['0', '1', '2']}, {key: 'B', label: 'Back', codes: ['6', '7', '8']}]}],
    stats: {pin: {L: {all: {n: 5000, n_players: 300, lat: {mean: 3.4, sd: 9, q10: -8, q90: 15}, along: {mean: -7, sd: 10, q10: -20, q90: 6}}, bands: {'1': {n: 800, n_players: 100, lat: {mean: 3.0, sd: 8, q10: -7, q90: 13}, along: {mean: -6, sd: 9, q10: -18, q90: 5}}}}},
      dep: {B: {all: {n: 4000, n_players: 300, lat: {mean: 0, sd: 9, q10: -10, q90: 10}, along: {mean: -9, sd: 10, q10: -21, q90: 4}}, bands: {}}}}},
  min_shots: {tee: 60, app: 100}})[kind];
const refV2 = {...ref, schema_version: 'player_dispersion.v2', situations: {tee: spec('tee'), app: spec('app')}};
const payloadV2 = {...payload, schema_version: 'player_dispersion.v2',
  tee: {...payload.tee, d: 'LLRS', w: '1?00', b: '0000', t: '????'}, app: {...payload.app, p: '0348', h: '0?f1'}};

test('v2 payloads parse (v1 still does), codes must match the shot count', () => {
  assert.ok(parsePayload(payloadV2, 7)); assert.ok(parseReference(refV2));
  assert.equal(parsePayload({...payloadV2, tee: {...payloadV2.tee, d: 'LL'}}, 7), null);
  assert.equal(parsePayload({...payloadV2, app: {...payloadV2.app, p: 123}}, 7), null);
  const pts = selectPoints(parsePayload(payloadV2, 7), 'tee', 20, DEFAULT_TEE_BANDS);
  assert.deepEqual(pts[0].c, {d: 'L', w: '1', b: '0', t: '?'});
  assert.equal(selectPoints(payload, 'tee', 20, DEFAULT_TEE_BANDS)[0].c, undefined, 'v1 payloads carry no codes');
  assert.deepEqual(selectPoints(payloadV2, 'app', 20, DEFAULT_APP_BANDS)[1].c, {p: '3', h: '?'});
});

test('dimensions are offered only when the payload has the code and the reference has buckets; unknown shots join no bucket', () => {
  assert.deepEqual(availableDims(refV2, 'tee', payloadV2).map((d) => d.key), ['dog', 'wat']);
  assert.deepEqual(availableDims(refV2, 'tee', payload), [], 'a v1 payload has no situation codes');
  assert.deepEqual(availableDims(ref, 'tee', payloadV2), [], 'a v1 reference has no buckets');
  const dims = availableDims(refV2, 'tee', payloadV2), pts = selectPoints(payloadV2, 'tee', 20, DEFAULT_TEE_BANDS);
  assert.deepEqual(bucketCounts(pts, dims[0]), {L: 2, S: 1, R: 1});
  assert.deepEqual(bucketCounts(pts, dims[1]), {'1': 1, '0': 2}, 'the "?" shot is in no bucket, never in "none"');
  assert.equal(inBucket(pts[3], dims[0], dims[0].buckets[1]), true);
  const a = availableDims(refV2, 'app', payloadV2), ap = selectPoints(payloadV2, 'app', 20, DEFAULT_APP_BANDS);
  assert.deepEqual(bucketCounts(ap, a[0]), {L: 2, C: 1, R: 1}); assert.deepEqual(bucketCounts(ap, a[1]), {F: 1, B: 1});
});

test("bucket statistics match the PGA mean to the golfer's own yardage mix and gate the width", () => {
  const dim = availableDims(refV2, 'tee', payloadV2)[0], bucket = dim.buckets[0];
  const mk = (band, lat, n) => Array.from({length: n}, (_, i) => ({lat: lat + (i % 2 ? 1 : -1), y: 300, dist: 300, along: 0, band, round: 0, lie: 'F', start: 'T', c: {d: 'L'}}));
  const pts = [...mk(2, -6, 30), ...mk(3, -6, 10)];
  const r = situationStat(pts, 'tee', dim, bucket, refV2);
  assert.equal(r.n, 40); assert.ok(Math.abs(r.meanLat + 6) < 1e-9);
  assert.ok(Math.abs(r.pgaLat - (0.75 * -2 + 0.25 * 0)) < 1e-9, 'band cells weighted 30:10');
  assert.equal(r.enough, false); assert.equal(r.showMean, true); assert.equal(r.gate, 60);
  assert.equal(situationStat(pts, 'tee', dim, dim.buckets[2], refV2), null, 'empty bucket');
  const big = situationStat([...mk(2, -6, 40), ...mk(3, -6, 30)], 'tee', dim, bucket, refV2);
  assert.equal(big.enough, true);
  const lone = situationStat(mk(0, -6, 4), 'tee', dim, bucket, refV2); // band 0 has no cell: falls back to the bucket's overall mean
  assert.equal(lone.pgaLat, -1); assert.equal(lone.showMean, false);
  assert.equal(situationStat(pts, 'tee', dim, bucket, null).pgaLat, null);
});

test('readout is one plain sentence, with a separate width clause that waits for enough shots', () => {
  const dim = availableDims(refV2, 'tee', payloadV2)[0], bucket = dim.buckets[0];
  const base = {n: 112, gate: 60, enough: true, showMean: true, meanLat: -6, meanAlong: 0, meanDist: 300, pgaLat: -2, pgaAlong: null, width: 40, pgaWidth: 46, sd: 17, pgaSd: 17};
  const r = situationReadout('tee', 'Rory McIlroy', dim, bucket, base);
  assert.equal(r.text, "On left doglegs, Rory McIlroy finishes 6 yd left of the field's landing line, 4 yd further left than a typical PGA player.");
  assert.equal(r.width, 'Middle 80% is 40 yd wide (PGA 46 yd), from 112 shots.');
  assert.match(situationReadout('tee', 'X', dim, bucket, {...base, enough: false, n: 40}).width, /not enough shots \(40; needs 60\)/);
  assert.match(situationReadout('tee', 'X', dim, bucket, {...base, showMean: false, n: 9}).text, /Only 9 drives/);
  assert.match(situationReadout('tee', 'X', dim, bucket, {...base, meanLat: -2.2}).text, /about the same as a typical PGA player/);
  assert.match(situationReadout('tee', 'X', dim, bucket, {...base, meanLat: 3, pgaLat: -2}).text, /3 yd right of .* 5 yd further right/);
  const ad = availableDims(refV2, 'app', payloadV2), pin = ad[0].buckets[0], dep = ad[1].buckets[1];
  const app = {n: 150, gate: 100, enough: true, showMean: true, meanLat: 3.2, meanAlong: -9, meanDist: 150, pgaLat: 3.4, pgaAlong: -7, width: 20, pgaWidth: 22, sd: 9, pgaSd: 9};
  assert.equal(situationReadout('app', 'Pat', ad[0], pin, app).text, 'With the pin in the left third of the green, Pat finishes 3 yd right of the pin, about the same as a typical PGA player.');
  assert.equal(situationReadout('app', 'Pat', ad[1], dep, app).text, 'With the pin in the back third of the green, Pat finishes 9 yd short of the pin, 2 yd further short than a typical PGA player.');
  assert.equal(situationReadout('app', 'Pat', ad[0], pin, {...app, n: 60, enough: false}).width, 'Width: not enough shots (60; needs 100).');
  for (const k of ['dog', 'wat', 'bun', 'tre', 'pin', 'dep', 'wtr']) assert.ok(SIT_HELP[k].length > 80, `${k} has a plain-words explanation of how it is detected`);
});

test('names are shown first-name first', () => {
  assert.equal(displayName('McIlroy, Rory'), 'Rory McIlroy'); assert.equal(displayName('Tiger Woods'), 'Tiger Woods'); assert.equal(displayName(''), 'this golfer'); assert.equal(displayName(undefined), 'this golfer');
});

test('slice filters: shots join history rounds by event id + round, replacing the last-N cut', () => {
  const p = {...payload, rounds: [['2026-09-01', 4, 'E', 'pga:2026:5'], ['2026-08-31', 3, 'E', 'pga:2026:5'], ['2026-08-30', 2, 'F', 'pga:2026:6']]};
  assert.equal(hasEventIds(p), true); assert.equal(hasEventIds(payload), false, 'older publishes carry no event ids');
  const events = [{id: 'pga:2026:5', round_data: [{round: 4}, {round_num: 3}]}, {id: 'euro:2026:9', round_data: [{round: 1}]}];
  const keys = sliceKeys(events);
  assert.deepEqual([...keys].sort(), ['euro:2026:9|1', 'pga:2026:5|3', 'pga:2026:5|4']);
  const allowed = sliceRounds(p, keys);
  assert.deepEqual([...allowed].sort(), [0, 1]);
  assert.equal(selectPoints(p, 'tee', 1, DEFAULT_TEE_BANDS, 'all', allowed).length, 3, 'rounds 0 and 1, not limited by last-N');
  assert.equal(selectPoints(p, 'tee', 20, DEFAULT_TEE_BANDS, 'all', new Set()).length, 0);
  assert.deepEqual(sliceSpan(p, 'tee', allowed), {n: 2, first: '2026-08-31', last: '2026-09-01'});
  assert.equal(sliceSpan(p, 'tee', new Set([2]))?.n, 1); assert.equal(sliceSpan(p, 'tee', new Set([5])), null);
  assert.equal(sliceRounds(payload, keys).size, 0, 'no ids -> empty slice');
});

test('slice caption states the filters in plain words', () => {
  assert.equal(sliceLabel({window: 'recent2y', rounds: [1, 2, 3, 4]}), 'last 2 years');
  assert.equal(sliceLabel({window: 'all', season: '2024', tour: 'pga', major: true, rounds: [2, 1]}), 'all history · 2024 season · PGA · majors only · rounds 1, 2');
  assert.match(sliceLabel({window: 'all', event: 'pga:2026:5', contention: 'near'}, 'The Masters'), /The Masters · within 2 shots of the lead/);
});
