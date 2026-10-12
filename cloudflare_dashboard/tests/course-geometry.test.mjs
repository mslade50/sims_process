import assert from 'node:assert/strict';
import test from 'node:test';
import {parseGeoIndex, venueForEvent, venuesForEvent, fitKeyFor, pickVenue, courseLabel, playedOverText, parseCard, parseFit, parseReference, parseTraits, cardView, cardRows, ordinalPercentile, visibleBadges, hasNoisy, strokesText, badgeMeta,
  fitLists, displayName, reliableTraits, traitValueText, playerCoverage, MAIN_KEYS} from '../app/course-geometry-rules.ts';

const P = 'golfprice/course_geometry/geometry-abc/';
const index = {schema_version: 'course_geometry.index.v1', generation: 'abc', prefix: P, reference_key: `${P}reference.json`, traits_key: `${P}traits.json`, as_of: '2026-10-11', count: 4,
  descriptive: 'Descriptive, not priced: nothing here moves a price or a bet.',
  venues: {
    V_port: {key: `${P}venues/V_port.json`, name: 'Butterfield Bermuda Championship', course_name: 'Port Royal', event_ids: ['pga:528'], event_uids: ['pga:528:2025-11-13'], coverage: 'yardage_only', editions: 6, coord_editions: 0},
    V_saw: {key: `${P}venues/V_saw.json`, name: 'THE PLAYERS Championship', course_name: 'TPC Sawgrass', event_ids: ['pga:11'], event_uids: ['pga:11:2025-03-13'], coverage: 'full', editions: 6, coord_editions: 6},
    V_a: {key: `${P}venues/V_a.json`, name: 'PGA Championship', event_ids: ['pga:7'], event_uids: ['pga:7:2024-05-16', 'pga:7:2025-05-15'], coverage: 'full', editions: 3, coord_editions: 3},
    V_b: {key: `${P}venues/V_b.json`, name: 'PGA Championship', event_ids: ['pga:7'], event_uids: ['pga:7:2023-05-18'], coverage: 'full', editions: 3, coord_editions: 3},
    V_none: {key: `${P}venues/V_none.json`, name: null, coverage: 'full', editions: 1, coord_editions: 1},
  },
  by_event: {'pga:528': ['V_port'], 'pga:528:2025-11-13': ['V_port'], 'pga:11': ['V_saw'], 'pga:7': ['V_a', 'V_b'], 'pga:7:2024-05-16': ['V_a']},
  current: {event_uid: 'pga:528:2026-10-22', event_id: 'pga:528', name: 'Butterfield Bermuda Championship', venue_uid: 'V_port', course_name: 'Port Royal', venue_source: 'latest_edition', ambiguous: false, fit_key: `${P}fit/V_port.json`},
  fits: {V_saw: `${P}fit/V_saw.json`, V_port: `${P}fit/V_port.json`}};
const f = (key, pct, extra = {}) => ({key, label: key, unit: 'x', display: 1, pct, z: 0, n_editions: 6, evidence: 'B', flag: null, text: `${key} sentence`, ...extra});
const card = {schema_version: 'course_geometry.v1', venue_uid: 'V_saw', event_name: 'THE PLAYERS Championship', as_of: 'x', editions: 6, coord_editions: 6, seasons: [2025], coverage: 'full', par: 72, yardage: 7200,
  features: [f('tee_bk', 54), f('green_area', 66, {flag: 'proxy', n_editions: 0}), f('rough_sev', 91, {flag: 'weak'}), f('par4_len', 22, {evidence: 'A'}), f('dog_L', 63), f('dog_R', 34), f('dog_net', 26), f('tee_wo', 90), f('app_wo', 90), f('long_app', 47), f('fw_hit', 40)]};
const cardPort = {schema_version: 'course_geometry.v1', venue_uid: 'V_port', event_name: 'Butterfield Bermuda Championship', as_of: 'x', editions: 6, coord_editions: 0, seasons: [2025], coverage: 'yardage_only', par: 71, yardage: 6828,
  note: 'No shot coordinates were published for this event, so only yardage is shown.', features: [f('par4_len', 2, {evidence: 'A'}), f('tee_wo', 50)]};
const badge = (o) => ({pair: 'dist_x_len', grade: 'A', kind: 'misfit', label: 'Long hitter on a short course', score: -1.4, trait_z: 2, trait_n: 2191, venue_z: -0.7, hidden: false, est_strokes_round: -0.068, ...o});
const fit = {schema_version: 'course_fit.v1', venue_uid: 'V_saw', event_name: 'THE PLAYERS Championship', as_of: 'x', coverage: 'full', descriptive: 'Descriptive, not priced.', player_source: 'top_by_skill',
  pairs: {dist_x_len: {label: 'Distance x par-4 length', grade: 'A', why: 'Survived walk-forward.', hidden_default: false}, width_x_narrow: {label: 'Drive spread x width', grade: 'C', why: 'Noisy.', hidden_default: true}},
  players: [
    {dg_id: 1, name: 'McIlroy, Rory', traits: {dist: {z: 2, n: 2191, v: 13.09}, aplong: {z: null, n: 60, v: null}}, badges: [badge({})], hidden_badges: []},
    {dg_id: 2, name: 'Fit, Frank', traits: {}, badges: [badge({kind: 'fit', score: 3, est_strokes_round: 0.1, label: 'Short hitter on a short course'})], hidden_badges: [badge({pair: 'width_x_narrow', grade: 'C', score: 9, est_strokes_round: null, hidden: true, kind: 'fit', label: 'Noisy fit'})]},
    {dg_id: 3, name: null, traits: {}, badges: [], hidden_badges: []},
    {dg_id: 4, name: 'Bee, Barry', traits: {}, badges: [badge({pair: 'nat_x_lean', grade: 'B', est_strokes_round: 0.5, score: -0.9})], hidden_badges: []},
  ]};
const reference = {schema_version: 'course_fit.v1', grades: {A: 'Walk-forward', B: 'Stable', C: 'Noisy'}, traits: [{key: 'dist', label: 'Driving distance', unit: 'yd vs field', min_n: 300}, {key: 'aplong', label: 'Long-iron SG', unit: 'strokes per 200+ yd approach', min_n: 100}], pairs: []};

test('pointer parsing is strict about schema and key prefixes; bad keys are dropped, not trusted', () => {
  assert.ok(parseGeoIndex(index));
  assert.equal(parseGeoIndex({...index, schema_version: 'x'}), null);
  assert.equal(parseGeoIndex({...index, prefix: 'golfprice/other/'}), null);
  assert.equal(parseGeoIndex({...index, prefix: `${P}../x/`}), null);
  assert.equal(parseGeoIndex({...index, reference_key: 'elsewhere/reference.json'}), null);
  assert.equal(parseGeoIndex(null), null); assert.equal(parseGeoIndex('x'), null); assert.equal(parseGeoIndex({}), null);
  const bad = parseGeoIndex({...index, venues: {...index.venues, V_evil: {key: 'golfprice/other/x.json', name: 'E', coverage: 'full'}}, fits: {V_saw: 'https://evil/x.json'}});
  assert.equal(bad.venues.V_evil, undefined); assert.equal(bad.fits.V_saw, undefined);
  assert.equal(parseGeoIndex({...index, traits_key: 'x'}).traits_key, null);
});

test('venue is matched by stable ids only: current, then listed event_uid, then a one-venue series; names never match', () => {
  const idx = parseGeoIndex(index);
  assert.equal(venueForEvent(idx, 'pga:528:2026-10-22'), 'V_port', 'current event');
  assert.equal(venueForEvent(idx, 'pga:528:2025-11-13'), 'V_port', 'known edition');
  assert.equal(venueForEvent(idx, 'pga:528:2027-10-21'), 'V_port', 'new date of a one-venue series');
  assert.equal(venueForEvent(idx, 'pga:11:2026-03-12'), 'V_saw');
  assert.equal(venueForEvent(idx, 'pga:7:2024-05-16'), 'V_a', 'rotating series: exact edition still resolves');
  assert.equal(venueForEvent(idx, 'pga:7:2026-05-14'), null, 'ambiguous series, unknown edition: hide');
  assert.equal(venueForEvent(idx, 'pga:999:2026-01-01'), null, 'unknown event');
  assert.equal(venueForEvent(idx, 'Butterfield Bermuda Championship'), null, 'names are display-only');
  assert.equal(venueForEvent(idx, null), null); assert.equal(venueForEvent(null, 'pga:528:2026-10-22'), null); assert.equal(venueForEvent(idx, ''), null);
});

test('current wins for the event it names (rotating course) and no_venue_card shows nothing', () => {
  const rot = parseGeoIndex({...index, current: {event_uid: 'pga:7:2026-05-14', event_id: 'pga:7', venue_uid: 'V_b', course_name: 'B', ambiguous: true}});
  assert.equal(venueForEvent(rot, 'pga:7:2026-05-14'), 'V_b');
  assert.equal(venueForEvent(rot, 'pga:7:2026-05-21'), null);
  const none = parseGeoIndex({...index, current: {status: 'no_venue_card', event_uid: 'pga:528:2026-10-22', event_id: 'pga:528'}});
  assert.equal(venueForEvent(none, 'pga:528:2026-10-22'), null);
  assert.equal(venueForEvent(none, 'pga:528:2025-11-13'), 'V_port', 'other events unaffected');
  const missing = parseGeoIndex({...index, current: {event_uid: 'pga:1:2026-01-01', venue_uid: 'V_gone'}});
  assert.equal(venueForEvent(missing, 'pga:1:2026-01-01'), null, 'current naming an unlisted venue');
  assert.equal(parseGeoIndex({...index, current: undefined, by_event: undefined}).current, null);
});

test('card parsing checks identity and drops malformed features', () => {
  assert.ok(parseCard(card, 'V_saw'));
  assert.equal(parseCard(card, 'V_port'), null);
  assert.equal(parseCard({...card, schema_version: 'v0'}, 'V_saw'), null);
  assert.equal(parseCard({...card, features: []}, 'V_saw'), null);
  assert.equal(parseCard({...card, features: [f('par4_len', 120), {key: 'x'}]}, 'V_saw'), null, 'pct out of range or missing fields dropped');
  assert.equal(parseCard({...card, coverage: 'maybe'}, 'V_saw'), null);
  assert.equal(parseCard(null, 'V_saw'), null);
});

test('full card: bars in the published order, weak and proxy muted, tee_bk only behind more', () => {
  const v = cardView(parseCard(card, 'V_saw'));
  assert.equal(v.fallback, false);
  assert.deepEqual(v.main.map((r) => r.key), ['par4_len', 'dog_L', 'dog_R', 'tee_wo', 'app_wo', 'long_app', 'rough_sev', 'green_area']);
  assert.deepEqual(v.more.map((r) => r.key), ['dog_net', 'fw_hit', 'tee_bk']);
  const rough = v.main.find((r) => r.key === 'rough_sev'), green = v.main.find((r) => r.key === 'green_area');
  assert.deepEqual([rough.flag, rough.muted, green.flag, green.muted, v.main[0].muted], ['weak', true, 'proxy', true, false]);
  assert.equal(v.main[0].text, 'par4_len sentence', 'publisher sentence passes through unchanged');
  assert.equal(v.main[0].pct, 22);
  assert.equal(v.summary, '6 editions, 6 with shot coordinates');
  assert.ok(!cardRows(parseCard(card, 'V_saw'), MAIN_KEYS).some((r) => r.key === 'tee_bk'));
});

test('yardage-only card shows the note and the par-4 length bar only', () => {
  const c = parseCard(cardPort, 'V_port');
  const v = cardView(c);
  assert.equal(v.fallback, true);
  assert.deepEqual(v.main.map((r) => r.key), ['par4_len']);
  assert.deepEqual(v.more, []);
  assert.equal(c.note, 'No shot coordinates were published for this event, so only yardage is shown.');
  assert.equal(v.summary, '6 editions, no shot coordinates published');
});

test('ordinal percentile wording', () => {
  assert.equal(ordinalPercentile(2), '2nd percentile of venues');
  assert.equal(ordinalPercentile(11), '11th percentile of venues');
  assert.equal(ordinalPercentile(22), '22nd percentile of venues');
  assert.equal(ordinalPercentile(53), '53rd percentile of venues');
  assert.equal(ordinalPercentile(91), '91st percentile of venues');
});

test('badges: C and hidden items stay behind the toggle; strokes per round only for grade A', () => {
  const fp = parseFit(fit, 'V_saw');
  assert.ok(fp); assert.equal(parseFit(fit, 'V_port'), null); assert.equal(parseFit({...fit, schema_version: 'x'}, 'V_saw'), null);
  const frank = fp.players[1];
  assert.deepEqual(visibleBadges(frank, false).map((b) => b.grade), ['A']);
  assert.deepEqual(visibleBadges(frank, true).map((b) => b.grade), ['C', 'A'], 'strongest first when shown');
  assert.equal(hasNoisy(fp), true);
  const leaked = {...frank, badges: [...frank.badges, badge({pair: 'width_x_narrow', grade: 'C', est_strokes_round: 0.3})]};
  assert.equal(visibleBadges(leaked, false).some((b) => b.grade === 'C'), false, 'a C badge in the visible list is still hidden by default');
  assert.equal(strokesText(frank.badges[0]), '+0.10 strokes/round');
  assert.equal(strokesText(fp.players[0].badges[0]), '−0.07 strokes/round');
  assert.equal(strokesText(fp.players[3].badges[0]), null, 'grade B carries no strokes figure even if the payload has one');
  assert.equal(strokesText(visibleBadges(frank, true)[0]), null, 'grade C: none');
  assert.equal(badgeMeta(fp.players[0].badges[0]), 'Grade A · n 2,191');
  assert.equal(badgeMeta({...badge({}), trait_n: null}), 'Grade A · sample size unknown');
});

test('compact fit lists: each golfer once under his strongest badge, sorted, grade C only on request', () => {
  const fp = parseFit(fit, 'V_saw');
  const l = fitLists(fp, false);
  assert.deepEqual(l.fits.map((e) => e.name), ['Frank Fit']);
  assert.deepEqual(l.misfits.map((e) => e.name), ['Rory McIlroy', 'Barry Bee']);
  assert.equal(l.nFits, 1); assert.equal(l.nMisfits, 2);
  const n = fitLists(fp, true);
  assert.equal(n.fits[0].badge.grade, 'C'); assert.equal(n.fits[0].others, 1);
  assert.equal(fitLists(fp, false, 1).misfits.length, 1);
  assert.equal(fitLists(fp, false, 1).nMisfits, 2);
  assert.match(playerCoverage(fp), /^Top 40 by skill/);
  assert.match(playerCoverage({...fp, player_source: 'field'}), /^Entered field/);
});

test('names and reliable traits: gated-out traits are withheld, never shown as numbers', () => {
  assert.equal(displayName('McIlroy, Rory'), 'Rory McIlroy'); assert.equal(displayName(null), 'Unknown golfer'); assert.equal(displayName('Madonna'), 'Madonna');
  const ref = parseReference(reference);
  assert.ok(ref); assert.equal(parseReference({}), null);
  const t = reliableTraits(parseFit(fit, 'V_saw').players[0].traits, ref);
  assert.equal(t.rows.length, 1); assert.equal(t.withheld, 1);
  assert.equal(traitValueText(t.rows[0]), '+13.1 yd vs field');
  assert.equal(traitValueText({value: -0.0211, unit: 'strokes'}), '−0.021 strokes');
  assert.deepEqual(reliableTraits({}, ref), {rows: [], withheld: 0});
  assert.deepEqual(reliableTraits({dist: {z: 1, n: 5, v: 1}}, null), {rows: [], withheld: 0});
  const doc = parseTraits({schema_version: 'course_fit.v1', as_of: 'x', players: {'1': {name: 'A, B', t: {dist: {z: 1, n: 400, v: 2}}}}});
  assert.equal(doc.players['1'].t.dist.n, 400); assert.equal(parseTraits({schema_version: 'x'}), null);
});

test('absent or malformed payloads yield null everywhere (card stays hidden, nothing throws)', () => {
  for (const bad of [undefined, null, 5, 'x', [], {}]) {
    assert.equal(parseGeoIndex(bad), null); assert.equal(parseCard(bad, 'V_saw'), null); assert.equal(parseFit(bad, 'V_saw'), null);
    assert.equal(parseReference(bad), null); assert.equal(parseTraits(bad), null);
  }
});

test('multi-course event: unnamed course lists every course with a fit list each; named course gives one', () => {
  const multiRaw = {...index, venues: {...index.venues, V_c: {key: `${P}venues/V_c.json`, name: 'AmEx', course_name: 'Pete Dye', event_ids: ['pga:2'], event_uids: ['pga:2:2025-01-16'], coverage: 'full', editions: 3, coord_editions: 3},
    V_d: {key: `${P}venues/V_d.json`, name: 'AmEx', course_name: 'Nicklaus', event_ids: ['pga:2'], event_uids: ['pga:2:2025-01-16'], coverage: 'full', editions: 3, coord_editions: 3}},
    current: {event_uid: 'pga:2:2026-01-15', event_id: 'pga:2', venue_uid: null, venue_uids: ['V_c', 'V_d', 'V_gone'], ambiguous: true, venue_source: 'edition',
      fit_keys: {V_c: `${P}fit/V_c.json`, V_d: `${P}fit/V_d.json`, V_x: 'https://evil/x.json'}}};
  const multi = parseGeoIndex(multiRaw);
  assert.deepEqual(venuesForEvent(multi, 'pga:2:2026-01-15'), ['V_c', 'V_d'], 'unknown uids are dropped');
  assert.equal(venueForEvent(multi, 'pga:2:2026-01-15'), null, 'single-course lookup stays null for several courses');
  assert.equal(fitKeyFor(multi, 'V_c'), `${P}fit/V_c.json`);
  assert.equal(fitKeyFor(multi, 'V_x'), null, 'fit keys outside the prefix are dropped');
  assert.equal(fitKeyFor(multi, 'V_saw'), `${P}fit/V_saw.json`, 'pointer fits still win');
  assert.equal(fitKeyFor(multi, 'V_d'), `${P}fit/V_d.json`); assert.equal(fitKeyFor(multi, null), null); assert.equal(fitKeyFor(null, 'V_c'), null);
  assert.equal(courseLabel(multi, 'V_d'), 'Nicklaus'); assert.equal(courseLabel(multi, 'V_none'), 'V_none');
  assert.equal(playedOverText(3), 'Played over 3 courses');
  assert.equal(pickVenue(['V_c', 'V_d'], 'V_d'), 'V_d'); assert.equal(pickVenue(['V_c', 'V_d'], 'V_zzz'), 'V_c'); assert.equal(pickVenue([], 'V_c'), null);
  const named = parseGeoIndex({...multiRaw, current: {...multiRaw.current, event_uid: 'pga:2:2026-01-15', venue_uid: 'V_d', venue_uids: ['V_c', 'V_d'], fit_keys: {V_d: `${P}fit/V_d.json`}}});
  assert.deepEqual(venuesForEvent(named, 'pga:2:2026-01-15'), ['V_d']);
  const unknown = parseGeoIndex({...index, current: {event_uid: 'pga:9:2026-01-01', venue_uid: null, venue_uids: [], ambiguous: false}});
  assert.deepEqual(venuesForEvent(unknown, 'pga:9:2026-01-01'), [], 'genuinely unknown event still hides');
  assert.deepEqual(venuesForEvent(multi, 'pga:7:2026-05-14'), [], 'ambiguous rotating series, no current: hide');
  assert.deepEqual(venuesForEvent(multi, 'pga:11:2026-03-12'), ['V_saw']);
  assert.equal(parseGeoIndex({...index, current: {event_uid: 'a', venue_uids: 'x', fit_keys: 5}}).current.venue_uids.length, 0, 'malformed lists tolerated');
});
