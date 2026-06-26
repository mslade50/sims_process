"""kalshi_match.py — shared, robust Kalshi golf event matching + name normalization.

Single source of truth for (a) which Kalshi EVENT our sim is pricing and (b)
normalizing Kalshi player names to the sim's 'last, first' keys. Replaces three
drifting matchers (kalshi_ancillary._match_tournament slug matcher,
round_sim._detect_tourney/_matches_target_rt, kalshi_maker._detect_target_event_code).

The robust approach: SIM-PLAYER OVERLAP — the Kalshi event whose markets contain the
most of OUR field is, by definition, the one we're pricing. This is immune to the
slug/stopword fragility, empty-distinct major slugs (pga_championship ->no distinct
word), and the abbreviated 3-ball event tags (RBCO26 vs RBBCAN26) that break the
title/slug matchers. An optional per-slug event_tag override (sim_inputs.kalshi_event_tags)
acts as a tie-breaker, never as the sole signal.
"""
import re

# Surname particles that belong WITH the surname:
# 'Erik van Rooyen' -> 'van rooyen, erik';  'Adrien Dumont de Chassart' -> 'dumont de chassart, adrien'
_PARTICLES = {"van", "von", "de", "del", "della", "da", "das", "dos", "du",
              "la", "le", "den", "der", "ter", "ten", "di", "vander", "van der"}


def ticker_event_code(ticker):
    """'KXPGATOUR-RBBCAN26-BKOH' -> 'RBBCAN26'.  '' if the shape doesn't match."""
    if not isinstance(ticker, str):
        return ""
    parts = ticker.split("-", 2)
    return parts[1] if len(parts) >= 2 else ""


def _split_first_last(x):
    """'erik van rooyen' -> ('van rooyen', 'erik'); particle-aware."""
    toks = x.split()
    if len(toks) < 2:
        return None
    i = len(toks) - 1
    # pull any leading particle tokens into the surname
    while i - 1 >= 1 and toks[i - 1] in _PARTICLES:
        i -= 1
    surname = " ".join(toks[i:])
    first = " ".join(toks[:i])
    return surname, first


def norm_name(s, name_replacements=None, field_set=None):
    """Normalize a Kalshi display name to the sim 'last, first' lowercase key.

    Particle-aware (van/von/de/dumont de...), dot/hyphen tolerant, with an optional
    `field_set` fallback that matches on (surname-token, first-initial) before giving
    up — so 'Min Woo Lee' / 'J.J. Spaun' style names still resolve to the field.
    Returns '' for empty input.
    """
    name_replacements = name_replacements or {}
    x = (s or "").strip().lower()
    if not x:
        return ""

    if "," in x:
        key = x
    else:
        sl = _split_first_last(x)
        key = f"{sl[0]}, {sl[1]}" if sl else x

    if key in name_replacements:
        return name_replacements[key]
    if field_set and key in field_set:
        return key

    # Fallback: match the field by (surname tokens, first initial), tolerant of dots/hyphens.
    if field_set:
        def canon(n):
            return re.sub(r"[.\-]", " ", n).strip()
        ksur = canon(key.split(",")[0])
        kfirst = canon(key.split(",")[-1])
        kfi = kfirst[:1]
        matches = [c for c in field_set
                   if canon(c.split(",")[0]) == ksur and canon(c.split(",")[-1])[:1] == kfi]
        if len(matches) == 1:
            return matches[0]

    return name_replacements.get(key, key)


def _player_from_title(title):
    """Outright/maker titles: 'RBC Canadian Open: Will X finish...' / 'Will X win the ...'."""
    t = title or ""
    m = re.match(r".*?:\s*Will (.+?) (?:finish|make|miss|lead|win)", t)
    if m:
        return m.group(1).strip()
    m = re.match(r"Will (.+?) (?:win|finish|lead|make|miss|beat) ", t)
    if m:
        return m.group(1).strip()
    return ""


def player_from_market(m, kind="subtitle"):
    """Raw player name from a Kalshi market.
    kind='subtitle' -> yes_sub_title (ancillary leader/top-N/3-ball), title fallback.
    kind='title'    -> parse the title (outright/maker)."""
    if kind == "subtitle":
        st = (m.get("yes_sub_title") or "").strip()
        if st:
            if " beats " in st:           # 3-ball: '<X> beats <Y> and <Z>' -> X
                return st.split(" beats ")[0].strip()
            return st
        return _player_from_title(m.get("title", ""))   # fall back to title field
    return _player_from_title(m.get("title", ""))


def three_ball_members(m):
    """All 3 (or N) players in a 3-ball market's yes_sub_title '<X> beats <Y> and <Z>'.
    Returns [X, Y, Z] raw names (best-effort; empty list if unparseable)."""
    st = (m.get("yes_sub_title") or "").strip()
    if " beats " not in st:
        return []
    head, _, tail = st.partition(" beats ")
    others = re.split(r",\s*|\s+and\s+", tail.strip())
    return [head.strip()] + [o.strip() for o in others if o.strip()]


def event_overlap_counts(markets, sim_player_set, name_replacements=None, kind="subtitle"):
    """How many markets per event_code have a player in sim_player_set."""
    counts = {}
    for m in markets:
        ec = ticker_event_code(m.get("ticker", ""))
        if not ec:
            continue
        raw = player_from_market(m, kind=kind)
        if not raw:
            continue
        if norm_name(raw, name_replacements, sim_player_set) in sim_player_set:
            counts[ec] = counts.get(ec, 0) + 1
    return counts


def match_markets(markets, sim_player_set, name_replacements=None, kind="subtitle",
                  event_tag=None, min_overlap=3):
    """Filter `markets` to the single Kalshi event our sim is pricing, by sim-player
    overlap. Returns (filtered, report).

    report = {raw, matched, target, leader, counts, reason}
      - raw       : markets fetched for this series
      - matched   : markets kept (belong to the target event)
      - target    : resolved event_code (or None)
      - reason    : non-empty when nothing resolved / low confidence (for alerting)
    `event_tag` (a sim_inputs override like 'RBBCAN26') is a TIE-BREAKER preferred when
    present, never the sole signal.
    """
    raw = len(markets)
    counts = event_overlap_counts(markets, sim_player_set, name_replacements, kind)
    report = {"raw": raw, "matched": 0, "target": None, "leader": 0,
              "counts": counts, "reason": ""}
    if not counts:
        report["reason"] = "no sim-player overlap on any event"
        return [], report

    target = None
    if event_tag:
        for ec in counts:
            if event_tag.upper() in ec.upper():
                target = ec
                break
    if target is None:
        target = max(counts, key=counts.get)

    leader = counts[target]
    report["leader"] = leader
    if leader < min_overlap:
        report["reason"] = f"top event {target} has only {leader} sim-player matches (< {min_overlap})"

    filtered = [m for m in markets if ticker_event_code(m.get("ticker", "")) == target]
    report["matched"] = len(filtered)
    report["target"] = target
    return filtered, report
