"""Pure parsing and consistency checks for PGA tournament-winner contracts."""
import re


def _key(value):
    return " ".join(value.casefold().split())


def _event_key(value):
    return _key(re.sub(r"^\d{4}\s+", "", value))


def _tournament_winner(player, event):
    # Round/group/hole outcomes cannot use a tournament winner probability.
    return not re.search(
        r"\b(?:round|hole|matchup|playoff|score|margin|birdies|eagles)\b|\b\d+-ball\b",
        f"{player} {event}", re.I,
    )


def winner_title_parts(title):
    """Return (player, tournament) for explicit old/new winner titles only."""
    value = str(title or "").strip()
    for pattern, player_group, event_group in (
        (r"([^:]+):\s*([^:]+?)\s+wins[?.]?", 2, 1),
        (r"Will\s+(.+?)\s+win\s+the\s+([^?]+)\?", 1, 2),
        (r"([^:]+):\s*Will\s+(.+?)\s+win\?", 2, 1),
    ):
        match = re.fullmatch(pattern, value, re.I)
        if match:
            player = match.group(player_group).strip()
            event = match.group(event_group).strip()
            if player and event and _tournament_winner(player, event):
                return player, event
    return "", ""


def winner_player(market):
    """Resolve a winner player only when supplied identity metadata agrees.

    A blank title can use yes_sub_title only with explicit winner rules proving
    the same player. An unfamiliar nonempty title is never rescued by a name.
    """
    ticker = str(market.get("ticker") or "")
    parts = ticker.split("-")
    if len(parts) != 3 or parts[0] not in {"KXPGATOUR", "KXPGAWIN"} or not all(parts):
        return ""
    if re.search(r"R[1-4]$", parts[1]):
        return ""
    if market.get("event_ticker") and market["event_ticker"] != "-".join(parts[:2]):
        return ""
    if market.get("market_type") not in (None, "", "binary"):
        return ""
    if market.get("primary_participant_key") not in (None, "", "golf_competitor"):
        return ""
    if market.get("_market_type") not in (None, "", "winner"):
        return ""

    title = str(market.get("title") or "").strip()
    player, event = winner_title_parts(title)
    if title and not player:
        return ""
    rules = str(market.get("rules_primary") or "").strip()
    rule_match = re.fullmatch(
        r"If (.+?) wins the (.+?), then the market resolves to Yes\.?", rules, re.I,
    )
    if rules:
        if not rule_match:
            return ""
        rule_player, rule_event = (s.strip() for s in rule_match.groups())
        if not _tournament_winner(rule_player, rule_event):
            return ""
        if player and (_key(player) != _key(rule_player) or
                       _event_key(event) != _event_key(rule_event)):
            return ""
        player = player or rule_player

    subtitle = str(market.get("yes_sub_title") or "").strip()
    if subtitle and _key(subtitle) not in {"yes", "no"}:
        if _key(subtitle) != _key(player):
            return ""
    elif not title:
        return ""
    return player


def winner_tournament(market):
    """Tournament proven by a consistent winner title or explicit winner rules."""
    if not winner_player(market):
        return ""
    _, event = winner_title_parts(market.get("title"))
    if event:
        return event
    match = re.fullmatch(
        r"If .+? wins the (.+?), then the market resolves to Yes\.?",
        str(market.get("rules_primary") or "").strip(), re.I,
    )
    return re.sub(r"^\d{4}\s+", "", match.group(1)).strip() if match else ""
