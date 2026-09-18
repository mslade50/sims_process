"""Explicit event-scoped exclusions for live simulation inputs only."""

from datetime import datetime


def filter_live_sim_players(frame):
    """Keep raw API data intact; omit only owner-configured simulation players."""
    if frame is None or frame.empty:
        return frame
    import sim_inputs

    configured = getattr(sim_inputs, "live_sim_exclusions", {})
    replacements = sim_inputs.name_replacements

    def canonical(value):
        name = str(value).strip().lower()
        return replacements.get(name, name)

    excluded = {
        canonical(name)
        for event_id in sim_inputs.event_ids
        for name in configured.get(f"{datetime.now().year}:{event_id}", [])
    }
    if not excluded:
        return frame
    names = frame["player_name"].map(canonical)
    removed = sorted(set(names) & excluded)
    if removed:
        print("  [live sim] Owner-excluded player(s): " + ", ".join(removed))
    return frame.loc[~names.isin(excluded)].copy()
