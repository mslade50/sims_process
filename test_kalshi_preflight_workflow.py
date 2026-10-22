"""The preflight workflow alerts on config lag and still exits 0."""
from pathlib import Path

WORKFLOW = Path(".github/workflows/kalshi-preflight.yml")
SAMPLE_ALERT = (
    "NOT READY YET: sim_inputs lags the confirmed live event. "
    "Advance sim_inputs.event_ids."
)


def _step(text, name):
    start = text.index(f"- name: {name}")
    rest = text[start + 1:]
    end = rest.find("\n      - name:")
    return rest[:end] if end >= 0 else rest


def test_lag_alert_uses_telegram_policy_and_stays_green():
    text = WORKFLOW.read_text(encoding="utf-8")
    preflight = _step(text, "Run Kalshi preflight")
    alert = _step(text, "Telegram when config lags")

    assert "python kalshi_preflight.py" in preflight
    assert 'tee "$RUNNER_TEMP/kalshi-preflight.txt"' in preflight
    assert "exit 1" not in preflight

    assert "if: success()" in alert
    assert 'grep -F -q "NOT READY YET"' in alert
    assert "telegram_policy.py" in alert
    assert "--key legacy-kalshi-preflight-lag" in alert
    assert SAMPLE_ALERT in alert
    assert "exit 1" not in alert
    assert "|| true" not in alert
