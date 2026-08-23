import os
from types import SimpleNamespace
from unittest.mock import patch

import publish_sim_fairs as psf


def test_board_dispatch_retries_transient_failure():
    responses = [
        TimeoutError("temporary timeout"),
        SimpleNamespace(status_code=204, text=""),
    ]

    def post(*_args, **_kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    with (
        patch.dict(
            os.environ,
            {"GH_TOKEN": "test-token", "BOARD_SUPPRESS_SIM_CASCADE": ""},
        ),
        patch("requests.post", side_effect=post) as post_mock,
        patch("time.sleep"),
    ):
        assert psf._dispatch_board_build("abc123") is True

    assert post_mock.call_count == 2
    assert post_mock.call_args.kwargs["json"] == {
        "event_type": "sim-fairs-published",
        "client_payload": {"sha": "abc123"},
    }


def test_board_dispatch_can_suppress_sim_cascade_for_pinned_publish():
    response = SimpleNamespace(status_code=204, text="")

    with (
        patch.dict(
            os.environ,
            {"GH_TOKEN": "test-token", "BOARD_SUPPRESS_SIM_CASCADE": "1"},
        ),
        patch("requests.post", return_value=response) as post_mock,
    ):
        assert psf._dispatch_board_build("fresh-fairs-sha") is True

    assert post_mock.call_args.kwargs["json"] == {
        "event_type": "sim-fairs-published",
        "client_payload": {
            "sha": "fresh-fairs-sha",
            "suppress_sim_cascade": True,
        },
    }


def test_midweek_workflow_requires_fairs_publish_and_dispatch():
    workflow = (
        psf.PROJECT_ROOT / ".github" / "workflows" / "midweek-round-automation.yml"
    ).read_text(encoding="utf-8")

    assert "REQUIRE_SIM_FAIRS_PUBLISH: '1'" in workflow


def test_manual_run_workflow_requires_fairs_publish_and_dispatch():
    workflow = (
        psf.PROJECT_ROOT / ".github" / "workflows" / "run-sim.yml"
    ).read_text(encoding="utf-8")

    assert "REQUIRE_SIM_FAIRS_PUBLISH: '1'" in workflow
