import unittest
from unittest.mock import patch

import sheet_config


class _FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeSheetError(Exception):
    def __init__(self, code, headers=None):
        super().__init__(f"HTTP {code}")
        self.code = code
        self.response = _FakeResponse(code, headers)


class SheetRequestRetryTests(unittest.TestCase):
    def test_retries_quota_error_until_request_succeeds(self):
        outcomes = [_FakeSheetError(429), _FakeSheetError(429), "ok"]

        def operation():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("sheet_config.time.sleep") as sleep:
            result = sheet_config._retry_sheet_request(
                operation,
                "test read",
                delays=(5, 10),
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [5.0, 10.0],
        )

    def test_honors_retry_after_header(self):
        outcomes = [_FakeSheetError(429, {"Retry-After": "60"}), "ok"]

        def operation():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("sheet_config.time.sleep") as sleep:
            result = sheet_config._retry_sheet_request(
                operation,
                "test read",
                delays=(5,),
            )

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(60.0)

    def test_does_not_retry_permanent_error(self):
        operation_calls = 0

        def operation():
            nonlocal operation_calls
            operation_calls += 1
            raise _FakeSheetError(403)

        with patch("sheet_config.time.sleep") as sleep:
            with self.assertRaises(_FakeSheetError):
                sheet_config._retry_sheet_request(
                    operation,
                    "test read",
                    delays=(5, 10),
                )

        self.assertEqual(operation_calls, 1)
        sleep.assert_not_called()

    def test_load_config_retries_the_actual_value_read(self):
        class Worksheet:
            def __init__(self):
                self.calls = 0

            def get(self, range_name):
                self.calls += 1
                if range_name != "A:B":
                    raise AssertionError(range_name)
                if self.calls == 1:
                    raise _FakeSheetError(429)
                return [["Parameter", "Value"], ["round", "2"]]

        worksheet = Worksheet()
        with (
            patch("sheet_config._connect_sheet", return_value=worksheet),
            patch("sheet_config.time.sleep") as sleep,
        ):
            config = sheet_config.load_config(verbose=False)

        self.assertEqual(config["round_num"], 2)
        self.assertEqual(worksheet.calls, 2)
        sleep.assert_called_once_with(5.0)


if __name__ == "__main__":
    unittest.main()
