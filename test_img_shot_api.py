import unittest
from unittest.mock import Mock, patch

from api_utils import fetch_img_player_rounds, fetch_img_player_shots


def response(payload, status=200):
    result = Mock()
    result.status_code = status
    result.json.return_value = payload
    return result


class ImgShotApiTests(unittest.TestCase):
    @patch("api_utils.requests.get")
    def test_fetch_player_rounds_uses_read_token_and_sim_name_aliases(self, get):
        get.return_value = response(
            {
                "ok": True,
                "rows": [
                    {
                        "tournament_id": 1400,
                        "round_no": 4,
                        "img_player_id": "8520",
                        "player_name": "stevens, sam",
                        "holes_completed": 18,
                        "round_strokes": 69,
                        "complete": True,
                    }
                ],
            }
        )

        frame = fetch_img_player_rounds(
            1400,
            4,
            base_url="https://shots.example",
            read_token="read-secret",
        )

        self.assertEqual(frame.loc[0, "player_name"], "stevens, samuel")
        self.assertEqual(frame.loc[0, "round_strokes"], 69)
        args, kwargs = get.call_args
        self.assertEqual(args[0], "https://shots.example/v1/player-rounds")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer read-secret")
        self.assertEqual(kwargs["params"], {"event_id": 1400, "round": 4})

    @patch("api_utils.requests.get")
    def test_fetch_player_shots_follows_cursor_pages(self, get):
        get.side_effect = [
            response(
                {
                    "ok": True,
                    "rows": [
                        {
                            "shot_key": "a",
                            "player_name": "hojgaard, rasmus",
                            "shot_no": 1,
                        }
                    ],
                    "next_cursor": "a",
                }
            ),
            response(
                {
                    "ok": True,
                    "rows": [
                        {
                            "shot_key": "b",
                            "player_name": "hojgaard, rasmus",
                            "shot_no": 2,
                        }
                    ],
                    "next_cursor": None,
                }
            ),
        ]

        frame = fetch_img_player_shots(
            1400,
            4,
            base_url="https://shots.example",
            read_token="read-secret",
            page_size=1,
        )

        self.assertEqual(frame["shot_key"].tolist(), ["a", "b"])
        self.assertEqual(get.call_count, 2)
        self.assertNotIn("cursor", get.call_args_list[0].kwargs["params"])
        self.assertEqual(get.call_args_list[1].kwargs["params"]["cursor"], "a")


if __name__ == "__main__":
    unittest.main()
