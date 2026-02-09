"""
update_sheet_courses.py - Auto-populate course_codes in Google Sheet

Run before R1 to detect course codes from the DataGolf field API.
Writes course_codes to the round_config sheet tab.

Usage:
    python update_sheet_courses.py
"""

import os
import json
import requests
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
SHEET_NAME = "golf_sims"
TAB_NAME = "round_config"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",  # read-write
    "https://www.googleapis.com/auth/drive",
]


def get_course_codes():
    """Fetch unique course codes from DataGolf field-updates API."""
    params = {"tour": "pga", "file_format": "json", "key": API_KEY}
    resp = requests.get("https://feeds.datagolf.com/field-updates", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    tourney_name = data.get("tournament_name", "Unknown")
    field = data.get("field", [])

    # Extract unique course values
    courses = set()
    for player in field:
        course = player.get("course")
        if course and str(course).strip():
            courses.add(str(course).strip())

    courses_sorted = sorted(courses)
    print(f"Tournament: {tourney_name}")
    print(f"Field size: {len(field)} players")
    print(f"Course codes found: {courses_sorted}")

    return courses_sorted, tourney_name


def connect_sheet_writable():
    """Connect to Google Sheet with WRITE permissions."""
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    else:
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)

    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_NAME)
    return spreadsheet.worksheet(TAB_NAME)


def update_course_codes(ws, course_codes):
    """Find the course_codes row and update its value."""
    all_values = ws.get("A:B")

    target_row = None
    for i, row in enumerate(all_values):
        if len(row) >= 1 and row[0].strip().lower() == "course_codes":
            target_row = i + 1  # 1-indexed for gspread
            break

    if target_row is None:
        # Append it
        next_row = len(all_values) + 1
        ws.update_cell(next_row, 1, "course_codes")
        ws.update_cell(next_row, 2, ",".join(course_codes))
        print(f"Added course_codes at row {next_row}: {','.join(course_codes)}")
    else:
        ws.update_cell(target_row, 2, ",".join(course_codes))
        print(f"Updated course_codes at row {target_row}: {','.join(course_codes)}")


if __name__ == "__main__":
    courses, tourney = get_course_codes()

    if not courses:
        print("No course codes found in API response. Single-course week or API issue.")
    else:
        ws = connect_sheet_writable()
        update_course_codes(ws, courses)

        print(f"\n{'='*50}")
        print(f"NEXT STEPS - fill in manually:")
        print(f"{'='*50}")
        for i, code in enumerate(courses):
            print(f"  Course {i+1}: '{code}'")
        print(f"\n  -> Set 'course_pars' to the par for each course (comma-separated, same order)")
        print(f"  -> Set 'expected_score_rN' values (comma-separated if multi-course)")
        print(f"\nExample for Pebble Beach week:")
        print(f"  course_codes:      PB,SG,ML")
        print(f"  course_pars:       72,72,71")
        print(f"  expected_score_r2: 72.1,71.5,70.8")
