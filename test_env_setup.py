"""
Test Environment Setup

Verifies .env credentials and data connections without running simulations.

Usage:
    python test_env_setup.py
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_variables():
    """Check if all required environment variables are set."""
    print("\n" + "="*60)
    print("  ENVIRONMENT VARIABLES TEST")
    print("="*60)
    
    required_vars = {
        "DATAGOLF_API_KEY": "DataGolf API access",
        "EMAIL_USER": "Email sender address",
        "EMAIL_PASSWORD": "Email app password",
        "EMAIL_RECIPIENTS": "Email recipient list",
        "GOOGLE_CREDS_JSON": "Google Sheets credentials"
    }
    
    all_present = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Show partial value for verification
            if var == "GOOGLE_CREDS_JSON":
                try:
                    creds = json.loads(value)
                    display = f"✓ JSON with {len(creds)} keys (email: {creds.get('client_email', 'N/A')[:30]}...)"
                except:
                    display = f"⚠️  Set but invalid JSON"
                    all_present = False
            elif var == "EMAIL_PASSWORD":
                display = f"✓ Set ({len(value)} chars)"
            elif var == "EMAIL_RECIPIENTS":
                recipients = value.split(",")
                display = f"✓ {len(recipients)} recipient(s): {recipients}"
            else:
                display = f"✓ {value[:20]}{'...' if len(value) > 20 else ''}"
            
            print(f"  {var:25} {display}")
        else:
            print(f"  {var:25} ✗ NOT SET")
            all_present = False
    
    print()
    return all_present


def test_google_sheets():
    """Test Google Sheets connection."""
    print("="*60)
    print("  GOOGLE SHEETS CONNECTION TEST")
    print("="*60)
    
    try:
        from sheet_config import _connect_sheet
        ws = _connect_sheet()
        
        # Try to read first cell
        test_value = ws.acell('A1').value
        print(f"  ✓ Connected to Google Sheets")
        print(f"  ✓ Sheet name: {ws.spreadsheet.title}")
        print(f"  ✓ Tab name: {ws.title}")
        print(f"  ✓ First cell (A1): {test_value}")
        print()
        return True
        
    except FileNotFoundError as e:
        print(f"  ✗ Credentials file not found: {e}")
        print()
        return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print()
        return False


def test_datagolf_api():
    """Test DataGolf API connection."""
    print("="*60)
    print("  DATAGOLF API TEST")
    print("="*60)
    
    try:
        import requests
        
        api_key = os.getenv("DATAGOLF_API_KEY")
        if not api_key:
            print("  ✗ DATAGOLF_API_KEY not set")
            print()
            return False
        
        # Test with a simple endpoint (field updates - minimal data transfer)
        url = "https://feeds.datagolf.com/field-updates"
        params = {
            "tour": "pga",
            "file_format": "json",
            "key": api_key
        }
        
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ API connection successful")
            print(f"  ✓ Current tournament: {data.get('tournament_name', 'N/A')}")
            print(f"  ✓ Field size: {len(data.get('field', []))} players")
        elif resp.status_code == 401:
            print(f"  ✗ API authentication failed (invalid key)")
        else:
            print(f"  ✗ API returned status {resp.status_code}")
            print(f"    Response: {resp.text[:100]}")
        
        print()
        return resp.status_code == 200
        
    except Exception as e:
        print(f"  ✗ API test failed: {e}")
        print()
        return False


def test_email_config():
    """Test email configuration (without actually sending)."""
    print("="*60)
    print("  EMAIL CONFIGURATION TEST")
    print("="*60)
    
    email_user = os.getenv("EMAIL_USER")
    email_password = os.getenv("EMAIL_PASSWORD")
    email_recipients = os.getenv("EMAIL_RECIPIENTS", "").split(",")
    
    if not email_user:
        print("  ✗ EMAIL_USER not set")
        print()
        return False
    
    if not email_password:
        print("  ✗ EMAIL_PASSWORD not set")
        print()
        return False
    
    if not email_recipients or email_recipients == ['']:
        print("  ✗ EMAIL_RECIPIENTS not set")
        print()
        return False
    
    # Test SMTP connection (don't send email)
    try:
        import smtplib
        
        print(f"  ✓ Email user: {email_user}")
        print(f"  ✓ Password: {'*' * len(email_password)} ({len(email_password)} chars)")
        print(f"  ✓ Recipients: {', '.join(email_recipients)}")
        print(f"\n  Testing SMTP connection to Gmail...")
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(email_user, email_password)
            print(f"  ✓ SMTP authentication successful")
        
        print()
        return True
        
    except smtplib.SMTPAuthenticationError:
        print(f"  ✗ SMTP authentication failed")
        print(f"    Check your EMAIL_PASSWORD (should be App Password, not regular password)")
        print()
        return False
    except Exception as e:
        print(f"  ✗ SMTP test failed: {e}")
        print()
        return False


def test_data_files():
    """Check for required data files."""
    print("="*60)
    print("  DATA FILES CHECK")
    print("="*60)
    
    # Import to get tournament name
    try:
        from sim_inputs import tourney
        print(f"  Tournament: {tourney}")
        print()
    except:
        tourney = "unknown"
        print("  ⚠️  Could not import sim_inputs.py")
        print()
    
    # Check for common files
    files_to_check = [
        "sim_inputs.py",
        f"pre_sim_summary_{tourney}.csv",
        f"model_predictions_r1.csv",
        f"model_predictions_r2.csv",
        "sheet_config.py",
        "round_sim.py",
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {filename:40} ({size:,} bytes)")
        else:
            print(f"  - {filename:40} (not found - may be ok)")
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  ENVIRONMENT SETUP TEST")
    print("="*60)
    print("  Testing credentials and connections without running sims")
    print("="*60)
    
    results = {
        "Environment Variables": test_env_variables(),
        "Google Sheets": test_google_sheets(),
        "DataGolf API": test_datagolf_api(),
        "Email Config": test_email_config(),
        "Data Files": test_data_files(),
    }
    
    # Summary
    print("="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25} {status}")
    
    print("="*60)
    
    if all(results.values()):
        print("\n  🎉 All tests passed! Ready to run simulations.")
    else:
        print("\n  ⚠️  Some tests failed. Fix issues before running simulations.")
    
    print()


if __name__ == "__main__":
    main()