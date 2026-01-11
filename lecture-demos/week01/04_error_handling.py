#!/usr/bin/env python3
"""
Week 01: Error Handling
Gracefully handling API failures
"""

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# =============================================================================
# PART 1: Check Status Codes
# =============================================================================

# Test different status codes
for code in [200, 404, 500]:
    response = requests.get(f"https://nipun-api-testing.hf.space/status/{code}")
    print(f"Status {code}: {response.status_code} - {response.reason}")

# =============================================================================
# PART 2: raise_for_status()
# =============================================================================

try:
    response = requests.get("https://nipun-api-testing.hf.space/status/404")
    response.raise_for_status()  # Raises exception for 4xx/5xx
    print("Success!")
except requests.HTTPError as e:
    print(f"HTTP Error: {e}")

# =============================================================================
# PART 3: Timeouts
# =============================================================================

try:
    # Set a timeout (in seconds)
    response = requests.get(
        "https://nipun-api-testing.hf.space/items",
        timeout=5  # Wait max 5 seconds
    )
    print(f"Got response in time: {response.status_code}")
except Timeout:
    print("Request timed out!")

# =============================================================================
# PART 4: Complete Error Handling Pattern
# =============================================================================

def fetch_items():
    """Robust function to fetch items from API"""
    url = "https://nipun-api-testing.hf.space/items"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    except Timeout:
        print("Error: Request timed out")
        return None

    except ConnectionError:
        print("Error: Could not connect to server")
        return None

    except requests.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code}")
        return None

    except RequestException as e:
        print(f"Error: {e}")
        return None

# Use the function
items = fetch_items()
if items:
    print(f"Got {len(items['items'])} items")

# =============================================================================
# PART 5: Retry Logic (Simple Version)
# =============================================================================

import time

def fetch_with_retry(url, max_retries=3):
    """Fetch URL with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    print("All retries failed")
    return None

# Test with a working endpoint
result = fetch_with_retry("https://nipun-api-testing.hf.space/items")
print(f"Result: {result is not None}")
