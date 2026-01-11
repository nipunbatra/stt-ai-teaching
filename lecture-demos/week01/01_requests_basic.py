#!/usr/bin/env python3
"""
Week 01: Basic GET Requests with requests library
Run each section separately to demonstrate
"""

import requests

# =============================================================================
# PART 1: Simple GET Request
# =============================================================================

response = requests.get("https://nipun-api-testing.hf.space/hello")
print(response.text)
print(f"Status: {response.status_code}")

# =============================================================================
# PART 2: JSON Response
# =============================================================================

response = requests.get("https://nipun-api-testing.hf.space/items")
data = response.json()

print("Items from API:")
for item in data["items"]:
    print(f"  - {item['name']}: ${item['price']}")

# =============================================================================
# PART 3: Query Parameters
# =============================================================================

# Method 1: Manual URL
response = requests.get("https://nipun-api-testing.hf.space/greet?name=Alice")
print(response.json())

# Method 2: Using params dict (cleaner!)
response = requests.get(
    "https://nipun-api-testing.hf.space/greet",
    params={"name": "Bob"}
)
print(response.json())

# =============================================================================
# PART 4: Response Object Properties
# =============================================================================

response = requests.get("https://nipun-api-testing.hf.space/items")

print(f"Status Code: {response.status_code}")
print(f"Content-Type: {response.headers['Content-Type']}")
print(f"Response Size: {len(response.content)} bytes")
print(f"Encoding: {response.encoding}")

# =============================================================================
# PART 5: Different Response Formats
# =============================================================================

# JSON format
json_resp = requests.get("https://nipun-api-testing.hf.space/format/json")
print("JSON:", json_resp.json())

# CSV format
csv_resp = requests.get("https://nipun-api-testing.hf.space/format/csv")
print("CSV:", csv_resp.text)

# XML format
xml_resp = requests.get("https://nipun-api-testing.hf.space/format/xml")
print("XML:", xml_resp.text)
