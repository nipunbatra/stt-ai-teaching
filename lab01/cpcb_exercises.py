#!/usr/bin/env python3
"""
CPCB API Discovery - Student Exercises
=======================================

Run this file and complete the exercises!
"""

import base64
import json

print("=" * 70)
print("CPCB API Discovery - Exercises")
print("=" * 70)

# =============================================================================
# EXERCISE 1: Decode a payload
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 1: Decode this payload from DevTools")
print("=" * 70)

encoded_payload = "eyJ1c2VyX2lkIjoidXNlcl8xMDEiLCJ1c2VyX25hbWUiOiIiLCJncm91cF9pZCI6InVzZXJfZ3JvdXBfMTAxIiwidXNlcl9ncm91cCI6IiIsInR5cGUiOiJuZXcifQ=="

print(f"\nEncoded: {encoded_payload[:50]}...")
print("\nTODO: Write code to decode this Base64 string")
print("Hint: Use base64.b64decode() and .decode('utf-8')")

# YOUR CODE HERE:
# decoded = ???
# print(decoded)

# SOLUTION (uncomment to check):
# decoded = base64.b64decode(encoded_payload).decode('utf-8')
# print(f"Decoded: {decoded}")

input("\nPress Enter to see the solution...")
decoded = base64.b64decode(encoded_payload).decode('utf-8')
print(f"\nSOLUTION:")
print(f"Decoded: {decoded}")
print(f"\nAs Python dict:")
print(json.dumps(json.loads(decoded), indent=2))


# =============================================================================
# EXERCISE 2: Identify components in a real API payload
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Find the station ID and parameters")
print("=" * 70)

real_payload = "eyJkcmF3IjoxLCJmaWx0ZXJzVG9BcHBseSI6eyJzdGF0ZSI6Ikd1amFyYXQiLCJjaXR5IjoiQWhtZWRhYmFkIiwic3RhdGlvbiI6InNpdGVfNTQ1NiIsInBhcmFtZXRlciI6WyJwYXJhbWV0ZXJfMTkzIiwicGFyYW1ldGVyXzIxNSJdLCJwYXJhbWV0ZXJOYW1lcyI6WyJQTTIuNSIsIlBNMTAiXX19"

print(f"\nThis is a real request to fetch air quality data.")
print(f"Encoded: {real_payload[:50]}...")

print("\nTODO: Decode this and answer:")
print("  1. What state is this request for?")
print("  2. What is the station ID?")
print("  3. What parameters are being requested?")

input("\nPress Enter to see the solution...")

data = json.loads(base64.b64decode(real_payload))
print(f"\nSOLUTION:")
print(json.dumps(data, indent=2))
print(f"\nAnswers:")
print(f"  1. State: {data['filtersToApply']['state']}")
print(f"  2. Station ID: {data['filtersToApply']['station']}")
print(f"  3. Parameters: {data['filtersToApply']['parameterNames']}")


# =============================================================================
# EXERCISE 3: Encode your own request
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Create a request for Mumbai data")
print("=" * 70)

print("\nTODO: Create a payload to request PM2.5 data for Mumbai")
print("Use this structure:")
print("""
{
    "state": "???",
    "city": "???",
    "station": "site_???",  # You'd find this in DevTools
    "parameter": ["parameter_193"]  # PM2.5
}
""")

print("\nHint: The encoding is just:")
print("  base64.b64encode(json.dumps(your_dict).encode()).decode()")

input("\nPress Enter to see an example...")

mumbai_payload = {
    "state": "Maharashtra",
    "city": "Mumbai",
    "station": "site_1234",  # Placeholder - find real ID in DevTools!
    "parameter": ["parameter_193"],
    "parameterNames": ["PM2.5"]
}

encoded = base64.b64encode(json.dumps(mumbai_payload).encode()).decode()
print(f"\nSOLUTION:")
print(f"Payload: {json.dumps(mumbai_payload, indent=2)}")
print(f"\nEncoded: {encoded}")


# =============================================================================
# EXERCISE 4: Recognize Base64
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Which of these are Base64 encoded JSON?")
print("=" * 70)

samples = [
    ("eyJuYW1lIjoiSm9obiJ9", "Sample A"),
    ("Hello World!", "Sample B"),
    ("eyJhZ2UiOjI1fQ==", "Sample C"),
    ("7B226E616D65223A224A6F686E227D", "Sample D"),  # This is hex
    ("e30=", "Sample E"),
]

print("\nExamine these strings and guess which are Base64 JSON:\n")
for sample, label in samples:
    print(f"  {label}: {sample}")

print("\nHints:")
print("  - Base64 JSON usually starts with 'ey' (because '{\"' encodes to 'ey')")
print("  - Base64 only contains: A-Z, a-z, 0-9, +, /, =")
print("  - Padding (=) appears at the end")

input("\nPress Enter to see answers...")

print("\nSOLUTION:")
for sample, label in samples:
    try:
        decoded = base64.b64decode(sample).decode('utf-8')
        parsed = json.loads(decoded)
        print(f"  {label}: YES - decodes to {parsed}")
    except Exception as e:
        print(f"  {label}: NO - {type(e).__name__}")


# =============================================================================
# EXERCISE 5: Real API call
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Make a real API call")
print("=" * 70)

print("""
Try this in your Python environment:

```python
import requests
import base64
import json

def encode(data):
    return base64.b64encode(json.dumps(data).encode()).decode()

def decode(text):
    return json.loads(base64.b64decode(text))

# This endpoint is public!
response = requests.post(
    "https://airquality.cpcb.gov.in/login/dashboard",
    data=encode({}),
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

data = decode(response.text)

# Print all station AQIs
for station in data['aqi']:
    print(f"{station['station_name']}: {station['avg']} ({station['status']})")
```
""")

input("\nPress Enter to run this code...")

import requests

def encode(data):
    return base64.b64encode(json.dumps(data).encode()).decode()

def decode(text):
    return json.loads(base64.b64decode(text))

print("\nFetching live data...")
response = requests.post(
    "https://airquality.cpcb.gov.in/login/dashboard",
    data=encode({}),
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

data = decode(response.text)

print(f"\nDelhi NCR AQI: {data['ncr_aqi']['avg']} ({data['ncr_aqi']['status']})")
print("\nStation-wise data:")
for station in data['aqi'][:10]:
    name = station['station_name'][:45]
    print(f"  {name:47} | {station['parameter']:6} {station['avg']:5.0f} | {station['status']}")


print("\n" + "=" * 70)
print("BONUS: DevTools Challenge")
print("=" * 70)
print("""
Open https://airquality.cpcb.gov.in/ccr/ in Chrome and:

1. Open DevTools (F12) → Network tab → Filter: Fetch/XHR
2. Navigate to a station in your city
3. Find the fetch_table_data request
4. Copy the Request Payload
5. Decode it in Console: atob("YOUR_PAYLOAD")
6. Find your station's ID (site_XXXX)

Share your findings:
  - City: ___________
  - Station Name: ___________
  - Station ID: site_____
""")

print("\n" + "=" * 70)
print("Exercises Complete!")
print("=" * 70)
