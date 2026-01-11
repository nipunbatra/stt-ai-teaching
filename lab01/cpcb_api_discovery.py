#!/usr/bin/env python3
"""
CPCB Air Quality API Discovery - Teaching Example
==================================================

This script demonstrates how to reverse-engineer a web API using browser DevTools.

## Discovery Process

1. **Open DevTools** (F12) > Network tab > Filter by "XHR" or "Fetch"
2. **Interact with the website** - select stations, change dates, etc.
3. **Observe the requests** - look for API calls (POST requests with data)
4. **Inspect payloads** - notice the Base64 encoding pattern
5. **Decode and understand** - use base64 decode to see the actual data structure

## Key Findings

### Base64 Encoding
The CPCB API uses Base64 encoding for all POST request bodies:
```javascript
// From main.bundle.js
var bodyString = window.btoa(JSON.stringify(model));
```

### API Endpoints Discovered
1. `/login/dashboard` - Public dashboard data (Delhi NCR AQI)
2. `/caaqms/onload_list_all` - Station list (requires auth now)
3. `/caaqms/fetch_table_data` - Historical data (requires auth now)

### Station ID Format
- Stations: `site_XXXX` (e.g., `site_5456` for SVPI Airport Ahmedabad)
- Parameters: `parameter_XXX` (e.g., `parameter_193` for PM2.5)

### Parameter ID Mappings
```python
PARAMETERS = {
    "PM2.5": "parameter_193",
    "PM10": "parameter_215",
    "NO": "parameter_226",
    "NO2": "parameter_194",
    "NOx": "parameter_225",
    "NH3": "parameter_311",
    "SO2": "parameter_312",
    "CO": "parameter_203",
    "Ozone": "parameter_222",
    "Benzene": "parameter_202",
}
```

## How to Decode API Payloads

```python
import base64
import json

# Encoded payload from DevTools
encoded = "eyJ1c2VyX2lkIjoidXNlcl8xMDEifQ=="

# Decode it
decoded = base64.b64decode(encoded).decode('utf-8')
data = json.loads(decoded)
print(data)  # {'user_id': 'user_101'}
```

## How to Encode API Requests

```python
import base64
import json

payload = {"user_id": "user_101", "type": "new"}
encoded = base64.b64encode(json.dumps(payload).encode()).decode()
print(encoded)  # eyJ1c2VyX2lkIjogInVzZXJfMTAxIiwgInR5cGUiOiAibmV3In0=
```
"""

import requests
import json
import base64
from datetime import datetime


def encode_payload(data: dict) -> str:
    """Encode a dict as Base64 JSON (CPCB API format)."""
    return base64.b64encode(json.dumps(data).encode()).decode()


def decode_payload(encoded: str) -> dict:
    """Decode a Base64 JSON payload."""
    return json.loads(base64.b64decode(encoded).decode())


# =============================================================================
# PUBLIC ENDPOINT - Works without authentication
# =============================================================================

def get_delhi_ncr_dashboard():
    """
    Get current AQI data for Delhi NCR stations.

    This endpoint is PUBLIC and doesn't require authentication.
    Discovered by observing network traffic on the login page.

    Note: Both request AND response are Base64 encoded!
    """
    url = "https://airquality.cpcb.gov.in/login/dashboard"

    # Empty payload, base64 encoded
    payload = encode_payload({})

    response = requests.post(
        url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # Response is ALSO base64 encoded
    return decode_payload(response.text)


# =============================================================================
# AUTHENTICATED ENDPOINTS - Require login (for reference)
# =============================================================================

def get_station_list_payload():
    """
    Returns the payload structure for fetching station list.

    Endpoint: POST /caaqms/onload_list_all
    Note: Now requires authentication.
    """
    return {
        "user_id": "user_101",
        "user_name": "",
        "group_id": "user_group_101",
        "user_group": "",
        "type": "new"
    }


def get_station_data_payload(station_id: str, state: str, city: str,
                              from_date: str, to_date: str):
    """
    Returns the payload structure for fetching station data.

    Endpoint: POST /caaqms/fetch_table_data
    Note: Now requires authentication.

    Args:
        station_id: e.g., "site_5456"
        state: e.g., "Gujarat"
        city: e.g., "Ahmedabad"
        from_date: e.g., "08-01-2026 T00:00:00Z"
        to_date: e.g., "09-01-2026 T05:51:59Z"
    """
    # Parameter mappings discovered from API responses
    PARAMETERS = {
        "PM2.5": "parameter_193",
        "PM10": "parameter_215",
        "NO": "parameter_226",
        "NO2": "parameter_194",
        "NOx": "parameter_225",
        "NH3": "parameter_311",
        "SO2": "parameter_312",
        "CO": "parameter_203",
        "Ozone": "parameter_222",
    }

    return {
        "draw": 1,
        "columns": [{
            "data": 0,
            "name": "",
            "searchable": True,
            "orderable": False,
            "search": {"value": "", "regex": False}
        }],
        "order": [],
        "start": 0,
        "length": 10,
        "search": {"value": "", "regex": False},
        "filtersToApply": {
            "parameter_list": [
                {"id": i, "itemName": name, "itemValue": pid}
                for i, (name, pid) in enumerate(PARAMETERS.items())
            ],
            "criteria": "24 Hours",
            "reportFormat": "Tabular",
            "fromDate": from_date,
            "toDate": to_date,
            "state": state,
            "city": city,
            "station": station_id,
            "parameter": list(PARAMETERS.values()),
            "parameterNames": list(PARAMETERS.keys()),
        },
        "pagination": 1
    }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CPCB Air Quality API Discovery Demo")
    print("=" * 70)

    # Demonstrate Base64 encoding/decoding
    print("\n[1] Base64 Encoding Demo")
    print("-" * 40)

    sample_payload = {"user_id": "user_101", "type": "new"}
    encoded = encode_payload(sample_payload)
    decoded = decode_payload(encoded)

    print(f"Original:  {sample_payload}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")

    # Fetch public dashboard data
    print("\n[2] Fetching Public Dashboard Data")
    print("-" * 40)

    try:
        data = get_delhi_ncr_dashboard()

        print(f"\nDelhi NCR Overall: AQI {data['ncr_aqi']['avg']} ({data['ncr_aqi']['status']})")

        print("\nStation-wise AQI (first 10):")
        for station in data.get('aqi', [])[:10]:
            name = station['station_name'][:40]
            avg = station['avg']
            param = station['parameter']
            status = station['status']
            print(f"  {name:42} | {param:6} {avg:5.0f} | {status}")

    except Exception as e:
        print(f"Error fetching data: {e}")

    # Show payload structures for reference
    print("\n[3] API Payload Structures (for reference)")
    print("-" * 40)

    print("\nStation List Payload:")
    print(json.dumps(get_station_list_payload(), indent=2))

    print("\nStation Data Payload (example):")
    payload = get_station_data_payload(
        "site_5456", "Gujarat", "Ahmedabad",
        "08-01-2026 T00:00:00Z", "09-01-2026 T05:51:59Z"
    )
    print(json.dumps(payload, indent=2)[:1000] + "...")
