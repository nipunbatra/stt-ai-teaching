# API Discovery with Chrome DevTools - Teaching Guide

## Lab: Reverse Engineering the CPCB Air Quality API

**Objective**: Learn to discover hidden APIs, understand encoding patterns, and extract data programmatically.

---

## Part 1: Opening DevTools and Finding API Calls

### Step 1: Open Chrome DevTools

1. Go to: https://airquality.cpcb.gov.in/ccr/
2. Press `F12` (or `Cmd+Option+I` on Mac)
3. Click the **Network** tab

### Step 2: Filter for API Calls

In the Network tab, you'll see a filter bar. Click:
- **Fetch/XHR** - This filters out images, CSS, JS and shows only data requests

```
┌─────────────────────────────────────────────────────────────┐
│ Network                                                      │
├─────────────────────────────────────────────────────────────┤
│ [All] [Fetch/XHR] [Doc] [CSS] [JS] [Font] [Img] [Media]     │  ← Click "Fetch/XHR"
├─────────────────────────────────────────────────────────────┤
│ Name              Status    Type      Size     Time          │
│ dashboard         200       fetch     17 KB    234 ms        │  ← API calls appear here
│ onload_list_all   200       fetch     45 KB    567 ms        │
└─────────────────────────────────────────────────────────────┘
```

### Step 3: Trigger API Calls

- Navigate through the website
- Select different states, cities, stations
- Watch new requests appear in the Network tab

---

## Part 2: Inspecting a Request

### Step 4: Click on a Request

Click on any request (e.g., `fetch_table_data`) to see details:

```
┌─────────────────────────────────────────────────────────────┐
│ Headers  Preview  Response  Initiator  Timing               │
├─────────────────────────────────────────────────────────────┤
│ General                                                      │
│   Request URL: https://airquality.cpcb.gov.in/caaqms/fetch_table_data
│   Request Method: POST                                       │
│   Status Code: 200 OK                                        │
├─────────────────────────────────────────────────────────────┤
│ Request Headers                                              │
│   Content-Type: application/x-www-form-urlencoded           │
│   Accept: application/json                                   │
├─────────────────────────────────────────────────────────────┤
│ Request Payload                                              │
│   eyJkcmF3IjoxLCJjb2x1bW5zIjpbeyJkYXRhIjowLC...             │  ← This looks like Base64!
└─────────────────────────────────────────────────────────────┘
```

### Step 5: Identify the Encoding

The payload `eyJkcmF3IjoxLC...` has telltale signs of **Base64**:
- Starts with `ey` (which is `{"` in Base64)
- Contains only A-Z, a-z, 0-9, +, /, =
- Often ends with `=` or `==` (padding)

**Quick test**: Base64 of `{"` is always `ey`

---

## Part 3: Decoding in the Console

### Step 6: Open Console Tab

Click the **Console** tab in DevTools (next to Network)

### Step 7: Decode the Payload

Copy the payload from the Request and decode it:

```javascript
// Paste this in Console:
atob("eyJ1c2VyX2lkIjoidXNlcl8xMDEiLCJ1c2VyX25hbWUiOiIiLCJncm91cF9pZCI6InVzZXJfZ3JvdXBfMTAxIiwidXNlcl9ncm91cCI6IiIsInR5cGUiOiJuZXcifQ==")

// Output:
'{"user_id":"user_101","user_name":"","group_id":"user_group_101","user_group":"","type":"new"}'
```

### Step 8: Pretty Print the JSON

```javascript
// For better readability:
JSON.parse(atob("eyJ1c2VyX2lkIjoidXNlcl8xMDEifQ=="))

// Or format it nicely:
console.log(JSON.stringify(JSON.parse(atob("YOUR_BASE64_HERE")), null, 2))
```

---

## Part 4: Finding Station IDs

### Step 9: Select a Different City

1. On the CPCB website, change the **State** dropdown to "Gujarat"
2. Change **City** to "Ahmedabad"
3. Select a **Station** (e.g., "SVPI Airport Hansol")
4. Watch the Network tab for new requests

### Step 10: Find the Station ID in the Request

Click on the `fetch_table_data` request that appeared:

```javascript
// Decode the payload - look for "station" field:
{
  "filtersToApply": {
    "state": "Gujarat",
    "city": "Ahmedabad",
    "station": "site_5456",        // ← THIS IS THE STATION ID!
    "parameter": ["parameter_193", "parameter_215", ...]
  }
}
```

### Step 11: Map Parameter IDs

From the same payload, you can see parameter mappings:

```javascript
"parameter_list": [
  {"itemName": "PM2.5", "itemValue": "parameter_193"},   // PM2.5 = 193
  {"itemName": "PM10", "itemValue": "parameter_215"},    // PM10 = 215
  {"itemName": "NO2", "itemValue": "parameter_194"},     // NO2 = 194
  ...
]
```

---

## Part 5: Finding the Source Code

Now let's find WHERE in the code this encoding happens.

### Step 12: See All JS Files Loaded

```
1. Stay in DevTools
2. Click Network tab → Refresh page (Cmd+R)
3. Filter by "JS" (click the JS button in filter bar)
```

You'll see:
```
┌─────────────────────────────────────────────────────────────────┐
│ Name                    Size       Time                         │
├─────────────────────────────────────────────────────────────────┤
│ inline.bundle.js        1.5 KB     15 ms    ← tiny, skip        │
│ polyfills.bundle.js     65 KB      45 ms    ← browser compat    │
│ scripts.bundle.js       150 KB     89 ms    ← libraries         │
│ styles.bundle.js        12 KB      23 ms    ← CSS stuff         │
│ vendor.bundle.js        2.1 MB     234 ms   ← frameworks        │
│ main.bundle.js          890 KB     156 ms   ← APP CODE! ✓       │
└─────────────────────────────────────────────────────────────────┘
```

**Rule**: Look for `main`, `app`, or `index` bundle - that's the application code.

### Step 13: Search Across ALL Files

Instead of opening each file, search across everything:

```
1. Go to "Sources" tab
2. Press Cmd+Shift+F (Mac) or Ctrl+Shift+F (Windows)
   This opens "Search" panel at bottom
3. Type: btoa
4. Press Enter
```

Results show which file(s) contain `btoa`:
```
┌─────────────────────────────────────────────────────────────────┐
│ Search Results (4 results in 1 file)                            │
├─────────────────────────────────────────────────────────────────┤
│ ▼ main.bundle.js                                                │
│     line 1: ...var bodyString = window.btoa(JSON.stringify...   │
│     line 1: ...let JsonSend = window.btoa(JSON.stringify...     │
│     line 1: ...var bodyString = window.btoa(JSON.stringify...   │
└─────────────────────────────────────────────────────────────────┘
```

**This tells you**: `main.bundle.js` is where the encoding happens!

### Step 14: Open the File and Navigate

```
1. Click on any search result
2. File opens with that line highlighted
3. Or: Press Cmd+P → type "main.bundle" → Enter
```

### Step 15: Pretty Print the Code

The code is minified (one long line). Make it readable:

```
1. Click the "{}" button at bottom-left of the code panel
   (tooltip says "Pretty print")
2. Code becomes formatted and readable
```

Before:
```javascript
var bodyString=window.btoa(JSON.stringify(model));this._http.post(url,bodyString)
```

After:
```javascript
var bodyString = window.btoa(JSON.stringify(model));
this._http.post(url, bodyString)
```

### Step 16: Find API Endpoints

Search for: `CAAQMS` or `API_END_POINT`

```javascript
// You'll find definitions like:
CAAQMS_END_POINT: 'https://airquality.cpcb.gov.in/caaqms/'
CAAQMS_VIEW_PDF_DOWNLOAD: CAAQMS_END_POINT + 'caaqms_view_data_pdf'
```

### What to Search For

| Search Term | What You'll Find |
|-------------|------------------|
| `btoa` | Where requests are encoded |
| `atob` | Where responses are decoded |
| `API_END_POINT` | Base URLs for APIs |
| `CAAQMS` | Air quality specific endpoints |
| `/caaqms/` | API route patterns |
| `fetch` or `post` | HTTP calls |

---

## Part 6: Quick Reference

### Common Bundle Names

| File Pattern | Contains |
|--------------|----------|
| `main.*.js`, `app.*.js` | Your target - app code |
| `vendor.*.js` | Frameworks (Angular, React, Vue) |
| `polyfills.*.js` | Browser compatibility |
| `runtime.*.js`, `inline.*.js` | Webpack bootstrap |
| `styles.*.js` | CSS |
| `chunk.*.js`, `lazy.*.js` | Code-split modules |

### Base64 Cheat Sheet

```javascript
// Encode (JavaScript)
btoa(JSON.stringify({key: "value"}))  // → "eyJrZXkiOiAidmFsdWUifQ=="

// Decode (JavaScript)
atob("eyJrZXkiOiAidmFsdWUifQ==")      // → '{"key": "value"}'

// Decode (Python)
import base64, json
json.loads(base64.b64decode("eyJrZXkiOiAidmFsdWUifQ=="))

# Decode (Command line)
echo "eyJrZXkiOiAidmFsdWUifQ==" | base64 -d
```

### Recognizing Base64

| Pattern | Meaning |
|---------|---------|
| Starts with `ey` | Likely JSON starting with `{` |
| Ends with `=` or `==` | Base64 padding |
| Only `A-Za-z0-9+/=` | Valid Base64 charset |
| Length divisible by 4 | Valid Base64 |

### Common ID Patterns (CPCB)

| Entity | Format | Example |
|--------|--------|---------|
| Station | `site_XXXX` | `site_5456` |
| Parameter | `parameter_XXX` | `parameter_193` |
| Organization | `organisation_XXX` | `organisation_372` |
| User Group | `user_group_XXX` | `user_group_101` |

---

## Part 7: Hands-On Exercises

### Exercise 1: Find the Station ID for IIT Delhi

1. Open https://airquality.cpcb.gov.in/ccr/
2. Open DevTools → Network → Filter by Fetch/XHR
3. Select State: "Delhi", City: "Delhi"
4. Look for a station near IIT Delhi
5. Capture the `fetch_table_data` request
6. Decode the payload using `atob()` in Console
7. Find the `station` field value

**Expected format**: `site_XXXX`

### Exercise 2: Decode a Response

The response is ALSO Base64 encoded. Try:

1. Click on a successful request in Network tab
2. Go to **Response** tab
3. Copy the response text
4. Decode it: `JSON.parse(atob("RESPONSE_TEXT"))`

### Exercise 3: Find the Encoding in Source

1. Go to Sources tab
2. Press Cmd+Shift+F
3. Search for `btoa`
4. Click the result
5. Pretty print with `{}`
6. Find the line: `window.btoa(JSON.stringify(model))`

---

## Part 8: Python Script to Fetch Data

```python
import requests
import base64
import json

def encode(data):
    return base64.b64encode(json.dumps(data).encode()).decode()

def decode(text):
    return json.loads(base64.b64decode(text))

# Fetch Delhi NCR dashboard (public endpoint)
response = requests.post(
    "https://airquality.cpcb.gov.in/login/dashboard",
    data=encode({}),
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

data = decode(response.text)
print(f"Delhi NCR AQI: {data['ncr_aqi']['avg']} ({data['ncr_aqi']['status']})")
```

---

## Part 9: Demo Script for Class

```
"Let's reverse engineer this air quality website's API..."

1. [Open Network tab, filter Fetch/XHR]
   "First, let's see what API calls the site makes"

2. [Click on a request, show payload]
   "See this gibberish? eyJkcmF3... - that's Base64 encoded"

3. [Open Console, decode with atob()]
   "We can decode it: atob('eyJ...') - now we see the JSON!"

4. [Change city, capture new request]
   "When I select Ahmedabad, a new request appears with station ID"

5. [Open Network tab, filter JS]
   "Now let's find WHERE this encoding happens in the code"
   "See these 6 JavaScript files? Which one has the app code?"

6. [Point to main.bundle.js]
   "main.bundle.js - 'main' usually means application code"

7. [Open Sources, Cmd+Shift+F, search 'btoa']
   "Instead of guessing, let's search ALL files for 'btoa'"

8. [Show results]
   "Found it! main.bundle.js, 4 matches"

9. [Click result, pretty print]
   "Here's the line: window.btoa(JSON.stringify(model))"
   "So the pattern is: Object → JSON → Base64 → send to server"
```

---

## Part 10: Keyboard Shortcuts Summary

| Action | Mac | Windows |
|--------|-----|---------|
| Open DevTools | Cmd+Option+I | Ctrl+Shift+I |
| Search in file | Cmd+F | Ctrl+F |
| Search ALL files | Cmd+Shift+F | Ctrl+Shift+F |
| Open file by name | Cmd+P | Ctrl+P |
| Pretty print | Click `{}` | Click `{}` |

---

## Summary: The Discovery Process

```
1. OBSERVE    → Open DevTools, filter XHR, interact with site
2. CAPTURE    → Click on requests, examine payloads
3. RECOGNIZE  → Identify encoding (Base64 starts with 'ey' for JSON)
4. DECODE     → Use atob() in Console or base64 in Python
5. UNDERSTAND → Map IDs (stations, parameters) from decoded data
6. FIND CODE  → Sources tab, Cmd+Shift+F, search 'btoa'
7. VERIFY     → Pretty print, read the encoding logic
8. REPLICATE  → Write Python code to make the same requests
```
