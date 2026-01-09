#!/bin/bash
# =============================================================================
# Week 01: Data Collection - curl Demo Commands
# =============================================================================
# Run these commands one by one during the lecture
# Copy-paste friendly - each section is independent
# =============================================================================

# -----------------------------------------------------------------------------
# PART 1: Basic GET Requests (No API key needed!)
# -----------------------------------------------------------------------------

# Simple hello world
curl https://nipun-api-testing.hf.space/hello

# Get list of items
curl https://nipun-api-testing.hf.space/items

# Query parameters
curl "https://nipun-api-testing.hf.space/greet?name=Alice"

# -----------------------------------------------------------------------------
# PART 2: Different Response Formats
# -----------------------------------------------------------------------------

# JSON format
curl https://nipun-api-testing.hf.space/format/json

# XML format
curl https://nipun-api-testing.hf.space/format/xml

# CSV format
curl https://nipun-api-testing.hf.space/format/csv

# -----------------------------------------------------------------------------
# PART 3: Pretty Printing with jq
# -----------------------------------------------------------------------------

# Raw output (hard to read)
curl -s https://nipun-api-testing.hf.space/items

# Pretty printed with jq
curl -s https://nipun-api-testing.hf.space/items | jq .

# Extract just the items array
curl -s https://nipun-api-testing.hf.space/items | jq '.items'

# Get first item only
curl -s https://nipun-api-testing.hf.space/items | jq '.items[0]'

# Get all names
curl -s https://nipun-api-testing.hf.space/items | jq '.items[].name'

# Transform to new structure
curl -s https://nipun-api-testing.hf.space/items | jq '.items[] | {product: .name, cost: .price}'

# -----------------------------------------------------------------------------
# PART 4: Viewing Headers
# -----------------------------------------------------------------------------

# See response headers only (-I = HEAD request)
curl -I https://nipun-api-testing.hf.space/items

# Verbose mode - see EVERYTHING (request + response)
curl -v https://nipun-api-testing.hf.space/hello 2>&1 | head -30

# See what headers YOUR request sends
curl https://nipun-api-testing.hf.space/headers | jq .

# -----------------------------------------------------------------------------
# PART 5: Adding Custom Headers
# -----------------------------------------------------------------------------

# Add Accept header
curl -H "Accept: application/json" https://nipun-api-testing.hf.space/items

# Add multiple headers
curl -H "Accept: application/json" \
     -H "User-Agent: MyApp/1.0" \
     https://nipun-api-testing.hf.space/headers | jq '.your_headers'

# -----------------------------------------------------------------------------
# PART 6: POST Requests
# -----------------------------------------------------------------------------

# POST with JSON body
curl -X POST https://nipun-api-testing.hf.space/items \
     -H "Content-Type: application/json" \
     -d '{"name": "Mango", "price": 2.50, "quantity": 50}'

# POST with form data
curl -X POST https://nipun-api-testing.hf.space/form/contact \
     -d "name=Alice" \
     -d "email=alice@example.com" \
     -d "message=Hello from curl!"

# Echo endpoint - see what you sent
curl -X POST https://nipun-api-testing.hf.space/echo \
     -H "Content-Type: application/json" \
     -d '{"test": "data", "number": 42}'

# -----------------------------------------------------------------------------
# PART 7: Saving Output
# -----------------------------------------------------------------------------

# Save to file
curl -s https://nipun-api-testing.hf.space/items -o items.json

# Save pretty-printed
curl -s https://nipun-api-testing.hf.space/items | jq . > items_pretty.json

# Check the files
cat items.json
cat items_pretty.json

# Cleanup
rm -f items.json items_pretty.json

# -----------------------------------------------------------------------------
# PART 8: Real API Example (OMDb - needs API key)
# -----------------------------------------------------------------------------

# Get your free key at: https://www.omdbapi.com/apikey.aspx
# Replace YOUR_KEY with your actual key

# curl "https://www.omdbapi.com/?t=Inception&apikey=YOUR_KEY" | jq .
# curl "https://www.omdbapi.com/?t=Avatar&apikey=YOUR_KEY" | jq '.Title, .Year, .imdbRating'

# -----------------------------------------------------------------------------
# PART 9: Error Handling & Debugging
# -----------------------------------------------------------------------------

# Test different status codes
curl https://nipun-api-testing.hf.space/status/200
curl https://nipun-api-testing.hf.space/status/404
curl https://nipun-api-testing.hf.space/status/500

# Follow redirects (-L)
curl -L https://nipun-api-testing.hf.space/status/301

# Timeout (useful for slow servers)
curl --max-time 5 https://nipun-api-testing.hf.space/items

# -----------------------------------------------------------------------------
# PART 10: robots.txt Examples
# -----------------------------------------------------------------------------

# Check robots.txt for various sites
curl https://www.google.com/robots.txt | head -20
curl https://www.amazon.com/robots.txt | head -20
curl https://twitter.com/robots.txt | head -20

echo "=== Demo Complete ==="
