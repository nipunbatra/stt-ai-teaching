#!/bin/bash
# Week 02: jq - JSON Processing
# Run sections separately to demonstrate in lecture

DATA_DIR="$(dirname "$0")/data"
cd "$DATA_DIR" || exit 1

# =============================================================================
# PART 1: Pretty Printing
# =============================================================================

echo "=== Pretty Print JSON ==="
echo ""

# Identity filter - just format
cat movie.json | jq .

# =============================================================================
# PART 2: Extracting Fields
# =============================================================================

echo ""
echo "=== Extract Fields ==="
echo ""

# Single field
echo "--- Title ---"
cat movie.json | jq '.Title'

# Multiple fields
echo ""
echo "--- Title and Year ---"
cat movie.json | jq '.Title, .Year'

# Nested field
echo ""
echo "--- First Rating ---"
cat movie.json | jq '.Ratings[0]'

# =============================================================================
# PART 3: Working with Arrays
# =============================================================================

echo ""
echo "=== Arrays ==="
echo ""

# Array length
echo "--- Number of movies ---"
cat movies.json | jq 'length'

# First element
echo ""
echo "--- First movie ---"
cat movies.json | jq '.[0]'

# All titles (iterate with .[])
echo ""
echo "--- All titles (first 5) ---"
cat movies.json | jq '.[].Title' | head -5

# =============================================================================
# PART 4: Building New Objects
# =============================================================================

echo ""
echo "=== Build New Objects ==="
echo ""

# Transform structure
echo "--- Renamed fields (first 3) ---"
cat movies.json | jq '.[:3] | .[] | {name: .Title, year: .Year, rating: .imdbRating}'

# Collect into array
echo ""
echo "--- As array (first 3) ---"
cat movies.json | jq '[.[:3][] | {name: .Title, year: .Year}]'

# =============================================================================
# PART 5: Filtering with select()
# =============================================================================

echo ""
echo "=== Filtering ==="
echo ""

# Find movies with N/A year
echo "--- Movies with N/A year ---"
cat movies.json | jq '.[] | select(.Year == "N/A") | .Title'

# Find movies with N/A BoxOffice
echo ""
echo "--- Movies with N/A BoxOffice ---"
cat movies.json | jq '.[] | select(.BoxOffice == "N/A") | .Title'

# Find movies with null title
echo ""
echo "--- Movies with null title ---"
cat movies.json | jq '.[] | select(.Title == null or .Title == "")'

# =============================================================================
# PART 6: Type Conversion
# =============================================================================

echo ""
echo "=== Type Conversion ==="
echo ""

# String to number (for valid years)
echo "--- Convert year to number ---"
echo '{"Year": "2010"}' | jq '.Year | tonumber'

# Handle N/A gracefully
echo ""
echo "--- Safe year extraction (first 5 valid) ---"
cat movies.json | jq '[.[] | select(.Year != "N/A" and .Year != null) | {title: .Title, year: (.Year | tonumber)}] | .[:5]'

# =============================================================================
# PART 7: Handling Missing Data
# =============================================================================

echo ""
echo "=== Missing Data ==="
echo ""

# Default value with //
echo "--- Default for missing fields ---"
echo '{"title": "Test"}' | jq '.rating // "N/A"'

# Check if field exists
echo ""
echo "--- Check if field exists ---"
cat movie.json | jq 'has("BoxOffice")'
cat movie.json | jq 'has("Budget")'

# Filter out nulls
echo ""
echo "--- Count non-null ratings ---"
cat movies.json | jq '[.[] | select(.imdbRating != null and .imdbRating != "N/A")] | length'

# =============================================================================
# PART 8: Aggregation
# =============================================================================

echo ""
echo "=== Aggregation ==="
echo ""

# Count
echo "--- Total movies ---"
cat movies.json | jq 'length'

# Unique values
echo ""
echo "--- Unique Rated values ---"
cat movies.json | jq '[.[].Rated] | unique'

# Group by
echo ""
echo "--- Count by Rated (simplified) ---"
cat movies.json | jq 'group_by(.Rated) | map({rated: .[0].Rated, count: length})'

# =============================================================================
# PART 9: Sorting
# =============================================================================

echo ""
echo "=== Sorting ==="
echo ""

# Sort by field
echo "--- Sort by Year (first 5 titles) ---"
cat movies.json | jq '[.[] | select(.Year != "N/A")] | sort_by(.Year) | .[:5] | .[].Title'

# Reverse sort
echo ""
echo "--- Top 5 by Year (newest) ---"
cat movies.json | jq '[.[] | select(.Year != "N/A")] | sort_by(.Year) | reverse | .[:5] | .[] | "\(.Title) (\(.Year))"'

# =============================================================================
# PART 10: Raw Output and CSV
# =============================================================================

echo ""
echo "=== Raw Output ==="
echo ""

# Without quotes
echo "--- Raw strings ---"
cat movies.json | jq -r '.[0:3][].Title'

# CSV format
echo ""
echo "--- CSV output (first 5) ---"
cat movies.json | jq -r '.[:5][] | [.Title, .Year, .imdbRating] | @csv'

# TSV format
echo ""
echo "--- TSV output (first 3) ---"
cat movies.json | jq -r '.[:3][] | [.Title, .Year] | @tsv'

# =============================================================================
# PART 11: Data Validation Examples
# =============================================================================

echo ""
echo "=== Data Validation ==="
echo ""

# Find duplicate titles
echo "--- Duplicate titles ---"
cat movies.json | jq 'group_by(.Title) | map(select(length > 1)) | .[].Title' | sort | uniq

# Find invalid years (not string, or future)
echo ""
echo "--- Suspicious years ---"
cat movies.json | jq '.[] | select(.Year == "N/A" or .Year == "-500" or (.Year | tonumber? // 0) > 2024) | {title: .Title, year: .Year}'

# Summary statistics
echo ""
echo "--- Data Summary ---"
cat movies.json | jq '{
  total: length,
  null_titles: [.[] | select(.Title == null or .Title == "")] | length,
  na_years: [.[] | select(.Year == "N/A")] | length,
  na_boxoffice: [.[] | select(.BoxOffice == "N/A")] | length
}'
