#!/bin/bash
# Week 02: Quick Data Inspection Script
# Run this to get a summary of data quality issues

DATA_DIR="$(dirname "$0")/data"
cd "$DATA_DIR" || exit 1

echo "=============================================="
echo "      DATA INSPECTION REPORT"
echo "=============================================="
echo ""

# Check if files exist
if [[ ! -f movies.csv ]]; then
    echo "Error: movies.csv not found in $DATA_DIR"
    exit 1
fi

echo "=== File Information ==="
file movies.csv
file movies.json 2>/dev/null || echo "movies.json: not found"
ls -lh movies.csv movies.json 2>/dev/null
echo ""

echo "=== CSV Shape ==="
ROWS=$(wc -l < movies.csv)
COLS=$(head -1 movies.csv | tr ',' '\n' | wc -l)
echo "Rows: $((ROWS - 1)) (excluding header)"
echo "Columns: $COLS"
echo ""

echo "=== Column Names ==="
head -1 movies.csv | tr ',' '\n' | nl
echo ""

echo "=== First 5 Rows ==="
head -6 movies.csv
echo ""

echo "=== Last 3 Rows ==="
tail -3 movies.csv
echo ""

echo "=== Data Quality Issues ==="
echo ""

# N/A values
NA_COUNT=$(grep -c "N/A" movies.csv)
echo "N/A string values: $NA_COUNT"

# Empty fields (consecutive commas)
EMPTY_COUNT=$(grep -c ",," movies.csv)
echo "Rows with empty fields (,,): $EMPTY_COUNT"

# Duplicate rows
DUP_ROWS=$(sort movies.csv | uniq -d | wc -l)
echo "Duplicate rows: $DUP_ROWS"

# Duplicate titles
echo ""
echo "Duplicate titles:"
cut -d',' -f1 movies.csv | tail -n +2 | sort | uniq -d | head -10
echo ""

# Sample N/A entries
echo "Sample rows with N/A (first 5):"
grep "N/A" movies.csv | head -5
echo ""

echo "=============================================="
echo "      JSON Data (if available)"
echo "=============================================="

if [[ -f movies.json ]] && command -v jq &> /dev/null; then
    echo ""
    echo "=== JSON Summary ==="
    cat movies.json | jq '{
        total_records: length,
        null_titles: [.[] | select(.Title == null or .Title == "")] | length,
        na_years: [.[] | select(.Year == "N/A")] | length,
        na_boxoffice: [.[] | select(.BoxOffice == "N/A")] | length,
        unique_rated_values: ([.[].Rated] | unique)
    }'
elif [[ ! -f movies.json ]]; then
    echo "movies.json not found"
else
    echo "jq not installed - skipping JSON analysis"
    echo "Install with: brew install jq"
fi

echo ""
echo "=============================================="
echo "      RECOMMENDATIONS"
echo "=============================================="
echo ""
echo "1. Handle N/A values: Convert to NULL or appropriate defaults"
echo "2. Fix duplicates: Remove exact duplicates, investigate near-duplicates"
echo "3. Validate types: Ensure Year/Rating are numeric"
echo "4. Clean formats: Parse 'Runtime' ('148 min' -> 148)"
echo "5. Standardize BoxOffice: Remove '$' and commas"
echo ""
