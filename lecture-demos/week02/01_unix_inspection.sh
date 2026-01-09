#!/bin/bash
# Week 02: Unix Data Inspection Tools
# Run sections separately to demonstrate in lecture

DATA_DIR="$(dirname "$0")/data"
cd "$DATA_DIR" || exit 1

# =============================================================================
# PART 1: file - What kind of file is this?
# =============================================================================

echo "=== The 'file' command ==="
echo ""

file movies.csv
file movies.json
file movie.json

# Check encoding specifically
echo ""
echo "--- With MIME type ---"
file -i movies.csv

# =============================================================================
# PART 2: wc - Word count (lines, words, bytes)
# =============================================================================

echo ""
echo "=== The 'wc' command ==="
echo ""

# Full output
wc movies.csv

# Just lines (most common)
echo ""
echo "--- Line count only ---"
wc -l movies.csv
wc -l movies.json

# =============================================================================
# PART 3: head - See first N lines
# =============================================================================

echo ""
echo "=== The 'head' command ==="
echo ""

# Default (10 lines)
echo "--- First 5 lines of CSV ---"
head -5 movies.csv

echo ""
echo "--- First 3 lines of JSON ---"
head -3 movies.json

# =============================================================================
# PART 4: tail - See last N lines
# =============================================================================

echo ""
echo "=== The 'tail' command ==="
echo ""

# Last 5 lines
echo "--- Last 5 lines ---"
tail -5 movies.csv

# Skip header (everything except first line)
echo ""
echo "--- Skip header (first 3 data rows) ---"
tail -n +2 movies.csv | head -3

# =============================================================================
# PART 5: sort - Sort lines
# =============================================================================

echo ""
echo "=== The 'sort' command ==="
echo ""

# Sort CSV by title (column 1)
echo "--- Sort by title (first 5) ---"
tail -n +2 movies.csv | sort -t',' -k1 | head -5

# Sort by year (column 2, numeric)
echo ""
echo "--- Sort by year descending (first 5) ---"
tail -n +2 movies.csv | sort -t',' -k2 -nr | head -5

# =============================================================================
# PART 6: uniq - Find/remove duplicates
# =============================================================================

echo ""
echo "=== The 'uniq' command ==="
echo ""

# Find duplicate titles
echo "--- Duplicate titles ---"
cut -d',' -f1 movies.csv | sort | uniq -d

# Count occurrences of each title
echo ""
echo "--- Title counts (top 5) ---"
cut -d',' -f1 movies.csv | sort | uniq -c | sort -rn | head -5

# =============================================================================
# PART 7: cut - Extract columns
# =============================================================================

echo ""
echo "=== The 'cut' command ==="
echo ""

# Get first column (titles)
echo "--- Just titles (first 5) ---"
cut -d',' -f1 movies.csv | head -5

# Get columns 1 and 4 (title and rating)
echo ""
echo "--- Title and rating (first 5) ---"
cut -d',' -f1,4 movies.csv | head -5

# =============================================================================
# PART 8: grep - Search patterns
# =============================================================================

echo ""
echo "=== The 'grep' command ==="
echo ""

# Find rows containing "N/A"
echo "--- Count N/A values ---"
grep -c "N/A" movies.csv

# Find specific title
echo ""
echo "--- Find Inception ---"
grep "Inception" movies.csv

# Show line numbers
echo ""
echo "--- N/A with line numbers (first 5) ---"
grep -n "N/A" movies.csv | head -5

# Case insensitive search
echo ""
echo "--- Case insensitive 'matrix' ---"
grep -i "matrix" movies.csv

# =============================================================================
# PART 9: Combining commands - Pipeline
# =============================================================================

echo ""
echo "=== Combined Pipeline ==="
echo ""

# Find all unique genres
echo "--- Unique rated values ---"
cut -d',' -f7 movies.csv | tail -n +2 | sort | uniq

# Find movies with missing boxoffice
echo ""
echo "--- Movies with N/A boxoffice (first 5) ---"
grep "N/A" movies.csv | cut -d',' -f1 | head -5

# Count empty fields
echo ""
echo "--- Rows with empty fields ---"
grep -c ",," movies.csv
