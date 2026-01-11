#!/bin/bash
# Week 02: CSVkit - CSV Swiss Army Knife
# Run sections separately to demonstrate in lecture
# Install: pip install csvkit

DATA_DIR="$(dirname "$0")/data"
cd "$DATA_DIR" || exit 1

# =============================================================================
# PART 1: csvlook - Pretty Print CSV
# =============================================================================

echo "=== csvlook: Pretty Print ==="
echo ""

csvlook movies.csv | head -15

# =============================================================================
# PART 2: csvstat - Data Profiling
# =============================================================================

echo ""
echo "=== csvstat: Full Profile ==="
echo ""

csvstat movies.csv 2>&1 | head -50

echo ""
echo "--- Specific column ---"
csvstat -c title movies.csv

echo ""
echo "--- Just counts ---"
csvstat --count movies.csv

# =============================================================================
# PART 3: csvcut - Select Columns
# =============================================================================

echo ""
echo "=== csvcut: Select Columns ==="
echo ""

# List column names
echo "--- Column names ---"
csvcut -n movies.csv

# Select by name
echo ""
echo "--- Title and year (first 5) ---"
csvcut -c title,year movies.csv | head -6

# Select by number
echo ""
echo "--- Columns 1,3,4 (first 5) ---"
csvcut -c 1,3,4 movies.csv | head -6

# Exclude columns
echo ""
echo "--- Exclude boxoffice ---"
csvcut -C boxoffice movies.csv | head -4

# =============================================================================
# PART 4: csvgrep - Filter Rows
# =============================================================================

echo ""
echo "=== csvgrep: Filter Rows ==="
echo ""

# Exact match
echo "--- Year = 2019 ---"
csvgrep -c year -m "2019" movies.csv | csvlook

# Regex pattern
echo ""
echo "--- Titles starting with 'The' ---"
csvgrep -c title -r "^The" movies.csv | csvcut -c title | head -10

# Inverse match (NOT)
echo ""
echo "--- Rows without N/A in boxoffice (count) ---"
csvgrep -c boxoffice -m "N/A" -i movies.csv | wc -l

# Find empty values
echo ""
echo "--- Rows with N/A rating ---"
csvgrep -c rating -r "^N/A$" movies.csv | csvlook

# =============================================================================
# PART 5: csvsort - Sort Data
# =============================================================================

echo ""
echo "=== csvsort: Sort ==="
echo ""

# Sort by column
echo "--- Sort by year (first 5) ---"
csvsort -c year movies.csv | head -6

# Sort descending
echo ""
echo "--- Sort by rating descending (first 5) ---"
csvsort -c rating -r movies.csv | head -6

# Sort by multiple columns
echo ""
echo "--- Sort by year, then rating ---"
csvsort -c year,rating movies.csv | head -10

# =============================================================================
# PART 6: csvjson - Convert to JSON
# =============================================================================

echo ""
echo "=== csvjson: CSV to JSON ==="
echo ""

echo "--- First 3 rows as JSON ---"
head -4 movies.csv | csvjson | jq '.'

# With indentation
echo ""
echo "--- Indented output ---"
head -3 movies.csv | csvjson -i 2

# =============================================================================
# PART 7: csvsql - Query with SQL
# =============================================================================

echo ""
echo "=== csvsql: SQL Queries ==="
echo ""

# Basic select
echo "--- SELECT title, rating WHERE rating > 8.5 ---"
csvsql --query "SELECT title, rating FROM movies WHERE rating > 8.5 ORDER BY rating DESC" movies.csv | csvlook

# Find duplicates
echo ""
echo "--- Duplicate titles ---"
csvsql --query "SELECT title, COUNT(*) as count FROM movies GROUP BY title HAVING count > 1" movies.csv | csvlook

# Aggregate stats
echo ""
echo "--- Movies per year (sample) ---"
csvsql --query "SELECT year, COUNT(*) as count FROM movies GROUP BY year ORDER BY count DESC LIMIT 10" movies.csv | csvlook

# Missing value analysis
echo ""
echo "--- Count N/A boxoffice by year ---"
csvsql --query "SELECT year, COUNT(*) as missing FROM movies WHERE boxoffice = 'N/A' GROUP BY year ORDER BY missing DESC LIMIT 5" movies.csv | csvlook

# =============================================================================
# PART 8: csvclean - Fix Issues
# =============================================================================

echo ""
echo "=== csvclean: Check for Issues ==="
echo ""

# Dry run - check for problems
echo "--- Check for structural issues ---"
csvclean -n movies.csv 2>&1 || echo "(no issues found)"

# =============================================================================
# PART 9: Combined Pipeline
# =============================================================================

echo ""
echo "=== Combined Pipeline ==="
echo ""

echo "--- Top rated movies by genre (sample) ---"
csvcut -c title,rating,genre movies.csv \
  | csvgrep -c rating -r "^[0-9]" \
  | csvsort -c rating -r \
  | head -10 \
  | csvlook

echo ""
echo "--- Data quality summary ---"
echo "Total rows: $(csvstat --count movies.csv)"
echo "Unique titles: $(csvcut -c title movies.csv | tail -n +2 | sort -u | wc -l)"
echo "N/A in boxoffice: $(csvgrep -c boxoffice -m "N/A" movies.csv | wc -l)"
echo "N/A in rating: $(csvgrep -c rating -m "N/A" movies.csv | wc -l)"
