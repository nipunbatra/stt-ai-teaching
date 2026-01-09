DATA_DIR="$(dirname "$0")/data"
cd "$DATA_DIR" || exit 1


# How many rows and columns?
head -1 movies.csv | tr ',' '\n' | wc -l

# rows (including header)
wc -l movies.csv 

# Or with csvstat
csvstat --count movies.csv

# =============================================================================
# PART 2: Column Types
# =============================================================================
csvstat movies.csv 2>&1 | grep "Type of data"

# =============================================================================
# PART 3: Count nulls per column
# =============================================================================
csvstat movies.csv 2>&1 | grep -A1 "Contains null"

# =============================================================================
# PART 4: Unique Values
# =============================================================================

csvstat movies.csv 2>&1 | grep "Unique values"
csvstat movies.csv 2>&1 | grep -A5 "Most common values"


# =============================================================================
# PART 5: Value Ranges
# =============================================================================

csvstat -c year movies.csv
 # Find extremes
csvsort -c year movies.csv | head -5 # oldest
csvsort -c year -r movies.csv | head -5 # newest


# =============================================================================
# PART 6: Pattern Detection
# =============================================================================
csvcut -c rating movies.csv | sort | uniq -c | sort -rn | head