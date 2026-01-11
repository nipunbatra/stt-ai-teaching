# Week 02: Data Validation - Demo Guide

## Prerequisites & Installation

### Google Colab (Recommended for class)
```python
# Run this cell first to install everything
!sudo apt-get install -y jq
!pip install csvkit pydantic jsonschema pandas

# Verify
!jq --version
!csvlook --version
```

Then use `!` prefix for shell commands:
```python
!head -5 movies.csv
!cat movies.json | jq '.[0]'
!csvlook movies.csv | head -10
```

Or open Colab terminal (Tools > Terminal) and run commands directly.

---

### macOS (with Homebrew)
```bash
# jq - JSON processor
brew install jq

# csvkit - CSV tools
pip install csvkit

# Or with uv
uv pip install csvkit

# Python packages for demos
pip install pydantic jsonschema pandas
# Or
uv pip install pydantic jsonschema pandas
```

### Verify Installation
```bash
jq --version          # Should show: jq-1.6 or similar
csvlook --version     # Should show: csvlook x.x.x
python -c "import pydantic; print(pydantic.__version__)"
```

---

## Directory Structure

```
lecture-demos/week02/
├── DEMO_GUIDE.md              # This file
├── 01_unix_inspection.sh      # Unix CLI tools
├── 02_jq_basics.sh            # jq JSON processing
├── 03_csvkit_demo.sh          # csvkit CSV tools
├── 04_json_schema_validation.py  # JSON Schema
├── 05_pydantic_basics.py      # Pydantic validation
├── 06_data_profiling.py       # Pandas profiling
├── 07_validation_pipeline.py  # Complete pipeline
└── data/
    ├── movies.csv             # CSV with quality issues
    ├── movies.json            # JSON array of movies
    ├── movie.json             # Single movie (detailed)
    └── movie_schema.json      # JSON Schema definition
```

---

## Demo Order by Slide Reference

### Part 3: First Look at Your Data (Slides 20-30)
**Unix tools for initial inspection**

```bash
cd lecture-demos/week02

# Run individual commands from the script
# Or run the whole script:
bash 01_unix_inspection.sh
```

**Live demos to run:**
```bash
# Slide 23: file command
file data/movies.csv
file data/movies.json

# Slide 24: wc command
wc -l data/movies.csv

# Slide 25-26: head/tail
head -5 data/movies.csv
tail -5 data/movies.csv

# Slide 27-28: sort/uniq
cut -d',' -f1 data/movies.csv | sort | uniq -d

# Slide 29: cut
cut -d',' -f1,4 data/movies.csv | head -5

# Slide 30: grep
grep -c "N/A" data/movies.csv
```

---

### Part 4: jq - JSON Processing (Slides 31-54)
**The Swiss Army knife for JSON**

```bash
cd lecture-demos/week02

# Run sections of the script
bash 02_jq_basics.sh
```

**Live demos to run:**
```bash
# Slide 33: Pretty print
cat data/movie.json | jq .

# Slide 34: Extract fields
cat data/movie.json | jq '.Title'
cat data/movie.json | jq '.Title, .Year'

# Slide 35-37: Arrays
cat data/movies.json | jq 'length'
cat data/movies.json | jq '.[0]'
cat data/movies.json | jq '.[].Title' | head -5

# Slide 38: Build objects
cat data/movies.json | jq '.[:3][] | {name: .Title, year: .Year}'

# Slide 39: Filter with select
cat data/movies.json | jq '.[] | select(.Year == "N/A")'
cat data/movies.json | jq '.[] | select(.BoxOffice == "N/A") | .Title'

# Slide 44-45: Aggregation
cat data/movies.json | jq '[.[].Rated] | unique'
cat data/movies.json | jq 'group_by(.Year) | map({year: .[0].Year, count: length}) | .[:5]'

# Slide 47: Raw output / CSV
cat data/movies.json | jq -r '.[:5][] | [.Title, .Year, .imdbRating] | @csv'
```

---

### Part 5: CSVkit (Slides 55-70)
**The CSV Swiss Army Knife**

```bash
cd lecture-demos/week02

# Make sure csvkit is installed
pip install csvkit

# Run demos
bash 03_csvkit_demo.sh
```

**Live demos to run:**
```bash
# Slide 56: csvlook
csvlook data/movies.csv | head -15

# Slide 57-58: csvstat
csvstat data/movies.csv | head -30
csvstat -c rating data/movies.csv

# Slide 59: csvcut
csvcut -n data/movies.csv
csvcut -c title,year data/movies.csv | head -5

# Slide 60: csvgrep
csvgrep -c year -m "2019" data/movies.csv | csvlook

# Slide 61: csvsort
csvsort -c rating -r data/movies.csv | head -10

# Slide 62: csvjson
head -5 data/movies.csv | csvjson | jq '.'

# Slide 63-64: csvsql
csvsql --query "SELECT title, rating FROM movies WHERE rating > 8.5" data/movies.csv | csvlook
csvsql --query "SELECT title, COUNT(*) as cnt FROM movies GROUP BY title HAVING cnt > 1" data/movies.csv | csvlook
```

---

### Part 7: Schema Validation (Slides 75-95)
**Contracts for your data**

```bash
cd lecture-demos/week02

# Run Python demo
python 04_json_schema_validation.py
```

**Show the schema file:**
```bash
cat data/movie_schema.json | jq .
```

---

### Part 8: Pydantic (Slides 96-115)
**Pythonic data validation**

```bash
cd lecture-demos/week02

# Run Python demo
python 05_pydantic_basics.py
```

**Interactive Python demo:**
```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str
    year: int
    rating: float

# Works - coerces strings
movie = Movie(title="Inception", year="2010", rating="8.8")
print(movie.year)  # 2010 (int)

# Fails - invalid year
Movie(title="Test", year="invalid", rating=8.0)
```

---

### Part 6: Data Profiling (Slides 71-74)
**Understanding your data before using it**

```bash
cd lecture-demos/week02

# Run profiling demo
python 06_data_profiling.py
```

---

### Part 11: Complete Pipeline (Slides 125-140)
**Putting it all together**

```bash
cd lecture-demos/week02

# Run the complete pipeline
python 07_validation_pipeline.py

# Check outputs
ls -la data/output/
cat data/output/movies_valid.json | jq 'length'
cat data/output/movies_invalid.json | jq '.[0]'
```

---

## Quick Reference Commands

### Data Quality Checks
```bash
# How many rows?
wc -l data/movies.csv

# How many N/A values?
grep -c "N/A" data/movies.csv

# Duplicate titles?
cut -d',' -f1 data/movies.csv | sort | uniq -d

# Null/empty values in JSON?
cat data/movies.json | jq '[.[] | select(.Title == null or .Title == "")] | length'
```

### jq Cheat Sheet
```bash
# Pretty print
jq .

# Get field
jq '.Title'

# Array length
jq 'length'

# Iterate array
jq '.[]'

# Filter
jq '.[] | select(.Year == "2010")'

# Build object
jq '{name: .Title, year: .Year}'

# Raw output
jq -r '.[].Title'

# CSV
jq -r '.[] | [.Title, .Year] | @csv'
```

### csvkit Cheat Sheet
```bash
csvlook file.csv          # Pretty print
csvstat file.csv          # Statistics
csvcut -c col1,col2       # Select columns
csvcut -n file.csv        # List columns
csvgrep -c col -m "val"   # Filter rows
csvsort -c col -r         # Sort (reverse)
csvjson file.csv          # To JSON
csvsql --query "..."      # SQL query
```

---

## Troubleshooting

### "command not found: jq"
```bash
brew install jq
```

### "command not found: csvlook"
```bash
pip install csvkit
# or
uv pip install csvkit
```

### "ModuleNotFoundError: No module named 'pydantic'"
```bash
pip install pydantic jsonschema pandas
```

### Permission denied on shell scripts
```bash
chmod +x *.sh
```
