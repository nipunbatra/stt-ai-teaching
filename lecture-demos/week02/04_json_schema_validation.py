#!/usr/bin/env python3
"""
Week 02: JSON Schema Validation
Demonstrates validating data against JSON Schema
"""

import json
from pathlib import Path

# Install: pip install jsonschema
from jsonschema import validate, ValidationError, Draft202012Validator

DATA_DIR = Path(__file__).parent / "data"

# =============================================================================
# PART 1: Load Schema and Data
# =============================================================================

print("=== Loading Schema and Data ===\n")

# Load the schema
with open(DATA_DIR / "movie_schema.json") as f:
    schema = json.load(f)

print("Schema loaded:")
print(json.dumps(schema, indent=2)[:500] + "...\n")

# =============================================================================
# PART 2: Valid Movie Example
# =============================================================================

print("=== Valid Movie ===\n")

valid_movie = {
    "title": "Inception",
    "year": 2010,
    "rating": 8.8,
    "revenue": 292576195,
    "runtime_minutes": 148,
    "genres": ["Action", "Sci-Fi", "Thriller"],
    "rated": "PG-13"
}

try:
    validate(instance=valid_movie, schema=schema)
    print("Valid movie:")
    print(json.dumps(valid_movie, indent=2))
    print("\nValidation: PASSED")
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# =============================================================================
# PART 3: Invalid Movie Examples
# =============================================================================

print("\n=== Invalid Movie Examples ===\n")

# Missing required field
print("--- Missing title (required) ---")
invalid_movie_1 = {
    "year": 2010,
    "genres": ["Action"]
}
try:
    validate(instance=invalid_movie_1, schema=schema)
    print("Validation: PASSED")
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# Wrong type
print("\n--- Year as string (should be integer) ---")
invalid_movie_2 = {
    "title": "Test",
    "year": "2010",  # String, not integer!
    "genres": ["Drama"]
}
try:
    validate(instance=invalid_movie_2, schema=schema)
    print("Validation: PASSED")
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# Out of range
print("\n--- Year before 1880 ---")
invalid_movie_3 = {
    "title": "Ancient Film",
    "year": 1500,  # Before cinema existed
    "genres": ["Drama"]
}
try:
    validate(instance=invalid_movie_3, schema=schema)
    print("Validation: PASSED")
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# Invalid enum value
print("\n--- Invalid rating (not in enum) ---")
invalid_movie_4 = {
    "title": "Test",
    "year": 2020,
    "genres": ["Action"],
    "rated": "XX"  # Not in enum
}
try:
    validate(instance=invalid_movie_4, schema=schema)
    print("Validation: PASSED")
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# Empty array (minItems violation)
print("\n--- Empty genres array ---")
invalid_movie_5 = {
    "title": "Test",
    "year": 2020,
    "genres": []  # minItems: 1
}
try:
    validate(instance=invalid_movie_5, schema=schema)
    print("Validation: PASSED")
except ValidationError as e:
    print(f"Validation failed: {e.message}")

# =============================================================================
# PART 4: Validate Batch of Movies
# =============================================================================

print("\n=== Batch Validation ===\n")

# Sample batch - mix of valid and invalid
movies_batch = [
    {"title": "Inception", "year": 2010, "genres": ["Action", "Sci-Fi"]},
    {"title": "Avatar", "year": 2009, "genres": ["Action"]},
    {"title": "", "year": 2020, "genres": ["Drama"]},  # Empty title
    {"title": "Future Film", "year": 2050, "genres": ["Sci-Fi"]},  # Future year
    {"title": "The Matrix", "year": 1999, "genres": ["Action", "Sci-Fi"]},
]

valid_movies = []
invalid_movies = []

for i, movie in enumerate(movies_batch):
    try:
        validate(instance=movie, schema=schema)
        valid_movies.append(movie)
    except ValidationError as e:
        invalid_movies.append({
            "index": i,
            "movie": movie,
            "error": e.message
        })

print(f"Valid: {len(valid_movies)}")
print(f"Invalid: {len(invalid_movies)}")
print("\nInvalid movies:")
for item in invalid_movies:
    print(f"  [{item['index']}] {item['movie'].get('title', 'N/A')}: {item['error']}")

# =============================================================================
# PART 5: Collect All Errors (not just first)
# =============================================================================

print("\n=== Collect All Errors ===\n")

# Movie with multiple issues
bad_movie = {
    "title": "",          # Empty (minLength: 1)
    "year": 1500,         # Too old (minimum: 1880)
    "rating": 15.0,       # Too high (maximum: 10)
    "genres": [],         # Empty (minItems: 1)
    "rated": "INVALID"    # Not in enum
}

validator = Draft202012Validator(schema)
errors = list(validator.iter_errors(bad_movie))

print(f"Movie: {bad_movie}")
print(f"\nFound {len(errors)} errors:")
for error in errors:
    print(f"  - {error.json_path}: {error.message}")

# =============================================================================
# PART 6: Schema from Python dict
# =============================================================================

print("\n=== Define Schema Inline ===\n")

# You can define schema directly in Python
inline_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "year": {"type": "integer", "minimum": 1888, "maximum": 2030},
        "rating": {"type": ["number", "null"], "minimum": 0, "maximum": 10}
    },
    "required": ["title", "year"]
}

test_movie = {"title": "Test Film", "year": 2020, "rating": 7.5}

try:
    validate(instance=test_movie, schema=inline_schema)
    print("Inline schema validation: PASSED")
except ValidationError as e:
    print(f"Failed: {e.message}")
