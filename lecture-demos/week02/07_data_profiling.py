#!/usr/bin/env python3
"""
Week 02: Data Profiling with Pandas
Systematic analysis of data quality
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# =============================================================================
# PART 1: Load Data
# =============================================================================

print("=== Load Data ===\n")

# Load CSV
df = pd.read_csv(DATA_DIR / "movies.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# =============================================================================
# PART 2: Basic Shape
# =============================================================================

print("\n=== Basic Shape ===\n")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# =============================================================================
# PART 3: Data Types
# =============================================================================

print("\n=== Data Types ===\n")

print(df.dtypes)

# Note: All columns are 'object' (string) - need cleaning!

# =============================================================================
# PART 4: Missing Values
# =============================================================================

print("\n=== Missing Values ===\n")

# Pandas null
null_counts = df.isnull().sum()
print("Null values per column:")
print(null_counts[null_counts > 0])

# String "N/A" (not detected as null!)
print("\n'N/A' string values per column:")
for col in df.columns:
    na_count = (df[col] == 'N/A').sum()
    if na_count > 0:
        print(f"  {col}: {na_count}")

# Empty strings
print("\nEmpty string values per column:")
for col in df.columns:
    empty_count = (df[col] == '').sum()
    if empty_count > 0:
        print(f"  {col}: {empty_count}")

# =============================================================================
# PART 5: Unique Values
# =============================================================================

print("\n=== Unique Values ===\n")

print("Unique values per column:")
for col in df.columns:
    unique = df[col].nunique()
    print(f"  {col}: {unique}")

# =============================================================================
# PART 6: Duplicates
# =============================================================================

print("\n=== Duplicates ===\n")

# Exact duplicates
dup_count = df.duplicated().sum()
print(f"Exact duplicate rows: {dup_count}")

# Duplicate titles
title_counts = df['title'].value_counts()
duplicate_titles = title_counts[title_counts > 1]
print(f"\nDuplicate titles:")
for title, count in duplicate_titles.items():
    print(f"  '{title}': {count} occurrences")

# =============================================================================
# PART 7: Value Ranges
# =============================================================================

print("\n=== Value Ranges ===\n")

# Year analysis (need to handle N/A first)
years = pd.to_numeric(df['year'], errors='coerce')
print(f"Year range: {years.min():.0f} - {years.max():.0f}")
print(f"Year mean: {years.mean():.1f}")

# Rating analysis
ratings = pd.to_numeric(df['rating'], errors='coerce')
valid_ratings = ratings.dropna()
print(f"\nRating range: {valid_ratings.min():.1f} - {valid_ratings.max():.1f}")
print(f"Rating mean: {valid_ratings.mean():.2f}")
print(f"Rating median: {valid_ratings.median():.1f}")

# =============================================================================
# PART 8: Pattern Detection
# =============================================================================

print("\n=== Pattern Detection ===\n")

# Unique 'rated' values
print("Unique 'rated' values:")
print(df['rated'].value_counts())

# Runtime format
print("\nRuntime formats (sample):")
print(df['runtime'].head(10).tolist())

# BoxOffice format
print("\nBoxOffice formats (sample):")
print(df['boxoffice'].head(10).tolist())

# =============================================================================
# PART 9: Comprehensive Profile Function
# =============================================================================

print("\n=== Comprehensive Profile ===\n")

def profile_dataframe(df):
    """Generate comprehensive data quality profile."""
    profile = {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "columns": {}
    }

    for col in df.columns:
        col_profile = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }

        # String-specific checks
        if df[col].dtype == 'object':
            col_profile["na_string_count"] = int((df[col] == 'N/A').sum())
            col_profile["empty_string_count"] = int((df[col] == '').sum())

            # Sample values
            col_profile["sample_values"] = df[col].dropna().head(3).tolist()

        # Numeric-specific checks (if convertible)
        numeric = pd.to_numeric(df[col], errors='coerce')
        valid_numeric = numeric.dropna()
        if len(valid_numeric) > 0:
            col_profile["numeric_valid_count"] = int(len(valid_numeric))
            col_profile["numeric_min"] = float(valid_numeric.min())
            col_profile["numeric_max"] = float(valid_numeric.max())
            col_profile["numeric_mean"] = float(valid_numeric.mean())

        profile["columns"][col] = col_profile

    # Duplicate analysis
    profile["duplicates"] = {
        "exact_duplicates": int(df.duplicated().sum()),
        "title_duplicates": int(len(df) - df['title'].nunique())
    }

    return profile

profile = profile_dataframe(df)
print(json.dumps(profile, indent=2))

# =============================================================================
# PART 10: Data Quality Score
# =============================================================================

print("\n=== Data Quality Score ===\n")

def calculate_quality_score(df):
    """Calculate a simple data quality score 0-100."""
    scores = []

    # Completeness (no nulls)
    null_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
    completeness = (1 - null_rate) * 100
    scores.append(("Completeness", completeness))

    # No N/A strings
    na_strings = sum((df[col] == 'N/A').sum() for col in df.columns)
    na_rate = na_strings / (len(df) * len(df.columns))
    na_score = (1 - na_rate) * 100
    scores.append(("No N/A strings", na_score))

    # No duplicates
    dup_rate = df.duplicated().sum() / len(df)
    uniqueness = (1 - dup_rate) * 100
    scores.append(("Uniqueness", uniqueness))

    # No empty titles
    empty_titles = (df['title'] == '').sum() + df['title'].isnull().sum()
    title_score = (1 - empty_titles / len(df)) * 100
    scores.append(("Valid titles", title_score))

    print("Quality Scores:")
    for name, score in scores:
        print(f"  {name}: {score:.1f}%")

    overall = sum(s for _, s in scores) / len(scores)
    print(f"\nOverall Score: {overall:.1f}%")

    return overall

score = calculate_quality_score(df)
