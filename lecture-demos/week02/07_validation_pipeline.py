#!/usr/bin/env python3
"""
Week 02: Complete Data Validation Pipeline
Putting it all together
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any
from pydantic import BaseModel, Field, field_validator, ValidationError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# =============================================================================
# PART 1: Define Clean Schema (Pydantic)
# =============================================================================

class CleanMovie(BaseModel):
    """Validated and cleaned movie schema."""
    title: str = Field(..., min_length=1)
    year: int = Field(..., ge=1888, le=2030)
    rating: Optional[float] = Field(None, ge=0, le=10)
    revenue: Optional[int] = Field(None, ge=0)
    runtime_minutes: Optional[int] = Field(None, ge=1, le=1000)
    genres: List[str] = Field(default_factory=list)
    rated: Optional[str] = None

    @field_validator('title')
    @classmethod
    def clean_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

# =============================================================================
# PART 2: Cleaning Functions
# =============================================================================

def clean_year(value: Any) -> Optional[int]:
    """Convert year string to integer, return None if invalid."""
    if value is None or value == '' or value == 'N/A':
        return None
    try:
        year = int(value)
        if 1888 <= year <= 2030:
            return year
        return None  # Out of range
    except (ValueError, TypeError):
        return None


def clean_rating(value: Any) -> Optional[float]:
    """Convert rating to float, return None if invalid."""
    if value is None or value == '' or value == 'N/A' or value == 'invalid':
        return None
    try:
        rating = float(value)
        if 0 <= rating <= 10:
            return rating
        return None  # Out of range
    except (ValueError, TypeError):
        return None


def clean_revenue(value: Any) -> Optional[int]:
    """Convert '$292,576,195' to integer."""
    if value is None or value == '' or value == 'N/A':
        return None
    try:
        # Remove $ and commas
        cleaned = str(value).replace('$', '').replace(',', '')
        revenue = int(cleaned)
        if revenue >= 0:
            return revenue
        return None  # Negative
    except (ValueError, TypeError):
        return None


def clean_runtime(value: Any) -> Optional[int]:
    """Convert '148 min' to integer 148."""
    if value is None or value == '' or value == 'N/A':
        return None
    try:
        match = re.search(r'(\d+)', str(value))
        if match:
            runtime = int(match.group(1))
            if 1 <= runtime <= 1000:
                return runtime
        return None
    except (ValueError, TypeError):
        return None


def clean_genres(value: Any) -> List[str]:
    """Convert 'Action, Drama, Sci-Fi' to ['Action', 'Drama', 'Sci-Fi']."""
    if value is None or value == '' or value == 'N/A':
        return []
    if isinstance(value, list):
        return value
    return [g.strip() for g in str(value).split(',') if g.strip()]


# =============================================================================
# PART 3: Transform Raw to Clean
# =============================================================================

def transform_movie(raw: dict) -> dict:
    """Transform raw API data to clean format."""
    return {
        'title': raw.get('Title', raw.get('title', '')),
        'year': clean_year(raw.get('Year', raw.get('year'))),
        'rating': clean_rating(raw.get('imdbRating', raw.get('rating'))),
        'revenue': clean_revenue(raw.get('BoxOffice', raw.get('boxoffice'))),
        'runtime_minutes': clean_runtime(raw.get('Runtime', raw.get('runtime'))),
        'genres': clean_genres(raw.get('Genre', raw.get('genre'))),
        'rated': raw.get('Rated', raw.get('rated')) if raw.get('Rated', raw.get('rated')) not in ['N/A', ''] else None
    }


# =============================================================================
# PART 4: Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """Complete data validation pipeline."""

    def __init__(self):
        self.valid_movies: List[CleanMovie] = []
        self.invalid_movies: List[dict] = []
        self.stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates_removed': 0
        }

    def load_json(self, filepath: Path) -> List[dict]:
        """Load raw data from JSON file."""
        logger.info(f"Loading data from {filepath}")
        with open(filepath) as f:
            data = json.load(f)
        self.stats['total'] = len(data)
        logger.info(f"Loaded {len(data)} records")
        return data

    def remove_duplicates(self, data: List[dict]) -> List[dict]:
        """Remove exact duplicate records."""
        seen = set()
        unique = []
        for item in data:
            # Create a hashable key from the dict
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(item)

        removed = len(data) - len(unique)
        self.stats['duplicates_removed'] = removed
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")
        return unique

    def validate_batch(self, data: List[dict]) -> Tuple[List[CleanMovie], List[dict]]:
        """Validate and transform a batch of records."""
        valid = []
        invalid = []

        for i, raw in enumerate(data):
            try:
                # Transform first
                cleaned = transform_movie(raw)

                # Skip if year couldn't be parsed (required field)
                if cleaned['year'] is None:
                    raise ValueError("Year is required and must be valid")

                # Validate with Pydantic
                movie = CleanMovie(**cleaned)
                valid.append(movie)

            except (ValidationError, ValueError) as e:
                error_msg = str(e) if isinstance(e, ValueError) else '; '.join(
                    f"{err['loc'][0]}: {err['msg']}" for err in e.errors()
                )
                invalid.append({
                    'index': i,
                    'raw_data': raw,
                    'error': error_msg
                })

        return valid, invalid

    def run(self, filepath: Path) -> dict:
        """Run the complete pipeline."""
        logger.info("=" * 50)
        logger.info("Starting Validation Pipeline")
        logger.info("=" * 50)

        # Step 1: Load
        data = self.load_json(filepath)

        # Step 2: Remove duplicates
        data = self.remove_duplicates(data)

        # Step 3: Validate
        logger.info("Validating records...")
        self.valid_movies, self.invalid_movies = self.validate_batch(data)

        self.stats['valid'] = len(self.valid_movies)
        self.stats['invalid'] = len(self.invalid_movies)

        logger.info(f"Valid: {self.stats['valid']}")
        logger.info(f"Invalid: {self.stats['invalid']}")

        return self.stats

    def export_valid(self, filepath: Path):
        """Export valid movies to JSON."""
        data = [m.model_dump() for m in self.valid_movies]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} valid movies to {filepath}")

    def export_invalid(self, filepath: Path):
        """Export invalid movies with errors to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.invalid_movies, f, indent=2)
        logger.info(f"Exported {len(self.invalid_movies)} invalid records to {filepath}")

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 50)
        print("VALIDATION REPORT")
        print("=" * 50)
        print(f"Total records:      {self.stats['total']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"Valid records:      {self.stats['valid']}")
        print(f"Invalid records:    {self.stats['invalid']}")

        valid_pct = (self.stats['valid'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        print(f"Validation rate:    {valid_pct:.1f}%")

        if self.invalid_movies:
            print("\nSample invalid records:")
            for item in self.invalid_movies[:5]:
                title = item['raw_data'].get('Title', item['raw_data'].get('title', 'N/A'))
                print(f"  - {title}: {item['error'][:60]}...")
        print("=" * 50)


# =============================================================================
# PART 5: Run Pipeline
# =============================================================================

if __name__ == "__main__":
    # Create pipeline
    pipeline = ValidationPipeline()

    # Run on sample data
    input_file = DATA_DIR / "movies.json"

    if input_file.exists():
        pipeline.run(input_file)
        pipeline.print_report()

        # Export results
        output_dir = DATA_DIR / "output"
        output_dir.mkdir(exist_ok=True)

        pipeline.export_valid(output_dir / "movies_valid.json")
        pipeline.export_invalid(output_dir / "movies_invalid.json")

        print(f"\nOutput files written to: {output_dir}")
    else:
        print(f"Input file not found: {input_file}")
        print("Please run from the lecture-demos/week02 directory")
