#!/usr/bin/env python3
"""
Week 02: Pydantic Data Validation
Pythonic data validation with type hints
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ValidationError
import json

# =============================================================================
# PART 1: Basic Pydantic Model
# =============================================================================

print("=== Basic Pydantic Model ===\n")

class Movie(BaseModel):
    title: str
    year: int
    rating: float

# Create from kwargs
movie = Movie(title="Inception", year=2010, rating=8.8)
print(f"Created: {movie}")
print(f"  Title: {movie.title}")
print(f"  Year: {movie.year}")
print(f"  Rating: {movie.rating}")

# =============================================================================
# PART 2: Automatic Type Coercion
# =============================================================================

print("\n=== Automatic Type Coercion ===\n")

# Pydantic converts types automatically when possible
movie2 = Movie(title="Avatar", year="2009", rating="7.9")  # Strings!
print(f"Input: year='2009', rating='7.9'")
print(f"Result: year={movie2.year} (type: {type(movie2.year).__name__})")
print(f"Result: rating={movie2.rating} (type: {type(movie2.rating).__name__})")

# =============================================================================
# PART 3: Validation Errors
# =============================================================================

print("\n=== Validation Errors ===\n")

# Invalid type that can't be coerced
try:
    Movie(title="Test", year="not a year", rating=8.0)
except ValidationError as e:
    print("Invalid year string:")
    print(f"  {e}")

# =============================================================================
# PART 4: Field Constraints
# =============================================================================

print("\n=== Field Constraints ===\n")

class MovieConstrained(BaseModel):
    title: str = Field(..., min_length=1, description="Movie title")
    year: int = Field(..., ge=1888, le=2030, description="Release year")
    rating: float = Field(..., ge=0, le=10, description="IMDB rating")
    revenue: Optional[int] = Field(None, ge=0, description="Box office revenue")

# Valid movie
movie3 = MovieConstrained(title="Inception", year=2010, rating=8.8)
print(f"Valid: {movie3.title} ({movie3.year})")

# Invalid: year too old
try:
    MovieConstrained(title="Ancient", year=1500, rating=5.0)
except ValidationError as e:
    print(f"\nYear constraint violation:")
    for error in e.errors():
        print(f"  {error['loc'][0]}: {error['msg']}")

# Invalid: rating too high
try:
    MovieConstrained(title="Perfect", year=2020, rating=15.0)
except ValidationError as e:
    print(f"\nRating constraint violation:")
    for error in e.errors():
        print(f"  {error['loc'][0]}: {error['msg']}")

# =============================================================================
# PART 5: Optional and Default Values
# =============================================================================

print("\n=== Optional and Default Values ===\n")

class MovieWithDefaults(BaseModel):
    title: str
    year: int
    rating: Optional[float] = None
    genres: List[str] = []
    is_released: bool = True

movie4 = MovieWithDefaults(title="Mystery Film", year=2020)
print(f"Title: {movie4.title}")
print(f"Rating: {movie4.rating}")  # None
print(f"Genres: {movie4.genres}")  # []
print(f"Released: {movie4.is_released}")  # True

# =============================================================================
# PART 6: Custom Validators
# =============================================================================

print("\n=== Custom Validators ===\n")

class MovieWithValidators(BaseModel):
    title: str
    year: int
    rating: Optional[float] = None

    @field_validator('title')
    @classmethod
    def title_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty or whitespace')
        return v.strip()

    @field_validator('rating')
    @classmethod
    def rating_must_be_valid(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Rating must be between 0 and 10')
        return v

# Test empty title
try:
    MovieWithValidators(title="   ", year=2020)
except ValidationError as e:
    print("Empty title rejected:")
    print(f"  {e.errors()[0]['msg']}")

# Test title stripping
movie5 = MovieWithValidators(title="  Inception  ", year=2010, rating=8.8)
print(f"\nTitle stripped: '{movie5.title}'")

# =============================================================================
# PART 7: Nested Models
# =============================================================================

print("\n=== Nested Models ===\n")

class Person(BaseModel):
    name: str
    birth_year: Optional[int] = None

class MovieWithDirector(BaseModel):
    title: str
    year: int
    director: Person
    actors: List[Person] = []

movie6 = MovieWithDirector(
    title="Inception",
    year=2010,
    director={"name": "Christopher Nolan", "birth_year": 1970},
    actors=[
        {"name": "Leonardo DiCaprio"},
        {"name": "Joseph Gordon-Levitt", "birth_year": 1981}
    ]
)

print(f"Movie: {movie6.title}")
print(f"Director: {movie6.director.name} (born {movie6.director.birth_year})")
print(f"Actors: {[a.name for a in movie6.actors]}")

# =============================================================================
# PART 8: Serialization
# =============================================================================

print("\n=== Serialization ===\n")

# To dictionary
movie_dict = movie6.model_dump()
print("As dict:")
print(json.dumps(movie_dict, indent=2))

# To JSON string
movie_json = movie6.model_dump_json(indent=2)
print("\nAs JSON:")
print(movie_json)

# =============================================================================
# PART 9: Parse from Dict/JSON
# =============================================================================

print("\n=== Parse from Dict/JSON ===\n")

# From dictionary
raw_data = {
    "title": "The Matrix",
    "year": 1999,
    "rating": 8.7
}
movie7 = Movie(**raw_data)
print(f"From dict: {movie7}")

# From JSON string
json_str = '{"title": "Avatar", "year": 2009, "rating": 7.9}'
movie8 = Movie.model_validate_json(json_str)
print(f"From JSON: {movie8}")

# =============================================================================
# PART 10: Batch Validation
# =============================================================================

print("\n=== Batch Validation ===\n")

raw_movies = [
    {"title": "Inception", "year": 2010, "rating": 8.8},
    {"title": "Avatar", "year": "2009", "rating": "7.9"},  # Will coerce
    {"title": "", "year": 2020, "rating": 7.0},  # Empty title
    {"title": "Future", "year": 2050, "rating": 8.0},  # Future year (if constrained)
    {"title": "Test", "year": "invalid", "rating": 5.0},  # Invalid year
]

valid_movies = []
invalid_movies = []

for i, raw in enumerate(raw_movies):
    try:
        # Using constrained model
        movie = MovieWithValidators(**raw)
        valid_movies.append(movie)
    except ValidationError as e:
        invalid_movies.append({
            "index": i,
            "data": raw,
            "errors": e.errors()
        })

print(f"Valid: {len(valid_movies)}")
print(f"Invalid: {len(invalid_movies)}")

print("\nInvalid entries:")
for item in invalid_movies:
    title = item['data'].get('title', 'N/A')
    errors = [f"{e['loc'][0]}: {e['msg']}" for e in item['errors']]
    print(f"  [{item['index']}] {title}: {', '.join(errors)}")
