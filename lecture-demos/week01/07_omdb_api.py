#!/usr/bin/env python3
"""
Week 01: OMDb API Example
Real-world API with API key authentication

Get your free API key at: https://www.omdbapi.com/apikey.aspx
"""

import requests
import os

# =============================================================================
# SETUP: Get API Key
# =============================================================================

# Option 1: Set environment variable
# export OMDB_API_KEY=your_key_here

# Option 2: Replace directly (not recommended for production)
API_KEY = os.environ.get("OMDB_API_KEY", "YOUR_KEY_HERE")

if API_KEY == "YOUR_KEY_HERE":
    print("=" * 60)
    print("Please set your OMDb API key!")
    print("Get a free key at: https://www.omdbapi.com/apikey.aspx")
    print("Then: export OMDB_API_KEY=your_key")
    print("=" * 60)
    exit(1)

# =============================================================================
# PART 1: Search by Title
# =============================================================================

def get_movie(title):
    """Fetch movie details by title"""
    response = requests.get(
        "https://www.omdbapi.com/",
        params={
            "t": title,
            "apikey": API_KEY
        }
    )
    return response.json()

movie = get_movie("Inception")
print(f"Title: {movie['Title']}")
print(f"Year: {movie['Year']}")
print(f"Director: {movie['Director']}")
print(f"IMDB Rating: {movie['imdbRating']}")
print(f"Plot: {movie['Plot'][:100]}...")

# =============================================================================
# PART 2: Search Multiple Movies
# =============================================================================

def search_movies(query):
    """Search for movies matching query"""
    response = requests.get(
        "https://www.omdbapi.com/",
        params={
            "s": query,
            "apikey": API_KEY
        }
    )
    data = response.json()
    return data.get("Search", [])

print("\n--- Movies with 'Batman' ---")
results = search_movies("Batman")
for movie in results[:5]:  # First 5 results
    print(f"  {movie['Title']} ({movie['Year']})")

# =============================================================================
# PART 3: Error Handling
# =============================================================================

def safe_get_movie(title):
    """Fetch movie with error handling"""
    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={"t": title, "apikey": API_KEY},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "False":
            print(f"Movie not found: {data.get('Error')}")
            return None

        return data

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Test with non-existent movie
print("\n--- Testing error handling ---")
result = safe_get_movie("ThisMovieDoesNotExist12345")
print(f"Result: {result}")

# Test with real movie
result = safe_get_movie("The Matrix")
if result:
    print(f"Found: {result['Title']} ({result['Year']})")

# =============================================================================
# PART 4: Build a Simple Movie Dataset
# =============================================================================

def build_movie_dataset(titles):
    """Fetch multiple movies and create a dataset"""
    import time

    movies = []
    for title in titles:
        movie = safe_get_movie(title)
        if movie:
            movies.append({
                "title": movie["Title"],
                "year": movie["Year"],
                "rating": movie.get("imdbRating", "N/A"),
                "genre": movie.get("Genre", "N/A"),
                "director": movie.get("Director", "N/A")
            })
        time.sleep(0.5)  # Rate limiting

    return movies

# Build dataset
print("\n--- Building Movie Dataset ---")
titles = ["Inception", "Interstellar", "The Dark Knight", "Pulp Fiction", "Fight Club"]
dataset = build_movie_dataset(titles)

print("\nCollected Movies:")
for m in dataset:
    print(f"  {m['title']} ({m['year']}) - {m['rating']} - {m['genre']}")
