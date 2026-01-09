#!/usr/bin/env python3
"""
Week 01: Complete Web Scraping Example
Combining requests + BeautifulSoup
"""

import requests
from bs4 import BeautifulSoup
import time

# =============================================================================
# PART 1: Fetch and Parse a Web Page
# =============================================================================

def scrape_page(url):
    """Fetch URL and return BeautifulSoup object"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Educational Bot)"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")

# =============================================================================
# PART 2: Example - Scrape Our Sample Movie Website
# =============================================================================

# Note: In actual lecture, use your hosted sample-movie-website.html
# For demo, we'll parse a local HTML string simulating the page

sample_html = """
<html>
<body>
    <div class="movie-card">
        <h2>Inception</h2>
        <p class="year">2010</p>
        <p class="genre">Sci-Fi, Action</p>
        <p class="rating">8.8/10</p>
    </div>
    <div class="movie-card">
        <h2>The Dark Knight</h2>
        <p class="year">2008</p>
        <p class="genre">Action, Crime</p>
        <p class="rating">9.0/10</p>
    </div>
    <div class="movie-card">
        <h2>Interstellar</h2>
        <p class="year">2014</p>
        <p class="genre">Sci-Fi, Drama</p>
        <p class="rating">8.6/10</p>
    </div>
</body>
</html>
"""

soup = BeautifulSoup(sample_html, "html.parser")

# Extract all movies
movies = []
for card in soup.select(".movie-card"):
    movie = {
        "title": card.h2.text,
        "year": card.select_one(".year").text,
        "genre": card.select_one(".genre").text,
        "rating": card.select_one(".rating").text
    }
    movies.append(movie)

print("Scraped Movies:")
for m in movies:
    print(f"  {m['title']} ({m['year']}) - {m['genre']} - {m['rating']}")

# =============================================================================
# PART 3: Ethical Scraping - Check robots.txt
# =============================================================================

def check_robots_txt(domain):
    """Check if robots.txt allows scraping"""
    robots_url = f"{domain}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            print(f"robots.txt for {domain}:")
            print(response.text[:500])  # First 500 chars
            return response.text
    except:
        print(f"Could not fetch robots.txt for {domain}")
    return None

# Example
check_robots_txt("https://www.google.com")

# =============================================================================
# PART 4: Rate Limiting
# =============================================================================

def scrape_multiple_pages(urls, delay=1.0):
    """Scrape multiple URLs with delay between requests"""
    results = []

    for url in urls:
        print(f"Fetching: {url}")
        try:
            soup = scrape_page(url)
            results.append({"url": url, "title": soup.title.text if soup.title else "N/A"})
        except Exception as e:
            results.append({"url": url, "error": str(e)})

        time.sleep(delay)  # Be polite!

    return results

# Example (commented out to not make actual requests)
# urls = [
#     "https://example.com",
#     "https://example.org",
# ]
# results = scrape_multiple_pages(urls)
# for r in results:
#     print(r)

# =============================================================================
# PART 5: Save Scraped Data
# =============================================================================

import json
import csv

# Save as JSON
with open("movies.json", "w") as f:
    json.dump(movies, f, indent=2)
print("\nSaved to movies.json")

# Save as CSV
with open("movies.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "year", "genre", "rating"])
    writer.writeheader()
    writer.writerows(movies)
print("Saved to movies.csv")

# Cleanup
import os
os.remove("movies.json")
os.remove("movies.csv")
print("(Cleaned up demo files)")
