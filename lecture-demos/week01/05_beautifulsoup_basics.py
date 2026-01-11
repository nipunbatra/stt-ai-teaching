#!/usr/bin/env python3
"""
Week 01: BeautifulSoup Basics
Parsing HTML for web scraping
"""

from bs4 import BeautifulSoup

# =============================================================================
# PART 1: Parse HTML String
# =============================================================================

html = """
<html>
<head><title>My Page</title></head>
<body>
    <h1>Welcome</h1>
    <p class="intro">This is an introduction.</p>
    <p class="content">Main content here.</p>
    <a href="https://example.com">Visit Example</a>
</body>
</html>
"""

soup = BeautifulSoup(html, "html.parser")

# Get title
print(f"Title: {soup.title.text}")

# Get first h1
print(f"Heading: {soup.h1.text}")

# =============================================================================
# PART 2: Finding Elements
# =============================================================================

# Find one element
intro = soup.find("p", class_="intro")
print(f"Intro paragraph: {intro.text}")

# Find all paragraphs
all_paragraphs = soup.find_all("p")
print(f"\nAll paragraphs ({len(all_paragraphs)}):")
for p in all_paragraphs:
    print(f"  - {p.text}")

# Find by attribute
link = soup.find("a", href=True)
print(f"\nLink: {link['href']} -> {link.text}")

# =============================================================================
# PART 3: CSS Selectors (Often Cleaner!)
# =============================================================================

# Select with CSS selector
intro = soup.select_one("p.intro")
print(f"\nCSS selector result: {intro.text}")

# Select all links
links = soup.select("a[href]")
for link in links:
    print(f"Link: {link['href']}")

# =============================================================================
# PART 4: Navigating the Tree
# =============================================================================

body = soup.body

# Children
print("\nDirect children of body:")
for child in body.children:
    if child.name:  # Skip text nodes
        print(f"  <{child.name}>")

# Parent
h1 = soup.h1
print(f"\nParent of h1: <{h1.parent.name}>")

# =============================================================================
# PART 5: Extracting Data
# =============================================================================

html_table = """
<table>
    <tr><th>Movie</th><th>Year</th><th>Rating</th></tr>
    <tr><td>Inception</td><td>2010</td><td>8.8</td></tr>
    <tr><td>Interstellar</td><td>2014</td><td>8.6</td></tr>
    <tr><td>The Matrix</td><td>1999</td><td>8.7</td></tr>
</table>
"""

soup = BeautifulSoup(html_table, "html.parser")

# Extract table data
rows = soup.find_all("tr")[1:]  # Skip header row
movies = []

for row in rows:
    cols = row.find_all("td")
    movie = {
        "title": cols[0].text,
        "year": int(cols[1].text),
        "rating": float(cols[2].text)
    }
    movies.append(movie)

print("Extracted movies:")
for m in movies:
    print(f"  {m['title']} ({m['year']}) - {m['rating']}")
