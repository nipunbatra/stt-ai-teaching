# =============================================================================
# PART 1: Lets look at the Data
# =============================================================================



import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv("data/movies.csv")
print(df.head())

# =============================================================================
# PART 2: Lets dig deeper
# =============================================================================

print(df.info())

# =============================================================================
# PART 3: What happens if we ignore
# =============================================================================

# X = df[['year', 'runtime', 'rating']]
y = df['boxoffice']
model = LinearRegression()
# model.fit(X, y)

# =============================================================================
# PART 4: Silent features
# =============================================================================


# "Fix" by forcing numeric conversion
df['Year'] = pd.to_numeric(df['year'], errors='coerce')
df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')
# Now 13 movies have NaN year, 108 have NaN rating
# We lost data silently!
# Train anyway
model.fit(df[['Year', 'Rating']].dropna(), y.dropna())
# Model trains on 521 movies instead of 1000!
