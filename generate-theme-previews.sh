#!/bin/bash

# Generate previews for all built-in Marp themes
# Built-in themes: default, gaia, uncover

SLIDE_FILE="data-collection-labeling.md"
OUTPUT_DIR="theme-previews"

mkdir -p "$OUTPUT_DIR"

echo "Generating theme previews..."

# Built-in themes
for theme in default gaia uncover; do
    echo "Building with theme: $theme"

    # Create a temporary markdown file with the theme
    TEMP_FILE="${OUTPUT_DIR}/temp-${theme}.md"

    # Replace theme in frontmatter
    sed "s/^theme: .*/theme: $theme/" "$SLIDE_FILE" > "$TEMP_FILE"

    # Generate PDF
    marp "$TEMP_FILE" -o "${OUTPUT_DIR}/${theme}.pdf" --pdf --allow-local-files

    # Generate HTML
    marp "$TEMP_FILE" -o "${OUTPUT_DIR}/${theme}.html" --html

    # Clean up temp file
    rm "$TEMP_FILE"
done

echo "✓ Theme previews generated in $OUTPUT_DIR/"
echo ""
echo "Available themes:"
ls -1 "$OUTPUT_DIR"/*.pdf | sed 's|.*/||;s|\.pdf$||'
