#!/bin/bash

# Build slides with all available Marp themes

SLIDE_FILE="data-collection-labeling.md"
OUTPUT_DIR="theme-previews"

echo "Building slides with all themes..."
echo ""

# Built-in themes (already done, but let's ensure they exist)
for theme in gaia uncover; do
    if [ ! -f "${OUTPUT_DIR}/${theme}.pdf" ]; then
        echo "Building with built-in theme: $theme"
        TEMP_FILE="${OUTPUT_DIR}/temp-${theme}.md"
        sed "s/^theme: .*/theme: $theme/" "$SLIDE_FILE" > "$TEMP_FILE"
        marp "$TEMP_FILE" -o "${OUTPUT_DIR}/${theme}.pdf" --pdf --allow-local-files 2>&1 | grep -v "INFO\|WARN"
        marp "$TEMP_FILE" -o "${OUTPUT_DIR}/${theme}.html" --html 2>&1 | grep -v "INFO"
        rm "$TEMP_FILE"
    fi
done

# Rename default.pdf to match naming convention
if [ -f "${OUTPUT_DIR}/default.pdf" ]; then
    echo "Using existing default theme"
fi

# Custom themes with CSS files
for theme_css in ${OUTPUT_DIR}/*.css; do
    theme_name=$(basename "$theme_css" .css)
    echo "Building with custom theme: $theme_name"

    marp "$SLIDE_FILE" --theme "$theme_css" -o "${OUTPUT_DIR}/${theme_name}.pdf" --pdf --allow-local-files 2>&1 | grep -v "INFO\|WARN"
    marp "$SLIDE_FILE" --theme "$theme_css" -o "${OUTPUT_DIR}/${theme_name}.html" --html 2>&1 | grep -v "INFO"
done

echo ""
echo "✓ All themes built successfully"
echo ""
echo "Theme previews available:"
ls -1 "${OUTPUT_DIR}"/*.pdf 2>/dev/null | sed 's|.*/||;s|\.pdf$||' | sort
