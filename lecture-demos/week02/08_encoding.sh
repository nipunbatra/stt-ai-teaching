DATA_DIR="$(dirname "$0")/data"
cd "$DATA_DIR" || exit 1

# Check file encoding
file movies.csv

# The file command guesses encoding
file -i movies.csv

# For more accuracy, use chardet (Python)
pip install chardet
chardetect movies.csv

python -c "import chardet; print(chardet.detect(open('movies.csv','rb').read()))"


# =============================================================================
# PART 2: Converting encodings
# =============================================================================

# Convert from Latin-1 to UTF-8
$ iconv -f ISO-8859-1 -t UTF-8 movies_latin1.csv > new_file.csv
# Convert from Windows-1252 to UTF-8
$ iconv -f WINDOWS-1252 -t UTF-8 movies_windows1252.csv > utf8_file.csv
# List available encodings
$ iconv -l



