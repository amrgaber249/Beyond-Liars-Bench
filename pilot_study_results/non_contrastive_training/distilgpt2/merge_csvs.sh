# Iterate over every file in the current directory. For every csv file,
# append its contents to the file with the same name in ./merged/
for file in ./*; do
    if [[ $file == *.csv ]]; then
        filename=$(basename "$file")
        # Only append the second line of the contents to avoid duplicating headers
        # Skip the header line and the last line
        tail -n +2 "$file" | head -n -1 >> "../_metric_tables/${filename}"
    fi
done