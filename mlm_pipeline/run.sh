#!/bin/bash

DIR="$(dirname "$(readlink -fm "$0")")"

# Create csv file if it doesn't exist
if [ ! -f "$DIR/data/model_df.csv" ]; then
  python $DIR/python/create_csv.py
else
  echo "model_df.csv already exists"
fi

# Run the basic MLM
Rscript $DIR/r/day_model.R
