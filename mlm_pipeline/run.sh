#!/bin/bash

set -e

DIR="$(dirname "$(readlink -fm "$0")")"

# Create csv file if it doesn't exist
if [ ! -f "$DIR/data/model_df.csv" ]; then
  python $DIR/python/create_csv.py
else
  echo "model_df.csv already exists"
fi

python $DIR/python/stats.py

# Compare the different multi level models - to motivate that the random intercept and slope model is the best
Rscript $DIR/r/compare_day_models.R

# Run models with fixed effects from demographic information
Rscript $DIR/r/demographic_models.R
