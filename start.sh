#!/bin/bash
conda activate myenv
# Add your app startup commands here
echo "Starting the app..."

pip install -r requirements.txt

gunicorn --bind :5000 main:app --workers 1 --threads 8 --timeout 0
