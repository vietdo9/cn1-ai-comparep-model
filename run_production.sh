#!/bin/bash
# Run app on VPS with Gunicorn

# Activate virtual environment
source venv/bin/activate

# Start Gunicorn server
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --timeout 60 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         web_app:app
