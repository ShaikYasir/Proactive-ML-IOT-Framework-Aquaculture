#!/usr/bin/env bash

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
uvicorn api:app --host 0.0.0.0 --port 8000
