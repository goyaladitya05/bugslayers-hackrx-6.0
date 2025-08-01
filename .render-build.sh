#!/usr/bin/env bash
pip install -r requirements.txt
pip freeze  # 👈 this will show in logs whether gunicorn/uvicorn got installed
