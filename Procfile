release: python -c "import os, sys; sys.exit(0 if os.path.exists('faiss.index') else 1)" || python build_index.py
web: uvicorn app:app --host 0.0.0.0 --port $PORT
