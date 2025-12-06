# Usage

This document explains how to run the project locally and the available endpoints.

Prerequisites
- Python 3.8+
- Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

Run the web UI

```powershell
python app.py
# then open http://127.0.0.1:5000
```

API Endpoints

- `GET /` — Serves the web UI (`templates/index.html`).
- `POST /predict` — Accepts an image and returns predicted text.
  - Accepts multipart form-data with `file` field, or JSON with `image` containing a base64 data URI.
  - Response: `{'success': True, 'text': '<predicted text>'}`
- `GET /health` — Simple health check returning `status`, `model_loaded`, and `device`.
- `GET /info` — Returns model information (`num_classes`, `model_parameters`, `device`, and `config`).

Notes for contributors
- If `models/best_model.pkl` isn't present, the app will run with an untrained demo model (default charset).
- Use `src/train.py` for training experiments. Add a config or CLI flags if you want to parameterize runs.

Testing locally
- Add unit tests under `tests/` and run with your test runner of choice (e.g., `pytest`).
