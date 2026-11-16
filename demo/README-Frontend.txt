# BlinkLab frontend

Place the `static/` directory beside your existing `app.py`, `config.py`, and `acquisition.py`.

Expected files:
- static/index.html
- static/style.css
- static/app.js
- static/assets/blink-icon.png

The frontend expects FastAPI to serve `/` and mount `/static`, and a WebSocket at `/ws`.

Expected JSON messages:
- `{"type":"status","message":"BITalino connected. Streaming started."}`
- `{"type":"eeg_batch","sampling_rate":1000,"channel":"A4","samples":[...],"received_at":123.45}`
- `{"type":"error","message":"..."}`

The interface intentionally does not implement blink classification. The blink event panel remains a placeholder until the trained model's actual input/output contract is agreed.
