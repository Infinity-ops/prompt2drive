# Public demo image: web UI in REPLAY mode (recorded runs only).
# No CARLA, GPU or API keys inside -- nothing here can spend credits.
FROM python:3.11-slim

WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY webui/ webui/
COPY scenarios/ scenarios/
COPY demo_runs/ demo_runs/

ENV P2D_UI_MODE=replay \
    P2D_UI_RUNS_DIR=demo_runs \
    P2D_UI_HOST=0.0.0.0 \
    PORT=8080
EXPOSE 8080
CMD ["python", "-m", "webui.server"]
