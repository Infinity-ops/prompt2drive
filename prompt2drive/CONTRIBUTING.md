# Contributing to Prompt-to-Drive

Contributions are welcome — this project accompanies a peer-reviewed workshop paper and we'd like the reference implementation to stay correct and useful as a research baseline.

## Ways to contribute

- **New scenario types**: extend `engine/scenario_spec.py`'s `Literal` types and add corresponding entries to `WEATHER_PRESETS` (carla_runner.py) and `ELEMENT_CORRECTIONS` (reflector/agent.py).
- **Additional evaluation scenarios**: add to `scenarios/dataset.json`, following the existing schema. PRs that add scenarios from real-world ODD descriptions or established benchmarks (e.g., nuScenes-derived prompts) are especially welcome.
- **Alternative VLM/LLM backends**: if you add support for a non-OpenAI provider, please keep the `Generator`/`Verifier` class interfaces unchanged so the orchestrator and eval scripts continue to work without modification.
- **Bug reports**: please include the `pipeline_run.json` for the failing scenario if possible — it contains the full iteration trace and is the fastest way to diagnose issues.

## Development setup

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
pytest tests/ -v --cov=engine --cov=generator --cov=verifier --cov=reflector
```

The unit tests in `tests/test_core.py` deliberately avoid any CARLA connection or API call so they can run in CI. Please keep new unit tests in that style; put anything requiring a live CARLA server or API key in clearly separated integration tests instead.

## Code style

- Type hints on all public function signatures.
- Docstrings explaining *why*, not just *what*, especially for anything touching the Generator/Verifier/Reflector boundary — the design rationale matters as much as the code for this project.
- Run `pytest` before submitting a PR.

## Reporting issues

Please open a GitHub issue with:
1. Your CARLA version
2. The exact command you ran
3. The full traceback or unexpected output
4. If applicable, the relevant `outputs/<scenario_id>/pipeline_run.json`
