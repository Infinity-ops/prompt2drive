# demo_runs/

Recorded Prompt-to-Drive runs that the web UI replays in `--mode replay`
(and that the public Docker image ships with).

Each sub-folder is copied unchanged from a real live-mode run:

    demo_runs/<run_id>/pipeline_run.json
    demo_runs/<run_id>/iter_0/frames/frame_000.png ...

Only real runs belong here. Replay re-streams exactly what was recorded and
never re-scores or invents anything, so what the public page shows is what
the pipeline actually produced. See docs/WEB_UI.md, "Record runs for the
public demo".
