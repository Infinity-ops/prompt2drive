# demo_runs/

Real, unmodified runs replayed by the web UI in `--mode replay` and by the Docker demo.
They are copied from the paper's full closed-loop traces with the Qwen2.5-VL-7B-AWQ verifier
(`outputs/full_cloud_reflector_run_QwenVL25_AWQ/`):

| Run | Prompt | Iterations |
|---|---|---|
| `s01` | A pedestrian suddenly crosses the road near a traffic signal in rainy weather. | 1 |
| `s04` | The ego vehicle performs an emergency brake at night when a pedestrian steps off the curb. | 1 |
| `s06` | Two pedestrians suddenly cross near a traffic signal in heavy rain at a signalized junction. | 1 |
| `s18` | Dense fog causes a multi-vehicle pile-up scenario on a dual carriageway as vehicles brake suddenly. | 2 (Reflector engaged) |

Replay never regenerates or re-scores anything. Recorded traces store the score, pass/fail and
the failed predicates, but not per-predicate agreement, so the UI shows only what was recorded.

To add a run of your own, copy its folder unchanged:
`cp -r outputs/ui/<run_id> demo_runs/`
