# Model artifacts

No trained checkpoints, feature caches, private evaluation reports, or dataset-derived artifacts are bundled. Model source and inference adapters remain under `apps/api/app/services/`; reusable training commands remain under `scripts/`.

`download_model_weights.py` optionally retrieves upstream DINOv2 and Torchvision backbones. It does not recreate the task-specific classifiers. Review upstream licenses and execute downloads only in a trusted environment.

Provide an authorized exterior-view checkpoint through `GUARDIAN_VIEW_MODEL_PATH`. Optional specialists use the paths defined by their service adapters. Consult [Model pipeline](../docs/model-pipeline.md) before enabling inference or promotion. Missing assets are surfaced through readiness, review states, or blocked capture gates; a clean checkout cannot perform every recognition task.
