# Model artifacts

English | [Traditional Chinese](README.zh-TW.md)

Guardian separates model integration from model artifacts. The portfolio shows how vision tasks enter the application workflow; trained checkpoints, feature caches, private evaluation reports, and dataset-derived artifacts are supplied separately.

## Implementation to explore

| Source | What it demonstrates |
| --- | --- |
| [Inference orchestration](../apps/api/app/services/inspection_analysis.py) | Task dispatch, failure handling, and review routing |
| [Service adapters](../apps/api/app/services/) | Model-specific integration with structured application results |
| [Training scripts](../scripts/) | Dataset preparation, candidate training, and evaluation tooling |

## Local provisioning

`download_model_weights.py` optionally retrieves upstream DINOv2 and Torchvision backbones. It does not recreate the task-specific classifiers. Review upstream licenses and execute downloads only in a trusted environment.

Provide an authorized exterior-view checkpoint through `GUARDIAN_VIEW_MODEL_PATH`. Optional specialists use the paths defined by their service adapters. Consult [Model pipeline](../docs/model-pipeline.md) before enabling inference or promotion. Missing assets are surfaced through readiness, review states, or blocked capture gates; a clean checkout cannot perform every recognition task.

[Project overview](../README.md)
