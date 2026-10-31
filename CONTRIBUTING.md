# Contributing

Contributions should improve the reusable architecture, source quality, or English engineering documentation. Keep operational data and generated artifacts outside the repository.

## Setup

Use Python 3.13, Node.js 22 or later, Git, and Windows PowerShell for the provided helpers.

```powershell
Copy-Item .env.example .env
.\scripts\setup.ps1
```

Supply your own bootstrap settings in `.env`. Backend runtime requires `scripts/setup.ps1 -WithModels`; isolated OCR provisioning uses `-WithOcr`. Task-specific checkpoints and authorized datasets must be provided separately. See the [README](README.md) and [model pipeline](docs/model-pipeline.md) before running inference or training.

## Change review

Describe the concrete behavior changed, the reason, and the checks performed. For workflow changes, consider interrupted jobs, duplicate processing, stale attempts, model failure, and authorization boundaries. For model changes, keep private evaluation evidence and artifact provenance outside source control, and summarize the method without publishing operational samples.

Do not add credentials, real vehicle identifiers or license plates, photographs, private datasets, checkpoints, database files, test fixtures, screenshots, or experiment reports to the public source tree. Use data-free diagrams and written scenarios in documentation. The explicitly reviewed, sanitized interface preview at `docs/assets/guardian-demo-redacted.png` is the sole screenshot exception; its display name and vehicle reference photograph were replaced before publication. This exception does not authorize adding other screenshots or original image inputs.

## Checks

Before staging, inspect source on disk:

```powershell
.\.venv\Scripts\python.exe scripts/check_public_release.py --worktree
```

After staging, run the release and build checks:

```powershell
.\scripts\verify.ps1
```

The checks scan staged content with heuristics, compile Python, and build both frontends. They do not replace behavioral evaluation, a dedicated security review, or model validation. Record meaningful additional checks for the behavior you change, without adding private test material to this repository. Keep all Markdown in English and verify local links.

## Commits

Use English Conventional Commits with a concise description, for example `feat: improve durable job recovery`, `fix: preserve review reason on timeout`, or `docs: clarify model provisioning`. Keep unrelated changes separate and describe material limitations in the change request.

## Licensing

A project license has not been selected. Do not introduce third-party code or artifacts without establishing their terms and your authority to contribute them. Dependencies, models, and excluded datasets retain their own licensing and usage constraints.
