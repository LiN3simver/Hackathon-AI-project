# Runtime data boundary

English | [Traditional Chinese](README.zh-TW.md)

This directory is the private runtime side of Guardian's evidence workflow. The API creates SQLite databases, inspection images, learning samples, and local dataset manifests here as needed. The portfolio includes the storage implementation, without operational records, photos, labels, or fixtures.

## Design intent

Database records connect uploaded files to inspection sessions, jobs, and review decisions. Keeping runtime data separate lets the source demonstrate that relationship without publishing inspection evidence.

## Local use

Use only data you are authorized to process and keep backups outside public source control. Renaming a photograph or hashing its filename does not anonymize its pixels, metadata, or associated records.

See [Data lifecycle](../docs/data-lifecycle.md) for storage boundaries and deletion considerations.

[Project overview](../README.md)
