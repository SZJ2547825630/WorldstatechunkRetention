# artifacts/

**Local-only** experiment dumps (gitignored). Not part of the public API.

Typical contents on a workstation:

- `archive_pre_paper/` — pre-cleanup benches, obsolete docs, relocated outputs
- `runs/` — ad-hoc run logs
- `archives/` — zips from `tools/pack_mosaic.sh`

Collaborators should regenerate benches under `lingbot-world/output/` instead
of relying on this folder. Do not commit large media or checkpoints here.
