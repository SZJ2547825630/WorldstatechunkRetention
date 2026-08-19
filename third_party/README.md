# third_party

Optional external dependencies that are **not** vendored into git.

## WorldKV (optional baseline)

Official training-free KV retrieval baseline used in RealCam-Vid comparisons.

```bash
# From repo root
git clone https://github.com/cvlab-kaist/WorldKV.git third_party/WorldKV
# or: git clone <your-fork> third_party/WorldKV
```

`env.sh` sets:

```bash
export WORLDKV_ROOT="$LINGBOT_ROOT/third_party/WorldKV"
```

Then:

```bash
source env.sh
cd lingbot-world
bash bench/realcamvid/run_worldkv_official.sh
```

See [`lingbot-world/bench/worldkv/README.md`](../lingbot-world/bench/worldkv/README.md).

> Do **not** commit a symlink to a machine-local absolute path (e.g. `/DATA/.../WorldKV`).
> Clone into this directory so the relative path works for every collaborator.
