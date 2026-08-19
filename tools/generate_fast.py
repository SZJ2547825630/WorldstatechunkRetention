import os
import runpy
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(ROOT_DIR, "lingbot-world")
TARGET_SCRIPT = os.path.join(PROJECT_DIR, "generate_fast.py")
PATH_ARGS = {"--ckpt_dir", "--image", "--action_path"}


def _resolve_input_path(raw_path: str) -> str:
    if not raw_path or os.path.isabs(raw_path):
        return raw_path

    candidate_roots = [
        os.getcwd(),
        ROOT_DIR,
        PROJECT_DIR,
    ]
    for candidate_root in candidate_roots:
        candidate_path = os.path.join(candidate_root, raw_path)
        if os.path.exists(candidate_path):
            return candidate_path
    return raw_path


def _rewrite_argv(argv: list[str]) -> list[str]:
    rewritten = [argv[0]]
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key in PATH_ARGS:
                rewritten.append(f"{key}={_resolve_input_path(value)}")
            else:
                rewritten.append(arg)
            idx += 1
            continue

        rewritten.append(arg)
        if arg in PATH_ARGS and idx + 1 < len(argv):
            rewritten.append(_resolve_input_path(argv[idx + 1]))
            idx += 2
            continue
        idx += 1
    return rewritten


def main() -> None:
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"Target script not found: {TARGET_SCRIPT}")

    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    sys.argv = _rewrite_argv(sys.argv)
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
