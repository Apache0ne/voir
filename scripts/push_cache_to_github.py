"""Push a generated cache directory to a dedicated GitHub branch.

The GitHub token is read from ``GITHUB_TOKEN`` and is never printed. The remote
URL is restored to the clean public URL after the push.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import quote


def _run(command: list[str], cwd: Path, *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="/content/voir")
    parser.add_argument("--repository", default="Apache0ne/voir")
    parser.add_argument("--branch", default="mage-cache-16")
    parser.add_argument("--path", default="mage_cache/real16")
    parser.add_argument("--message", default="Cache 16 actual Mage-Flow-Edit-Turbo trajectories")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    repo_dir = Path(args.repo_dir).resolve()
    cache_path = repo_dir / args.path
    if not (cache_path / "run.json").exists() or not (cache_path / "manifest.jsonl").exists():
        raise RuntimeError(f"cache is incomplete: {cache_path}")

    _run(["git", "config", "user.name", "VOIR Colab Cache"], repo_dir)
    _run(["git", "config", "user.email", "voir-cache@users.noreply.github.com"], repo_dir)
    _run(["git", "fetch", "origin", "--prune"], repo_dir)
    remote_exists = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{args.branch}"],
        cwd=repo_dir,
    ).returncode == 0
    start = f"origin/{args.branch}" if remote_exists else "origin/main"
    _run(["git", "checkout", "-B", args.branch, start], repo_dir)
    _run(["git", "add", args.path], repo_dir)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir).returncode != 0
    if changed:
        _run(["git", "commit", "-m", args.message], repo_dir)

    clean_url = f"https://github.com/{args.repository}.git"
    auth_url = f"https://x-access-token:{quote(token, safe='')}@github.com/{args.repository}.git"
    _run(["git", "remote", "set-url", "origin", auth_url], repo_dir)
    try:
        _run(["git", "push", "origin", f"HEAD:{args.branch}"], repo_dir)
    finally:
        _run(["git", "remote", "set-url", "origin", clean_url], repo_dir)

    commit = _run(["git", "rev-parse", "HEAD"], repo_dir, capture=True)
    print(f"CACHE_BRANCH={args.branch}")
    print(f"CACHE_COMMIT={commit}")
    print(f"CACHE_URL=https://github.com/{args.repository}/tree/{args.branch}/{args.path}")


if __name__ == "__main__":
    main()
