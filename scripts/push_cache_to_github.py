"""Push a generated cache directory to a dedicated GitHub branch.

Authentication can use either:

* ``GITHUB_DEPLOY_KEY``: an SSH private deploy key with write access to this repo.
* ``GITHUB_TOKEN``: a fine-grained personal access token with Contents write.

Secrets are never printed. The remote URL is restored after the push and any
temporary deploy-key file is deleted.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote


def _run(
    command: list[str],
    cwd: Path,
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
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
    deploy_key = os.environ.get("GITHUB_DEPLOY_KEY", "").strip()
    if not token and not deploy_key:
        raise RuntimeError("Set either GITHUB_DEPLOY_KEY or GITHUB_TOKEN")

    repo_dir = Path(args.repo_dir).resolve()
    cache_path = repo_dir / args.path
    if not (cache_path / "run.json").exists() or not (cache_path / "manifest.jsonl").exists():
        raise RuntimeError(f"cache is incomplete: {cache_path}")

    clean_url = f"https://github.com/{args.repository}.git"
    environment = os.environ.copy()
    temporary_ssh_dir: Path | None = None

    if deploy_key:
        temporary_ssh_dir = Path(tempfile.mkdtemp(prefix="voir-github-ssh-"))
        key_path = temporary_ssh_dir / "deploy_key"
        key_path.write_text(deploy_key.rstrip() + "\n", encoding="utf-8")
        key_path.chmod(0o600)
        ssh_command = (
            f"ssh -i {shlex.quote(str(key_path))} "
            "-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
        )
        environment["GIT_SSH_COMMAND"] = ssh_command
        authenticated_url = f"git@github.com:{args.repository}.git"
        auth_method = "deploy_key"
    else:
        authenticated_url = (
            f"https://x-access-token:{quote(token, safe='')}@github.com/{args.repository}.git"
        )
        auth_method = "fine_grained_token"

    _run(["git", "config", "user.name", "VOIR Colab Cache"], repo_dir)
    _run(["git", "config", "user.email", "voir-cache@users.noreply.github.com"], repo_dir)
    _run(["git", "remote", "set-url", "origin", authenticated_url], repo_dir)
    try:
        _run(["git", "fetch", "origin", "--prune"], repo_dir, env=environment)
        _run(["git", "checkout", "-B", args.branch, "origin/main"], repo_dir, env=environment)
        _run(["git", "add", args.path], repo_dir, env=environment)
        changed = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_dir,
            env=environment,
        ).returncode != 0
        if changed:
            _run(["git", "commit", "-m", args.message], repo_dir, env=environment)
        _run(
            ["git", "push", "--force-with-lease", "origin", f"HEAD:{args.branch}"],
            repo_dir,
            env=environment,
        )
    finally:
        _run(["git", "remote", "set-url", "origin", clean_url], repo_dir)
        if temporary_ssh_dir is not None:
            shutil.rmtree(temporary_ssh_dir, ignore_errors=True)

    commit = _run(["git", "rev-parse", "HEAD"], repo_dir, capture=True)
    print(f"AUTH_METHOD={auth_method}")
    print(f"CACHE_BRANCH={args.branch}")
    print(f"CACHE_COMMIT={commit}")
    print(f"CACHE_URL=https://github.com/{args.repository}/tree/{args.branch}/{args.path}")


if __name__ == "__main__":
    main()
