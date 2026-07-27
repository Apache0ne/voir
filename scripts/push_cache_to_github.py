"""Push a generated cache directory to a dedicated GitHub branch.

Authentication can use either ``GITHUB_DEPLOY_KEY`` or ``GITHUB_TOKEN``. The
uploader preserves an already-created local cache commit after a failed push,
checks GitHub's regular-file limit before committing, and reports complete,
secret-redacted Git errors.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_REGULAR_FILE_LIMIT = 100_000_000


def _run(
    command: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            + " ".join(shlex.quote(part) for part in command)
        )
    return result


def _output(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def _validate_pat(token: str, repository: str) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "voir-colab-cache",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub token validation failed: HTTP {exc.code}: {body}") from exc
    permissions = payload.get("permissions") or {}
    if not permissions.get("push", False):
        raise RuntimeError(
            "The fine-grained PAT can read the repository but cannot push. "
            "Select Apache0ne/voir and set Repository permissions > Contents to Read and write."
        )
    return {
        "repository": payload.get("full_name"),
        "private": payload.get("private"),
        "push": bool(permissions.get("push")),
    }


def _largest_files(root: Path, count: int = 10) -> list[tuple[int, Path]]:
    files = [(path.stat().st_size, path) for path in root.rglob("*") if path.is_file()]
    return sorted(files, reverse=True)[:count]


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
        raise RuntimeError("Set either GITHUB_TOKEN or GITHUB_DEPLOY_KEY")

    repo_dir = Path(args.repo_dir).resolve()
    cache_path = (repo_dir / args.path).resolve()
    if not (cache_path / "run.json").exists() or not (cache_path / "manifest.jsonl").exists():
        raise RuntimeError(f"cache is incomplete: {cache_path}")

    largest = _largest_files(cache_path)
    print("Largest cache files:")
    for size, path in largest:
        print(f"  {size / 2**20:8.2f} MiB  {path.relative_to(repo_dir)}")
    oversized = [(size, path) for size, path in largest if size >= GITHUB_REGULAR_FILE_LIMIT]
    if oversized:
        size, path = oversized[0]
        raise RuntimeError(
            f"GitHub blocks regular files >= 100,000,000 bytes: "
            f"{path.relative_to(repo_dir)} is {size:,} bytes"
        )

    clean_url = f"https://github.com/{args.repository}.git"
    environment = os.environ.copy()
    temporary_auth_dir: Path | None = None
    auth_method: str

    if deploy_key:
        temporary_auth_dir = Path(tempfile.mkdtemp(prefix="voir-github-ssh-"))
        key_path = temporary_auth_dir / "deploy_key"
        key_path.write_text(deploy_key.rstrip() + "\n", encoding="utf-8")
        key_path.chmod(0o600)
        environment["GIT_SSH_COMMAND"] = (
            f"ssh -i {shlex.quote(str(key_path))} "
            "-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
        )
        authenticated_url = f"git@github.com:{args.repository}.git"
        auth_method = "deploy_key"
    else:
        print("Token validation:", json.dumps(_validate_pat(token, args.repository), indent=2))
        temporary_auth_dir = Path(tempfile.mkdtemp(prefix="voir-github-pat-"))
        askpass_path = temporary_auth_dir / "askpass.py"
        askpass_path.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "prompt = sys.argv[1].lower() if len(sys.argv) > 1 else ''\n"
            "print('x-access-token' if 'username' in prompt else os.environ['GITHUB_TOKEN'])\n",
            encoding="utf-8",
        )
        askpass_path.chmod(0o700)
        environment["GIT_ASKPASS"] = str(askpass_path)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        authenticated_url = clean_url
        auth_method = "fine_grained_token"

    _run(["git", "config", "user.name", "VOIR Colab Cache"], repo_dir)
    _run(["git", "config", "user.email", "voir-cache@users.noreply.github.com"], repo_dir)
    _run(["git", "config", "http.version", "HTTP/1.1"], repo_dir)
    _run(["git", "config", "http.postBuffer", "524288000"], repo_dir)
    _run(["git", "remote", "set-url", "origin", authenticated_url], repo_dir)

    try:
        _run(["git", "fetch", "origin", "--prune"], repo_dir, env=environment)
        current_branch = _output(["git", "branch", "--show-current"], repo_dir, environment)
        print(f"Current branch: {current_branch or '(detached)'}")

        # A previous failed upload commonly leaves a valid local cache commit on the
        # requested branch. Preserve and push it rather than resetting it away.
        if current_branch != args.branch:
            _run(["git", "switch", "-C", args.branch, "origin/main"], repo_dir, env=environment)

        _run(["git", "add", "--", args.path], repo_dir, env=environment)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_dir,
            env=environment,
        ).returncode != 0
        if staged:
            _run(["git", "commit", "-m", args.message], repo_dir, env=environment)
        else:
            print("No new cache changes to commit; using the existing local commit.")

        commit = _output(["git", "rev-parse", "HEAD"], repo_dir, environment)
        print(f"Local cache commit: {commit}")
        _run(
            ["git", "push", "--set-upstream", "origin", f"HEAD:{args.branch}"],
            repo_dir,
            env=environment,
        )
    finally:
        _run(["git", "remote", "set-url", "origin", clean_url], repo_dir, check=False)
        if temporary_auth_dir is not None:
            shutil.rmtree(temporary_auth_dir, ignore_errors=True)

    print(f"AUTH_METHOD={auth_method}")
    print(f"CACHE_BRANCH={args.branch}")
    print(f"CACHE_COMMIT={commit}")
    print(f"CACHE_URL=https://github.com/{args.repository}/tree/{args.branch}/{args.path}")


if __name__ == "__main__":
    main()
