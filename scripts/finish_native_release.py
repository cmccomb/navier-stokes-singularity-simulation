"""One bounded interactive release job: copy, audit, publish, deploy, then update Pages.

This is not a recurring watcher. Failures preserve the previous published release
and all source data. The only GitHub updates are two immutable-reference records.
"""

import argparse
import hashlib
import json
import os
import resource
import shlex
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from gradio_client import Client
from httpx import HTTPError
from huggingface_hub import HfApi

from scripts.deploy_native_space import deploy
from scripts.export_native_dataset import export
from scripts.publish_native_dataset import publish

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ("spaces/native_explorer/dataset.json", "site/data/native-explorer.json")


def hosted_frame_matches(result, times):
    """The fixed-resolution API returns images and the displayed time control."""
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        return False
    control = result[-1]
    last = len(times) - 1
    label = f"Saved frame {last + 1}/{len(times)} · t = {times[last]:.8f}"
    return (
        isinstance(control, dict)
        and control.get("value") == last
        and control.get("label") == label
    )


def gh(repo, path, data=None, method=None):
    command = ["gh", "api", f"repos/{repo}/{path}"]
    if method:
        command += ["--method", method]
    if data is not None:
        command += ["--input", "-"]
    return json.loads(
        subprocess.check_output(
            command, input=None if data is None else json.dumps(data).encode()
        )
    )


def update_references(repo, expected, replacement, code_shas):
    """Atomically update just the two records, preserving unrelated main changes."""
    head = gh(repo, "git/ref/heads/main")["object"]["sha"]
    commit = gh(repo, f"git/commits/{head}")
    tree = gh(repo, f"git/trees/{commit['tree']['sha']}?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("Cannot verify the complete remote tree")
    entries = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    for path, content in expected.items():
        sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if entries.get(path, {}).get("sha") != sha:
            raise RuntimeError(f"Reference changed during preparation: {path}")
    for path, sha in code_shas.items():
        if entries.get(path, {}).get("sha") != sha:
            raise RuntimeError(f"Explorer source changed during preparation: {path}")
    new_tree = gh(
        repo,
        "git/trees",
        {
            "base_tree": commit["tree"]["sha"],
            "tree": [
                {
                    "path": path,
                    "mode": "100644",
                    "type": "blob",
                    "content": replacement[path],
                }
                for path in REFERENCES
            ],
        },
    )
    audit = gh(repo, f"git/trees/{new_tree['sha']}?recursive=1")
    if audit.get("truncated"):
        raise RuntimeError("Cannot verify the proposed tree")
    after = {row["path"]: row for row in audit["tree"] if row["type"] == "blob"}
    if set(after) != set(entries) or any(
        after[p]["sha"] != row["sha"]
        for p, row in entries.items()
        if p not in REFERENCES
    ):
        raise RuntimeError("Unexpected change outside the two reference records")
    new_commit = gh(
        repo,
        "git/commits",
        {
            "message": "Promote all 280 verified native states in the HF explorer",
            "tree": new_tree["sha"],
            "parents": [head],
        },
    )
    gh(repo, "git/refs/heads/main", {"sha": new_commit["sha"], "force": False}, "PATCH")
    return new_commit["sha"]


def run(args):
    args.job.mkdir(parents=True, exist_ok=False)
    record_path = args.job / "status.json"
    record = {
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "max_seconds": args.max_seconds,
        "status": "running",
    }

    def phase(name, **details):
        record.update(phase=name, updated_at=datetime.now(UTC).isoformat(), **details)
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(record_path)
        print(json.dumps(record), flush=True)

    def expired(signum, frame):
        raise TimeoutError("The bounded release job reached its wall-clock limit")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    try:
        if shutil.disk_usage(args.job).free < 55 * 2**30:
            raise RuntimeError(
                "Require 55 GiB free for native copy, export, and reserve"
            )
        best = args.job / "source-best.json"
        shutil.copyfile(ROOT / "site/data/best.json", best)
        source_record = json.loads(best.read_text())
        expected = {path: (ROOT / path).read_bytes() for path in REFERENCES}
        # Freeze reviewed app sources; refuse to overwrite a concurrent app edit.
        snapshot = args.job / "app"
        snapshot.mkdir()
        code_shas = {}
        for name in ("app.py", "reader.py", "requirements.txt", "README.md"):
            path = f"spaces/native_explorer/{name}"
            content = (ROOT / path).read_bytes()
            shutil.copyfile(ROOT / path, snapshot / name)
            code_shas[path] = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest()
        space_revision = HfApi().space_info(args.hf_repo).sha
        phase("copying", source_frames=source_record["saved_frames"])
        args.native.mkdir(parents=True, exist_ok=True)
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        limit = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
        transport = f"ssh -i {shlex.quote(str(args.identity))} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
        command = [
            "rsync",
            "-arzL",
            "--stats",
            "--timeout=60",
            "--files-from=-",
            "--rsync-path=ulimit -n 4096 && rsync",
            "-e",
            transport,
            f"{args.host}:{args.remote_source}/",
            str(args.native) + "/",
        ]
        with (args.job / "transfer.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.communicate(
                    "".join(r["path"] + "/\n" for r in source_record["frames"]).encode()
                )
                if process.returncode:
                    raise RuntimeError(
                        f"Archive copy failed ({process.returncode}); see transfer.log"
                    )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
        phase("exporting")
        folder = args.job / "dataset"
        manifest = export(args.native, best, args.exporter, folder)
        phase("publishing", saved_frames=len(manifest["times"]))
        config = publish(folder, args.hf_repo)
        phase("deploying", dataset=config)
        space_commit = deploy(
            args.hf_repo,
            config=config,
            folder=snapshot,
            expected_revision=space_revision,
        )
        phase("checking_hosted_app", space_commit=space_commit)
        ready = False
        for _ in range(40):
            if HfApi().get_space_runtime(args.hf_repo).stage == "RUNNING":
                try:
                    client = Client(args.hf_repo, token=False, verbose=False)
                    result = client.predict(
                        len(manifest["times"]) - 1,
                        0,
                        0,
                        0,
                        "velocity",
                        api_name="/slice",
                    )
                    if hosted_frame_matches(result, manifest["times"]):
                        ready = True
                        break
                except (HTTPError, OSError, ValueError):
                    pass
            time.sleep(30)
        if not ready:
            raise RuntimeError(
                "Hosted full-history endpoint did not pass; main references unchanged"
            )
        site_record = json.loads(expected[REFERENCES[1]])
        site_record.update(
            complete_history=True,
            saved_frames=len(manifest["times"]),
            dataset=config,
            space_commit=space_commit,
            promoted_at=datetime.now(UTC).isoformat(),
        )
        replacement = {
            REFERENCES[0]: json.dumps(config, indent=2) + "\n",
            REFERENCES[1]: json.dumps(site_record, indent=2) + "\n",
        }
        phase("updating_main")
        main = update_references(args.github_repo, expected, replacement, code_shas)
        phase("complete", status="complete", main_commit=main)
    except (
        OSError,
        ValueError,
        RuntimeError,
        HTTPError,
        subprocess.SubprocessError,
    ) as error:
        phase("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--host", choices=["oliver"], required=True)
    parser.add_argument("--remote-source", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--hf-repo", default="ccm/navier-stokes-singularity-simulation")
    parser.add_argument(
        "--github-repo", default="cmccomb/navier-stokes-singularity-simulation"
    )
    parser.add_argument("--max-seconds", type=int, default=21600)
    args = parser.parse_args()
    for name in ("job", "native", "exporter", "identity"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)
