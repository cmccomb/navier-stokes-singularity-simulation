"""Deploy only reviewed app sources and an immutable public dataset reference."""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from spaces.native_explorer.reader import NativeDataset

ROOT = Path(__file__).resolve().parents[1]


def deploy(repo_id, config=None, folder=None, expected_revision=None):
    folder = folder or ROOT / "spaces/native_explorer"
    config = config or json.loads((folder / "dataset.json").read_text())
    dataset = NativeDataset(**config)
    dataset.sample_plane(len(dataset.times) - 1, "xz", 0)
    api = HfApi()
    if expected_revision and api.space_info(repo_id).sha != expected_revision:
        raise RuntimeError("Space changed during preparation; refusing to replace it")
    api.create_repo(
        repo_id,
        repo_type="space",
        private=False,
        exist_ok=True,
        space_sdk="gradio",
        space_hardware="cpu-basic",
    )
    paths = ["app.py", "reader.py", "requirements.txt", "README.md"]
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=folder / name)
        for name in paths
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=ROOT / "LICENSE")
    )
    operations.append(
        CommitOperationAdd(
            path_in_repo="dataset.json",
            path_or_fileobj=(json.dumps(config, indent=2) + "\n").encode(),
        )
    )
    result = api.create_commit(
        repo_id,
        repo_type="space",
        operations=operations,
        commit_message=f"Deploy native explorer: {len(dataset.times)} saved states",
    )
    print(
        json.dumps({"space": repo_id, "commit": result.oid, "dataset": config}),
        flush=True,
    )
    return result.oid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    deploy(parser.parse_args().repo)
