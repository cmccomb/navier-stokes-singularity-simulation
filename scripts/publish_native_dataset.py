"""Publish checked, immutable HF releases; never upload native scratch or credentials."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

from spaces.native_explorer.reader import NativeDataset

ROOT = Path(__file__).resolve().parents[1]


def publish(folder, repo_id, allow_prototype=False):
    data = NativeDataset(folder=folder)
    if not data.manifest["complete_history"] and not allow_prototype:
        raise ValueError("Prototype publication requires explicit --allow-prototype")
    for i, frame in enumerate(data.manifest["frames"]):
        for array, expected in zip(data.arrays, frame["level_sha256"], strict=True):
            values = np.asarray(array[i])
            if hashlib.sha256(values.tobytes()).hexdigest() != expected:
                raise ValueError(f"Field digest changed at frame {i}")
    digest = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    prefix = f"releases/{digest[:16]}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder,
        path_in_repo=prefix,
        allow_patterns=["fields.zarr/**", "manifest.json", "source-best.json"],
        commit_message=f"Add verified native release: {len(data.times)} saved states",
    )
    config = {"repo_id": repo_id, "revision": result.oid, "prefix": prefix}
    # A public reader performs real remote range reads before this release is advertised.
    remote = NativeDataset(**config)
    for i in sorted({0, len(data.times) // 2, len(data.times) - 1}):
        for plane in ("xy", "xz", "yz"):
            for quantity in ("velocity", "force"):
                local_values, _ = data.sample_plane(i, plane, 0, quantity)
                remote_values, _ = remote.sample_plane(i, plane, 0, quantity)
                np.testing.assert_array_equal(local_values, remote_values)
    card = (ROOT / "datasets/native-voxels/README.md").read_text()
    state = (
        "complete history" if data.manifest["complete_history"] else "prototype subset"
    )
    card += (
        f"\n## Current verified release\n\n{len(data.times)} states, {state}; "
        f"t = {data.times[0]} to {data.times[-1]}.\n\n"
        f"- Dataset commit: `{result.oid}`\n- Release: `{prefix}`\n"
    )
    api.create_commit(
        repo_id,
        repo_type="dataset",
        commit_message="Document verified native release",
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
            CommitOperationAdd(
                path_in_repo="LICENSE", path_or_fileobj=ROOT / "LICENSE"
            ),
            CommitOperationAdd(
                path_in_repo="latest.json",
                path_or_fileobj=(json.dumps(config, indent=2) + "\n").encode(),
            ),
        ],
    )
    print("NS_HF_RELEASE " + json.dumps(config), flush=True)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--allow-prototype", action="store_true")
    args = parser.parse_args()
    publish(args.folder, args.repo, args.allow_prototype)
