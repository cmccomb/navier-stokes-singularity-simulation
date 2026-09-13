"""Offline promotion checks: no credentials, network, or branch changes."""

import hashlib

import pytest

pytest.importorskip("gradio_client")
pytest.importorskip("zarr")
release = pytest.importorskip("scripts.finish_native_release")


def blob(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


@pytest.mark.parametrize("conflict", [None, "reference", "source", "unrelated"])
def test_only_expected_records_can_be_promoted(monkeypatch, conflict):
    expected = dict.fromkeys(release.REFERENCES, b"old\n")
    replacement = dict.fromkeys(release.REFERENCES, "new\n")
    source = "spaces/native_explorer/app.py"
    rows = [
        {"path": path, "sha": blob(content), "type": "blob", "mode": "100644"}
        for path, content in {
            **expected,
            source: b"app",
            "site/media/retained.gif": b"untouched",
        }.items()
    ]
    if conflict == "reference":
        rows[0]["sha"] = "concurrent"
    if conflict == "source":
        rows[2]["sha"] = "concurrent"
    promoted = []

    def fake_gh(repo, path, data=None, method=None):
        if path == "git/ref/heads/main":
            return {"object": {"sha": "old-head"}}
        if path == "git/commits/old-head":
            return {"tree": {"sha": "old-tree"}}
        if path == "git/trees/old-tree?recursive=1":
            return {"tree": rows}
        if path == "git/trees":
            assert data["base_tree"] == "old-tree"
            assert {r["path"] for r in data["tree"]} == set(release.REFERENCES)
            return {"sha": "new-tree"}
        if path == "git/trees/new-tree?recursive=1":
            changed = [dict(row) for row in rows]
            if conflict == "unrelated":
                changed[-1]["sha"] = "changed-unrelated-media"
            return {"tree": changed}
        if path == "git/commits":
            assert data["parents"] == ["old-head"]
            return {"sha": "new-head"}
        if path == "git/refs/heads/main":
            assert method == "PATCH" and data == {"sha": "new-head", "force": False}
            promoted.append(data)
            return {}
        raise AssertionError(path)

    monkeypatch.setattr(release, "gh", fake_gh)
    if conflict:
        with pytest.raises(RuntimeError):
            release.update_references(
                "example/repo", expected, replacement, {source: blob(b"app")}
            )
        assert not promoted
    else:
        assert (
            release.update_references(
                "example/repo", expected, replacement, {source: blob(b"app")}
            )
            == "new-head"
        )
        assert len(promoted) == 1
