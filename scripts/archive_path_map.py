"""Resolve immutable archive paths after a byte-preserving machine transfer."""

from __future__ import annotations

import json
from pathlib import Path


def load_path_map(path: Path | None) -> dict[Path, Path]:
    """Load absolute source-to-local-prefix mappings from a small JSON record."""
    if path is None:
        return {}
    record = json.loads(path.resolve(strict=True).read_text())
    if record.get("schema_version") != 1 or not isinstance(
        record.get("mappings"), list
    ):
        raise ValueError("archive path map has the wrong schema")
    mappings: dict[Path, Path] = {}
    for item in record["mappings"]:
        if not isinstance(item, dict) or set(item) != {"source", "target"}:
            raise ValueError("archive path map entries require source and target")
        source, target = Path(item["source"]), Path(item["target"])
        if not source.is_absolute() or not target.is_absolute() or source in mappings:
            raise ValueError("archive path map prefixes must be unique absolute paths")
        mappings[source] = target.resolve(strict=True)
    return mappings


def resolve_archive_path(value: str | Path, mappings: dict[Path, Path]) -> Path:
    """Resolve a recorded path locally, applying the longest matching prefix."""
    recorded = Path(value)
    for source in sorted(mappings, key=lambda item: len(item.parts), reverse=True):
        try:
            relative = recorded.relative_to(source)
        except ValueError:
            continue
        return (mappings[source] / relative).resolve(strict=True)
    return recorded.resolve(strict=True)


def path_map_record(mappings: dict[Path, Path]) -> list[dict[str, str]]:
    """Return a deterministic, JSON-safe description for provenance records."""
    return [
        {"source": str(source), "target": str(mappings[source])}
        for source in sorted(mappings, key=str)
    ]
