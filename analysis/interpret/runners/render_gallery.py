"""HTML gallery helpers for interpret showcase outputs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def write_manifest(output_dir: Path, entries: Sequence[Mapping[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(list(entries), indent=2), encoding="utf-8")
    return manifest_path


def write_html_gallery(
    output_dir: Path,
    *,
    title: str,
    image_paths: Iterable[Path],
    captions: Iterable[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in image_paths]
    caps = list(captions) if captions is not None else [p.name for p in paths]
    if len(caps) != len(paths):
        raise ValueError("captions length must match image_paths")

    rel_paths = []
    for p in paths:
        try:
            rel_paths.append(p.relative_to(output_dir))
        except ValueError:
            rel_paths.append(Path(p.name))
    cards = []
    for rel, cap in zip(rel_paths, caps):
        cards.append(
            f"""<figure class="card">
  <img src="{html.escape(str(rel))}" alt="{html.escape(cap)}" loading="lazy" />
  <figcaption>{html.escape(cap)}</figcaption>
</figure>"""
        )

    body = "\n".join(cards)
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
    h1 {{ font-size: 1.25rem; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px; box-shadow: 0 1px 2px rgba(0,0,0,.06); }}
    .card img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ font-size: 12px; margin-top: 8px; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="grid">
{body}
  </div>
</body>
</html>
"""
    out_path = output_dir / "index.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
