"""Parse sdf selection xlsx and flat naming helpers."""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class SelectionRow:
    index: int
    filename: str
    rel_path: str
    export_id: str


def _read_shared_strings(z: zipfile.ZipFile) -> List[str]:
    ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings: List[str] = []
    for si in ss.findall(".//m:si", NS):
        t = si.find("m:t", NS)
        if t is not None and t.text:
            strings.append(t.text)
        else:
            parts = [r.find("m:t", NS) for r in si.findall("m:r", NS)]
            strings.append("".join((x.text or "") for x in parts if x is not None))
    return strings


def _cell_value(cell: ET.Element, strings: List[str]) -> str:
    t = cell.get("t")
    v = cell.find("m:v", NS)
    if v is None or v.text is None:
        return ""
    val = v.text
    if t == "s":
        return strings[int(val)]
    return val


def load_selection_xlsx(path: Path) -> List[SelectionRow]:
    rows: List[SelectionRow] = []
    with zipfile.ZipFile(path) as z:
        strings = _read_shared_strings(z)
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//m:row", NS):
            cells = {c.get("r", "")[0]: c for c in row.findall("m:c", NS)}
            if "A" not in cells:
                continue
            a_val = _cell_value(cells["A"], strings)
            if a_val in ("序号", "index", "#", ""):
                continue
            try:
                idx = int(a_val)
            except ValueError:
                continue
            filename = _cell_value(cells.get("B", cells["A"]), strings) if "B" in cells else ""
            rel_path = _cell_value(cells["C"], strings) if "C" in cells else ""
            if not filename or not rel_path:
                continue
            rows.append(
                SelectionRow(
                    index=idx,
                    filename=filename,
                    rel_path=rel_path,
                    export_id=Path(filename).stem,
                )
            )
    return rows


def parse_dataset_id_from_rel_path(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) < 2 or parts[0] != "SDF_for_docking":
        raise ValueError(f"unexpected rel_path: {rel_path}")
    return parts[1]


def flat_name_for_row(row: SelectionRow) -> Tuple[str, str]:
    dataset_id = parse_dataset_id_from_rel_path(row.rel_path)
    return dataset_id, f"{dataset_id}__{row.filename}"
