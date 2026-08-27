"""Helpers for the two-row headers used in OSNACA workbooks."""

from __future__ import annotations

from typing import Iterable


def forward_fill(row: Iterable) -> list:
    """Fill merged-cell gaps with the previous heading."""
    filled = []
    last = None
    for cell in row:
        if cell not in (None, ""):
            last = cell
        filled.append(last)
    return filled


def flatten_headers(row1: Iterable, row2: Iterable) -> list[tuple[str, str]]:
    """Combine group and detail rows into one tuple per column."""
    filled = forward_fill(row1)
    return [(g or "", d or "") for g, d in zip(filled, row2)]


def header_label(group: str, detail: str) -> str:
    """Use the detailed label when present, otherwise use the group."""
    return (detail or group or "").strip()
