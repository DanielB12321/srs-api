"""
Two-row header flattening for OSNACA workbooks.

The first row holds element-group labels (some merged cells leave it blank
on continuation columns); the second row holds the analytical detail. We
forward-fill row 1 across blanks, then combine with row 2 to produce one
header per column.
"""
from typing import Iterable


def forward_fill(row: Iterable) -> list:
    """Carry the previous non-empty value across blanks. Used to undo merged-cell gaps."""
    filled = []
    last = None
    for cell in row:
        if cell not in (None, ""):
            last = cell
        filled.append(last)
    return filled


def flatten_headers(row1: Iterable, row2: Iterable) -> list[tuple[str, str]]:
    """
    Combine the two header rows into per-column (group, detail) tuples.

      row1: ['Sample', None, None, 'AU1',     'Pt', 'Pd', 'Au AR', ...]
      row2: ['Sample', 'Code', 'Code Tester', 'Au (FA)', 'Pt', 'Pd', 'Au (AR)', ...]
      result:
        [('Sample','Sample'), ('Sample','Code'), ('Sample','Code Tester'),
         ('AU1','Au (FA)'), ('Pt','Pt'), ('Pd','Pd'), ('Au AR','Au (AR)'), ...]
    """
    filled = forward_fill(row1)
    return [(g or "", d or "") for g, d in zip(filled, row2)]


def header_label(group: str, detail: str) -> str:
    """Pick the best single label for a column.  Detail takes priority; fall back to group."""
    return (detail or group or "").strip()
