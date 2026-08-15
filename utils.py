"""
utils.py
--------
Shared utility functions for the AI-Based Smart Timetable Generator.

Responsibilities:
    * Loading and saving CSV/Excel datasets
    * Validating uploaded data against the expected schema
    * Small formatting / helper functions reused across the app

Keeping these helpers in one module avoids duplicating I/O and
validation logic between app.py, scheduler.py and optimizer.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Schema definitions - the single source of truth for expected columns.
# Used both for validation and for generating helpful error messages.
# --------------------------------------------------------------------------
SCHEMAS: Dict[str, List[str]] = {
    "subjects": ["SubjectID", "SubjectName", "WeeklyHours", "FacultyID", "Department", "Semester"],
    "teachers": ["FacultyID", "TeacherName", "Department", "MaxLecturesPerDay", "PreferredTimeSlots"],
    "rooms": ["RoomNumber", "Capacity", "RoomType"],
    "timeslots": ["Day", "StartTime", "EndTime", "SlotType"],
    "classes": ["Department", "Semester", "Section", "StudentStrength"],
    "teacher_availability": ["FacultyID", "Day", "AvailableTimeSlots"],
}

DATASET_FILENAMES: Dict[str, str] = {
    "subjects": "subjects.csv",
    "teachers": "teachers.csv",
    "rooms": "rooms.csv",
    "timeslots": "timeslots.csv",
    "classes": "classes.csv",
    "teacher_availability": "teacher_availability.csv",
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class ValidationResult:
    """Container describing whether a dataframe matches its expected schema."""

    name: str
    ok: bool
    missing_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_csv_or_excel(file_or_path) -> pd.DataFrame:
    """Load a dataset from a path or a file-like object (Streamlit upload).

    Supports .csv, .xlsx and .xls transparently by inspecting the filename.
    """
    name = getattr(file_or_path, "name", str(file_or_path))
    ext = os.path.splitext(name)[1].lower()

    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        raise ValueError(f"Could not read '{name}': {exc}") from exc

    # Normalise column names (strip whitespace) so minor formatting
    # differences in uploaded files don't break validation.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Persist a dataframe back to disk as CSV, creating folders as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_all_datasets(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """Load every known dataset from a folder, skipping any that are missing."""
    datasets: Dict[str, pd.DataFrame] = {}
    for key, filename in DATASET_FILENAMES.items():
        path = os.path.join(dataset_dir, filename)
        if os.path.exists(path):
            datasets[key] = load_csv_or_excel(path)
    return datasets


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_dataset(name: str, df: pd.DataFrame) -> ValidationResult:
    """Validate a single dataframe against its expected schema.

    Checks for missing required columns and flags a handful of common
    data-quality issues (empty required fields, non-numeric hour/capacity
    columns, duplicate IDs) as warnings rather than hard failures so the
    user can still preview the data before fixing it.
    """
    expected = SCHEMAS.get(name, [])
    missing = [c for c in expected if c not in df.columns]
    warnings: List[str] = []

    if not missing:
        # Numeric sanity checks
        numeric_cols = {
            "subjects": ["WeeklyHours"],
            "teachers": ["MaxLecturesPerDay"],
            "rooms": ["Capacity"],
            "classes": ["StudentStrength"],
        }.get(name, [])
        for col in numeric_cols:
            if col in df.columns:
                non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
                if non_numeric:
                    warnings.append(f"{non_numeric} row(s) have a non-numeric '{col}' value.")

        # Duplicate ID checks
        id_cols = {
            "subjects": "SubjectID",
            "teachers": "FacultyID",
            "rooms": "RoomNumber",
        }.get(name)
        if id_cols and id_cols in df.columns:
            dupes = df[id_cols].duplicated().sum()
            if dupes:
                warnings.append(f"{dupes} duplicate value(s) found in '{id_cols}'.")

        if df.empty:
            warnings.append("Dataset has no rows.")

    return ValidationResult(
        name=name,
        ok=not missing,
        missing_columns=missing,
        warnings=warnings,
        row_count=len(df),
    )


def validate_all(datasets: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
    return {name: validate_dataset(name, df) for name, df in datasets.items()}


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------
def slot_label(day: str, start: str, end: str) -> str:
    return f"{day} {start}-{end}"


def sort_days(days: List[str]) -> List[str]:
    return sorted(set(days), key=lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


# --------------------------------------------------------------------------
# Export helpers (CSV / Excel / PDF)
# --------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write one or more dataframes to an in-memory .xlsx file, one sheet each."""
    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31] if sheet_name else "Sheet1"
            (df if not df.empty else pd.DataFrame({"Info": ["No data"]})).to_excel(
                writer, sheet_name=safe_name, index=False
            )
    return buffer.getvalue()


def to_pdf_bytes(title: str, df: pd.DataFrame, subtitle: str = "") -> bytes:
    """Render a dataframe as a simple, printable PDF table using reportlab."""
    import io

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Title"])]
    if subtitle:
        elements.append(Paragraph(subtitle, styles["Normal"]))
    elements.append(Spacer(1, 12))

    if df.empty:
        elements.append(Paragraph("No data available.", styles["Normal"]))
    else:
        data = [list(df.columns)] + df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4374")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elements.append(table)

    doc.build(elements)
    return buffer.getvalue()
