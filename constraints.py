"""
constraints.py
--------------
Defines the scheduling constraints used by optimizer.py.

Hard constraints (must always hold in any valid timetable):
    H1. A teacher cannot teach two classes in the same time slot.
    H2. A room cannot host two classes in the same time slot.
    H3. A class/section cannot attend two subjects in the same time slot.
    H4. Each subject must be scheduled exactly its required weekly hours.
    H5. A lecture can only be placed in a slot the teacher is available in.
    H6. A room's capacity must be >= the class's student strength.
    H7. Lab subjects must be scheduled in Lab-type rooms.
    H8. No lecture may be placed in a Break/Lunch slot.
    H9. A teacher cannot exceed their MaxLecturesPerDay.

Soft constraints (rewarded/penalised, traded off in the objective):
    S1. Distribute a subject's lectures evenly across the week
        (avoid piling up a subject on one or two days).
    S2. Avoid scheduling the same subject in consecutive slots on the same day.
    S3. Minimise idle/free periods for a class between its first and last
        lecture of the day.
    S4. Balance total weekly workload across teachers.

This module centralises the definitions so optimizer.py stays focused on
the CP-SAT model wiring, and so the constraint list is easy to audit or
extend without touching solver internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd


@dataclass
class ConstraintConfig:
    """Toggle and weight settings for the optimizer.

    Hard constraints are always enforced (they can only be turned off if the
    dataset genuinely has no such notion, e.g. no lab subjects). Soft
    constraint weights control how strongly the objective function penalises
    violations - higher weight = more strongly discouraged.
    """

    enforce_room_capacity: bool = True
    enforce_lab_rooms: bool = True
    enforce_teacher_availability: bool = True
    enforce_max_lectures_per_day: bool = True

    weight_even_distribution: int = 5
    weight_avoid_consecutive_same_subject: int = 3
    weight_minimize_gaps: int = 4
    weight_balance_teacher_load: int = 2


def build_teacher_availability_map(
    availability_df: pd.DataFrame, days: List[str]
) -> Dict[str, Set[str]]:
    """Return {faculty_id: {available days}} treating 'All' as every day.

    The sample dataset expresses availability per-day with an
    AvailableTimeSlots value of 'All' or 'None'; per-slot availability text
    (e.g. "09:00-09:55;11:05-12:00") is also supported and handled by
    optimizer.py when finer-grained checks are needed.
    """
    availability: Dict[str, Set[str]] = {}
    if availability_df is None or availability_df.empty:
        return availability

    for _, row in availability_df.iterrows():
        fid = str(row["FacultyID"])
        day = str(row["Day"])
        slots = str(row.get("AvailableTimeSlots", "All")).strip()
        availability.setdefault(fid, set())
        if slots.lower() == "none":
            continue
        availability[fid].add(day)
    return availability


def room_is_suitable(room_row: pd.Series, subject_row: pd.Series, class_strength: int) -> bool:
    """Check room capacity + room-type constraints (H6, H7)."""
    if int(room_row["Capacity"]) < int(class_strength):
        return False
    subject_name = str(subject_row.get("SubjectName", "")).lower()
    is_lab_subject = "lab" in subject_name
    is_lab_room = str(room_row.get("RoomType", "")).strip().lower() == "lab"
    if is_lab_subject and not is_lab_room:
        return False
    if is_lab_room and not is_lab_subject:
        # Keep pure lecture subjects out of lab rooms when a normal
        # classroom is available; not a hard rule, but avoids wasting labs.
        return False
    return True


def get_lecture_slot_indices(timeslots_df: pd.DataFrame) -> List[int]:
    """Row indices of timeslots usable for lectures (excludes Break/Lunch)."""
    return list(timeslots_df.index[timeslots_df["SlotType"] == "Lecture"])


def describe_constraints(config: ConstraintConfig) -> List[str]:
    """Human-readable summary of active constraints, shown in the UI."""
    lines = [
        "No teacher double-booked across classes (hard)",
        "No room double-booked across classes (hard)",
        "No class double-booked across subjects (hard)",
        "Each subject scheduled its exact weekly hours (hard)",
        "Lectures only placed in Lecture-type slots (hard)",
    ]
    if config.enforce_teacher_availability:
        lines.append("Teacher availability respected (hard)")
    if config.enforce_max_lectures_per_day:
        lines.append("Teacher daily lecture cap respected (hard)")
    if config.enforce_room_capacity:
        lines.append("Room capacity >= class strength (hard)")
    if config.enforce_lab_rooms:
        lines.append("Lab subjects placed in lab rooms (hard)")
    lines.append("Even distribution of subject hours across the week (soft)")
    lines.append("Avoid consecutive lectures of the same subject (soft)")
    lines.append("Minimise free/gap periods per class per day (soft)")
    lines.append("Balance total weekly load across teachers (soft)")
    return lines
