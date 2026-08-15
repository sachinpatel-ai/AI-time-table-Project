"""
scheduler.py
------------
Orchestration layer that sits between the raw datasets, the optimizer, and
the Streamlit UI (app.py).

Responsibilities:
    * Running a timetable generation (optimized or random) via optimizer.py
    * Slicing the resulting timetable into the different views the UI needs
      (department-wise, teacher-wise, room-wise, semester-wise, daily)
    * Computing analytics (teacher workload, room utilization, subject
      distribution) used by the charts
    * Saving/loading timetable history to disk so past runs aren't lost
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from constraints import ConstraintConfig
from optimizer import OptimizationResult, TimetableOptimizer
from utils import DAY_ORDER, ensure_dirs

HISTORY_DIR = "outputs/history"


@dataclass
class GenerationRequest:
    """Bundles everything needed to kick off one generation run."""

    subjects: pd.DataFrame
    teachers: pd.DataFrame
    rooms: pd.DataFrame
    timeslots: pd.DataFrame
    classes: pd.DataFrame
    teacher_availability: pd.DataFrame
    config: ConstraintConfig
    time_limit_seconds: float = 30.0
    random_mode: bool = False


class TimetableScheduler:
    """High-level API used by the Streamlit app to generate and inspect timetables."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir
        ensure_dirs(self.output_dir, os.path.join(self.output_dir, "history"))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, request: GenerationRequest) -> OptimizationResult:
        optimizer = TimetableOptimizer(
            subjects=request.subjects,
            teachers=request.teachers,
            rooms=request.rooms,
            timeslots=request.timeslots,
            classes=request.classes,
            teacher_availability=request.teacher_availability,
            config=request.config,
            time_limit_seconds=request.time_limit_seconds,
            random_mode=request.random_mode,
        )
        result = optimizer.solve()
        return result

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @staticmethod
    def department_view(timetable: pd.DataFrame, department: str) -> pd.DataFrame:
        return timetable[timetable["Department"] == department].copy()

    @staticmethod
    def semester_view(timetable: pd.DataFrame, department: str, semester) -> pd.DataFrame:
        return timetable[
            (timetable["Department"] == department) & (timetable["Semester"] == semester)
        ].copy()

    @staticmethod
    def teacher_view(timetable: pd.DataFrame, teacher_name: str) -> pd.DataFrame:
        return timetable[timetable["TeacherName"] == teacher_name].copy()

    @staticmethod
    def room_view(timetable: pd.DataFrame, room_number: str) -> pd.DataFrame:
        return timetable[timetable["RoomNumber"] == room_number].copy()

    @staticmethod
    def daily_view(timetable: pd.DataFrame, day: str) -> pd.DataFrame:
        return timetable[timetable["Day"] == day].copy()

    @staticmethod
    def as_grid(timetable: pd.DataFrame) -> pd.DataFrame:
        """Pivot a (single class/teacher/room) timetable into a Day x Time grid."""
        if timetable.empty:
            return pd.DataFrame()
        grid = timetable.copy()
        grid["Slot"] = grid["StartTime"] + "-" + grid["EndTime"]
        grid["Label"] = grid["SubjectName"] + "\n" + grid.get("RoomNumber", "")
        pivot = grid.pivot_table(
            index="Slot", columns="Day", values="Label", aggfunc=lambda x: " / ".join(x)
        )
        ordered_days = [d for d in DAY_ORDER if d in pivot.columns]
        pivot = pivot.reindex(columns=ordered_days)
        pivot = pivot.reindex(sorted(pivot.index, key=lambda s: s.split("-")[0]))
        return pivot

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    @staticmethod
    def teacher_workload(timetable: pd.DataFrame) -> pd.DataFrame:
        if timetable.empty:
            return pd.DataFrame(columns=["TeacherName", "LecturesPerWeek"])
        return (
            timetable.groupby("TeacherName")
            .size()
            .reset_index(name="LecturesPerWeek")
            .sort_values("LecturesPerWeek", ascending=False)
        )

    @staticmethod
    def room_utilization(timetable: pd.DataFrame, total_available_slots: int) -> pd.DataFrame:
        if timetable.empty:
            return pd.DataFrame(columns=["RoomNumber", "LecturesScheduled", "UtilizationPercent"])
        util = timetable.groupby("RoomNumber").size().reset_index(name="LecturesScheduled")
        util["UtilizationPercent"] = (
            util["LecturesScheduled"] / max(total_available_slots, 1) * 100
        ).round(1)
        return util.sort_values("UtilizationPercent", ascending=False)

    @staticmethod
    def subject_distribution(timetable: pd.DataFrame) -> pd.DataFrame:
        if timetable.empty:
            return pd.DataFrame(columns=["SubjectName", "LecturesPerWeek"])
        return (
            timetable.groupby("SubjectName")
            .size()
            .reset_index(name="LecturesPerWeek")
            .sort_values("LecturesPerWeek", ascending=False)
        )

    @staticmethod
    def weekly_summary(timetable: pd.DataFrame) -> pd.DataFrame:
        if timetable.empty:
            return pd.DataFrame(columns=["Day", "LecturesScheduled"])
        summary = timetable.groupby("Day").size().reset_index(name="LecturesScheduled")
        summary["_order"] = summary["Day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
        summary = summary.sort_values("_order").drop(columns="_order")
        return summary

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def save_to_history(self, result: OptimizationResult, label: Optional[str] = None) -> str:
        ensure_dirs(os.path.join(self.output_dir, "history"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = label or f"timetable_{timestamp}"
        csv_path = os.path.join(self.output_dir, "history", f"{name}.csv")
        meta_path = os.path.join(self.output_dir, "history", f"{name}.json")

        result.timetable.to_csv(csv_path, index=False)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "status": result.status,
                    "optimization_score": result.optimization_score,
                    "solve_seconds": result.solve_seconds,
                    "num_lectures": len(result.timetable),
                    "generated_at": timestamp,
                },
                f,
                indent=2,
            )
        return csv_path

    def list_history(self) -> List[Dict]:
        ensure_dirs(os.path.join(self.output_dir, "history"))
        entries = []
        for meta_path in sorted(glob.glob(os.path.join(self.output_dir, "history", "*.json")), reverse=True):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["name"] = os.path.splitext(os.path.basename(meta_path))[0]
            meta["csv_path"] = os.path.join(self.output_dir, "history", meta["name"] + ".csv")
            entries.append(meta)
        return entries

    def load_history_entry(self, csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path)
