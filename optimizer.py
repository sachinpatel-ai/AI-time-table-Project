"""
optimizer.py
------------
The AI optimization engine for the Smart Timetable Generator.

Implements timetable generation as a Constraint Satisfaction / Optimization
Problem solved with Google OR-Tools CP-SAT. See constraints.py for the full
list of hard and soft constraints this module enforces.

Design summary
--------------
For every (class, subject, lecture-slot, room) combination that is *feasible*
(room big enough, correct room type, teacher available) we create one
boolean decision variable ``x[c, subj, slot, room]`` meaning "this subject is
taught to this class in this slot, in this room". Hard constraints are
encoded directly as linear/boolean constraints on these variables. Soft
constraints are captured as penalty terms combined into a single objective
that CP-SAT minimises, so the returned timetable is not just *feasible* but
also *good* (well-spread subjects, few gaps, balanced teacher workload).

The module purposefully keeps all OR-Tools specific code in one place so the
rest of the app (scheduler.py, app.py) never has to know how the solving is
done - they just call ``TimetableOptimizer.solve()`` and get back a tidy
DataFrame plus a diagnostics report.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from constraints import ConstraintConfig, room_is_suitable


@dataclass
class OptimizationResult:
    """Everything the rest of the app needs after a solve attempt."""

    status: str  # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "NO_SOLUTION"
    timetable: pd.DataFrame
    conflict_report: List[str] = field(default_factory=list)
    optimization_score: float = 0.0
    solve_seconds: float = 0.0
    stats: Dict[str, int] = field(default_factory=dict)


class TimetableOptimizer:
    """Builds and solves the CP-SAT model for one timetable generation run."""

    def __init__(
        self,
        subjects: pd.DataFrame,
        teachers: pd.DataFrame,
        rooms: pd.DataFrame,
        timeslots: pd.DataFrame,
        classes: pd.DataFrame,
        teacher_availability: Optional[pd.DataFrame] = None,
        config: Optional[ConstraintConfig] = None,
        time_limit_seconds: float = 30.0,
        random_mode: bool = False,
    ) -> None:
        self.subjects = subjects.reset_index(drop=True)
        self.teachers = teachers.reset_index(drop=True)
        self.rooms = rooms.reset_index(drop=True)
        self.timeslots = timeslots.reset_index(drop=True)
        self.classes = classes.reset_index(drop=True)
        self.teacher_availability = (
            teacher_availability.reset_index(drop=True)
            if teacher_availability is not None
            else pd.DataFrame(columns=["FacultyID", "Day", "AvailableTimeSlots"])
        )
        self.config = config or ConstraintConfig()
        self.time_limit_seconds = time_limit_seconds
        self.random_mode = random_mode

        # Only lecture-type slots are schedulable; break/lunch slots are
        # reserved automatically simply by excluding them here (H8).
        self.lecture_slots = self.timeslots[self.timeslots["SlotType"] == "Lecture"].copy()
        self.lecture_slots.reset_index(drop=True, inplace=True)

        self._availability_lookup = self._build_availability_lookup()

    # ------------------------------------------------------------------
    # Availability lookup
    # ------------------------------------------------------------------
    def _build_availability_lookup(self) -> Dict[Tuple[str, str], bool]:
        """(FacultyID, Day) -> is the teacher available that day at all."""
        lookup: Dict[Tuple[str, str], bool] = {}
        for _, row in self.teacher_availability.iterrows():
            fid, day = str(row["FacultyID"]), str(row["Day"])
            slots = str(row.get("AvailableTimeSlots", "All")).strip().lower()
            lookup[(fid, day)] = slots != "none"
        return lookup

    def _teacher_available(self, faculty_id: str, day: str) -> bool:
        if not self.config.enforce_teacher_availability:
            return True
        # Default to available if no explicit record exists.
        return self._availability_lookup.get((faculty_id, day), True)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def solve(self) -> OptimizationResult:
        start = time.time()
        model = cp_model.CpModel()

        class_rows = list(self.classes.itertuples(index=False))
        n_slots = len(self.lecture_slots)

        # x[(class_i, subject_id, slot_i, room_id)] = BoolVar
        x: Dict[Tuple[int, str, int, str], cp_model.IntVar] = {}
        # Track, per (class, subject), which slot-vars exist -> for H4.
        class_subject_slotvars: Dict[Tuple[int, str], List[cp_model.IntVar]] = {}
        # Track occupancy helpers for soft constraints.
        class_day_subject_slotvars: Dict[Tuple[int, str, str], List[Tuple[int, cp_model.IntVar]]] = {}
        teacher_total_vars: Dict[str, List[cp_model.IntVar]] = {}

        subjects_by_dept_sem: Dict[Tuple[str, str], pd.DataFrame] = {
            key: grp for key, grp in self.subjects.groupby(["Department", "Semester"])
        }

        skipped_no_variables: List[str] = []

        for class_i, crow in enumerate(class_rows):
            dept, sem, section, strength = (
                str(crow.Department),
                crow.Semester,
                crow.Section,
                int(crow.StudentStrength),
            )
            subj_group = subjects_by_dept_sem.get((dept, sem))
            if subj_group is None or subj_group.empty:
                skipped_no_variables.append(f"{dept} Sem {sem} Sec {section}: no subjects found")
                continue

            for _, srow in subj_group.iterrows():
                subject_id = str(srow["SubjectID"])
                faculty_id = str(srow["FacultyID"])
                weekly_hours = int(srow["WeeklyHours"])

                # Precompute the suitable rooms for this subject once.
                suitable_rooms = [
                    r for _, r in self.rooms.iterrows()
                    if room_is_suitable(r, srow, strength)
                ] if (self.config.enforce_room_capacity or self.config.enforce_lab_rooms) else list(
                    self.rooms.iterrows()
                )
                if not suitable_rooms:
                    skipped_no_variables.append(
                        f"{subject_id} ({dept} Sem {sem} Sec {section}): no suitable room available"
                    )
                    continue

                created_any = False
                for slot_i, slot_row in self.lecture_slots.iterrows():
                    day = str(slot_row["Day"])
                    if not self._teacher_available(faculty_id, day):
                        continue
                    for room_row in suitable_rooms:
                        room_id = str(room_row["RoomNumber"])
                        var = model.NewBoolVar(
                            f"x_c{class_i}_{subject_id}_s{slot_i}_{room_id}"
                        )
                        x[(class_i, subject_id, slot_i, room_id)] = var
                        created_any = True

                        class_subject_slotvars.setdefault((class_i, subject_id), []).append(var)
                        class_day_subject_slotvars.setdefault((class_i, day, subject_id), []).append(
                            (slot_i, var)
                        )
                        teacher_total_vars.setdefault(faculty_id, []).append(var)

                if not created_any:
                    skipped_no_variables.append(
                        f"{subject_id} ({dept} Sem {sem} Sec {section}): no feasible slot/room/availability combination"
                    )
                    continue

                # H4: exact weekly hours for this subject/class.
                vars_for_subject = class_subject_slotvars.get((class_i, subject_id), [])
                if vars_for_subject:
                    model.Add(sum(vars_for_subject) == weekly_hours)

        # H1: teacher not double-booked in the same slot.
        for slot_i in range(n_slots):
            for faculty_id, srow_faculty in self._faculty_subject_pairs():
                pass  # placeholder replaced below with a direct grouping

        # Group variables by (faculty_id, slot_i) for H1, by (room_id, slot_i)
        # for H2, and by (class_i, slot_i) for H3, in a single pass.
        by_teacher_slot: Dict[Tuple[str, int], List[cp_model.IntVar]] = {}
        by_room_slot: Dict[Tuple[str, int], List[cp_model.IntVar]] = {}
        by_class_slot: Dict[Tuple[int, int], List[cp_model.IntVar]] = {}
        by_teacher_day: Dict[Tuple[str, str], List[cp_model.IntVar]] = {}

        subject_faculty_map = dict(zip(self.subjects["SubjectID"].astype(str), self.subjects["FacultyID"].astype(str)))
        slot_day_map = dict(zip(self.lecture_slots.index, self.lecture_slots["Day"].astype(str)))

        for (class_i, subject_id, slot_i, room_id), var in x.items():
            faculty_id = subject_faculty_map[subject_id]
            day = slot_day_map[slot_i]
            by_teacher_slot.setdefault((faculty_id, slot_i), []).append(var)
            by_room_slot.setdefault((room_id, slot_i), []).append(var)
            by_class_slot.setdefault((class_i, slot_i), []).append(var)
            by_teacher_day.setdefault((faculty_id, day), []).append(var)

        for vars_ in by_teacher_slot.values():
            if len(vars_) > 1:
                model.Add(sum(vars_) <= 1)
        for vars_ in by_room_slot.values():
            if len(vars_) > 1:
                model.Add(sum(vars_) <= 1)
        for vars_ in by_class_slot.values():
            if len(vars_) > 1:
                model.Add(sum(vars_) <= 1)

        # H9: max lectures/day per teacher.
        if self.config.enforce_max_lectures_per_day:
            max_lookup = dict(
                zip(self.teachers["FacultyID"].astype(str), self.teachers["MaxLecturesPerDay"])
            )
            for (faculty_id, day), vars_ in by_teacher_day.items():
                cap = int(max_lookup.get(faculty_id, len(vars_)))
                if vars_:
                    model.Add(sum(vars_) <= cap)

        # --------------------------------------------------------------
        # Soft constraints -> objective terms
        # --------------------------------------------------------------
        penalty_terms = []

        # S1: even distribution - penalise a subject appearing more than
        # once on the same day for the same class.
        for (class_i, day, subject_id), slot_vars in class_day_subject_slotvars.items():
            if len(slot_vars) > 1:
                only_vars = [v for _, v in slot_vars]
                extra = model.NewIntVar(0, len(only_vars), f"extra_{class_i}_{day}_{subject_id}")
                model.Add(extra >= sum(only_vars) - 1)
                penalty_terms.append((self.config.weight_even_distribution, extra))

        # S2: avoid consecutive lecture slots (same day, adjacent slot index)
        # for the same subject/class.
        lecture_slots_by_day: Dict[str, List[int]] = {}
        for idx, day in slot_day_map.items():
            lecture_slots_by_day.setdefault(day, []).append(idx)
        for day, idxs in lecture_slots_by_day.items():
            idxs.sort()

        for (class_i, day, subject_id), slot_vars in class_day_subject_slotvars.items():
            var_by_slot = {s: v for s, v in slot_vars}
            ordered = lecture_slots_by_day.get(day, [])
            for a, b in zip(ordered, ordered[1:]):
                if a in var_by_slot and b in var_by_slot:
                    consec = model.NewBoolVar(f"consec_{class_i}_{day}_{subject_id}_{a}_{b}")
                    model.AddMultiplicationEquality(consec, [var_by_slot[a], var_by_slot[b]])
                    penalty_terms.append((self.config.weight_avoid_consecutive_same_subject, consec))

        # S3: minimise free/gap periods - penalise a slot being empty when
        # both its neighbours (same day, same class) are occupied.
        class_day_occupied: Dict[Tuple[int, str, int], List[cp_model.IntVar]] = {}
        for (class_i, subject_id, slot_i, room_id), var in x.items():
            day = slot_day_map[slot_i]
            class_day_occupied.setdefault((class_i, day, slot_i), []).append(var)

        occ_bool: Dict[Tuple[int, str, int], cp_model.IntVar] = {}
        for (class_i, day, slot_i), vars_ in class_day_occupied.items():
            b = model.NewBoolVar(f"occ_{class_i}_{day}_{slot_i}")
            model.Add(sum(vars_) >= 1).OnlyEnforceIf(b)
            model.Add(sum(vars_) == 0).OnlyEnforceIf(b.Not())
            occ_bool[(class_i, day, slot_i)] = b

        for class_i, _ in enumerate(class_rows):
            for day, idxs in lecture_slots_by_day.items():
                idxs_sorted = sorted(idxs)
                for a, mid, b in zip(idxs_sorted, idxs_sorted[1:], idxs_sorted[2:]):
                    oa = occ_bool.get((class_i, day, a))
                    om = occ_bool.get((class_i, day, mid))
                    ob = occ_bool.get((class_i, day, b))
                    if oa is not None and om is not None and ob is not None:
                        gap = model.NewBoolVar(f"gap_{class_i}_{day}_{mid}")
                        model.AddBoolAnd([oa, om.Not(), ob]).OnlyEnforceIf(gap)
                        model.AddBoolOr([oa.Not(), om, ob.Not()]).OnlyEnforceIf(gap.Not())
                        penalty_terms.append((self.config.weight_minimize_gaps, gap))

        # S4: balance teacher workload (minimise spread between busiest and
        # least-busy teacher).
        teacher_ids = list(teacher_total_vars.keys())
        if teacher_ids:
            max_load = model.NewIntVar(0, n_slots, "max_load")
            min_load = model.NewIntVar(0, n_slots, "min_load")
            for fid in teacher_ids:
                total = model.NewIntVar(0, n_slots, f"load_{fid}")
                model.Add(total == sum(teacher_total_vars[fid]))
                model.Add(max_load >= total)
                model.Add(min_load <= total)
            spread = model.NewIntVar(0, n_slots, "load_spread")
            model.Add(spread == max_load - min_load)
            penalty_terms.append((self.config.weight_balance_teacher_load, spread))

        if penalty_terms and not self.random_mode:
            model.Minimize(sum(w * v for w, v in penalty_terms))

        # --------------------------------------------------------------
        # Solve
        # --------------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = 8
        if self.random_mode:
            solver.parameters.randomize_search = True
            solver.parameters.random_seed = int(time.time()) % 100000

        status = solver.Solve(model)
        solve_seconds = time.time() - start

        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return OptimizationResult(
                status="INFEASIBLE" if status == cp_model.INFEASIBLE else "NO_SOLUTION",
                timetable=pd.DataFrame(),
                conflict_report=skipped_no_variables
                or ["Solver could not find a feasible timetable with the current data/constraints."],
                optimization_score=0.0,
                solve_seconds=solve_seconds,
                stats={"num_variables": len(x)},
            )

        # --------------------------------------------------------------
        # Extract solution into a tidy DataFrame
        # --------------------------------------------------------------
        rows = []
        subject_lookup = self.subjects.set_index("SubjectID").to_dict("index")
        teacher_lookup = self.teachers.set_index("FacultyID").to_dict("index")

        for (class_i, subject_id, slot_i, room_id), var in x.items():
            if solver.Value(var):
                crow = class_rows[class_i]
                srow = subject_lookup[subject_id]
                trow = teacher_lookup.get(srow["FacultyID"], {})
                slot_row = self.lecture_slots.loc[slot_i]
                rows.append(
                    {
                        "Department": crow.Department,
                        "Semester": crow.Semester,
                        "Section": crow.Section,
                        "SubjectID": subject_id,
                        "SubjectName": srow["SubjectName"],
                        "FacultyID": srow["FacultyID"],
                        "TeacherName": trow.get("TeacherName", srow["FacultyID"]),
                        "Day": slot_row["Day"],
                        "StartTime": slot_row["StartTime"],
                        "EndTime": slot_row["EndTime"],
                        "RoomNumber": room_id,
                    }
                )

        timetable_df = pd.DataFrame(rows)
        if not timetable_df.empty:
            from utils import DAY_ORDER

            timetable_df["_day_order"] = timetable_df["Day"].apply(
                lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99
            )
            timetable_df.sort_values(
                ["Department", "Semester", "Section", "_day_order", "StartTime"], inplace=True
            )
            timetable_df.drop(columns="_day_order", inplace=True)
            timetable_df.reset_index(drop=True, inplace=True)

        conflicts = self._detect_conflicts(timetable_df)
        score = self._compute_score(solver, status, penalty_terms)

        return OptimizationResult(
            status=status_name,
            timetable=timetable_df,
            conflict_report=conflicts + skipped_no_variables,
            optimization_score=score,
            solve_seconds=solve_seconds,
            stats={
                "num_variables": len(x),
                "num_lectures_scheduled": len(timetable_df),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _faculty_subject_pairs(self):
        return list(zip(self.subjects["FacultyID"], self.subjects.itertuples(index=False)))

    @staticmethod
    def _detect_conflicts(timetable_df: pd.DataFrame) -> List[str]:
        """Defensive double-check that hard constraints truly hold.

        Since the constraints are enforced structurally by the CP-SAT model,
        this should always return an empty list - it exists as a safety net
        and for the UI's "Detect conflicts automatically" feature.
        """
        conflicts: List[str] = []
        if timetable_df.empty:
            return conflicts

        dupe_teacher = timetable_df.duplicated(subset=["TeacherName", "Day", "StartTime"], keep=False)
        for _, row in timetable_df[dupe_teacher].iterrows():
            conflicts.append(
                f"Teacher conflict: {row['TeacherName']} double-booked on {row['Day']} {row['StartTime']}"
            )

        dupe_room = timetable_df.duplicated(subset=["RoomNumber", "Day", "StartTime"], keep=False)
        for _, row in timetable_df[dupe_room].iterrows():
            conflicts.append(
                f"Room conflict: {row['RoomNumber']} double-booked on {row['Day']} {row['StartTime']}"
            )

        dupe_class = timetable_df.duplicated(
            subset=["Department", "Semester", "Section", "Day", "StartTime"], keep=False
        )
        for _, row in timetable_df[dupe_class].iterrows():
            conflicts.append(
                f"Class conflict: {row['Department']} Sem{row['Semester']} {row['Section']} "
                f"double-booked on {row['Day']} {row['StartTime']}"
            )

        return sorted(set(conflicts))

    @staticmethod
    def _compute_score(solver: cp_model.CpSolver, status: int, penalty_terms) -> float:
        """Convert the objective value into a friendly 0-100 optimization score."""
        if not penalty_terms:
            return 100.0
        achieved_penalty = solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
        if achieved_penalty is None:
            return 0.0
        max_possible_penalty = sum(w for w, _ in penalty_terms) or 1
        score = max(0.0, 100.0 * (1 - achieved_penalty / max_possible_penalty))
        return round(score, 1)
