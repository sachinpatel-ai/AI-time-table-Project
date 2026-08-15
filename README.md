# AI-Based Smart Timetable Generator

An intelligent scheduling system that automatically generates optimized,
conflict-free timetables for schools, colleges, and universities using
**Google OR-Tools' CP-SAT constraint solver**, wrapped in a **Streamlit**
dashboard.

Built as a modular, well-commented reference implementation suitable for a
final-year MCA project — every hard constraint (no double-booking, exact
subject hours, room capacity, teacher availability) is enforced structurally
by the solver, while soft constraints (even distribution, minimal gaps,
balanced teacher load) are optimized via a weighted objective function.

## Features

- **Dashboard** — live stats (teachers, subjects, rooms, classes) + dataset validation
- **Data Management** — upload/edit/delete CSV or Excel datasets with schema validation, in-app data editor
- **Timetable Generator** — one-click generation, adjustable constraint weights, progress indicator, optimization score, automatic conflict detection
- **Timetable Views** — department-wise, semester-wise, teacher-wise, classroom-wise, and daily views, with search/filter and a weekly grid layout
- **Analytics** — teacher workload, room utilization, subject distribution, weekly summary, conflict report (interactive Plotly charts)
- **Export** — CSV, Excel (multi-sheet), and printable PDF
- **History** — every saved run is stored and can be reloaded later
- **Light/Dark mode** toggle
- **Optimized** (AI constraint solver) or **Random** (quick feasible schedule) generation modes

## Project Structure

```
AI_Timetable_Generator/
├── app.py              # Streamlit UI - all pages
├── scheduler.py        # Orchestration: generation, views, analytics, history
├── optimizer.py         # CP-SAT (Google OR-Tools) constraint solver
├── constraints.py       # Constraint definitions & configuration
├── utils.py             # Data I/O, validation, export helpers (CSV/Excel/PDF)
├── datasets/            # Sample input CSVs (edit or replace with your own)
│   ├── subjects.csv
│   ├── teachers.csv
│   ├── rooms.csv
│   ├── timeslots.csv
│   ├── classes.csv
│   └── teacher_availability.csv
├── outputs/              # Generated timetables & history (created at runtime)
├── templates/            # Reserved for future document templates
├── assets/               # Reserved for static assets (logos, icons)
├── requirements.txt
└── README.md
```

## How It Works

1. **Datasets** describe subjects (with weekly hours + assigned faculty),
   teachers (with availability & daily lecture caps), rooms (capacity +
   type), timeslots (with `Lecture`/`Break`/`Lunch` markers), and classes
   (department/semester/section groups).
2. **optimizer.py** builds a boolean decision variable for every feasible
   `(class, subject, slot, room)` combination and encodes:
   - **Hard constraints**: no teacher/room/class double-booking, exact
     weekly-hour totals per subject, teacher availability, room capacity
     and type matching, and per-teacher daily lecture caps.
   - **Soft constraints** (combined into a single weighted objective that
     CP-SAT minimizes): even distribution of a subject's lectures across
     the week, avoiding back-to-back repeats of the same subject,
     minimizing free/gap periods, and balancing total load across teachers.
3. Break and lunch periods are never touched by the solver since they are
   excluded from the pool of schedulable slots — they're reserved
   automatically simply by construction.
4. **scheduler.py** turns the raw solution into the views, charts, exports,
   and history the UI needs.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL in your browser. Start on **Data
Management** to review/replace the sample datasets, then use **Timetable
Generator** to produce your first schedule.

## Customizing Constraints

Constraint weights and toggles are exposed directly in the **Timetable
Generator** page and are backed by `ConstraintConfig` in `constraints.py`.
Increase a soft-constraint weight to make the solver prioritize it more
strongly relative to the others; hard constraints can be selectively
relaxed (e.g. disable "lab-room matching" if your dataset has no labs).

## Notes on the Sample Data

The bundled `datasets/` folder contains a small illustrative dataset (two
CSE semesters + one ECE semester, ~18 subjects, 15 teachers, 7 rooms, a
6-day week) so the app works out of the box. Replace these files (or use
the in-app uploader/editor) with your institution's real data — the schema
each file must follow is documented in `utils.py::SCHEMAS` and enforced by
the validation step.
