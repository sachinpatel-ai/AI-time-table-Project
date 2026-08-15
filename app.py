"""
app.py
------
AI-Based Smart Timetable Generator - Streamlit front-end.

Run with:
    streamlit run app.py

This file wires together utils.py (data I/O + validation), scheduler.py
(orchestration + views + analytics), optimizer.py (the CP-SAT engine) and
constraints.py (constraint configuration) into a multi-page dashboard.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from constraints import ConstraintConfig, describe_constraints
from scheduler import GenerationRequest, TimetableScheduler
from utils import (
    DATASET_FILENAMES,
    DAY_ORDER,
    load_all_datasets,
    load_csv_or_excel,
    save_dataframe,
    to_csv_bytes,
    to_excel_bytes,
    to_pdf_bytes,
    validate_all,
    validate_dataset,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

st.set_page_config(
    page_title="AI Smart Timetable Generator",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------
def init_state():
    if "datasets" not in st.session_state:
        st.session_state.datasets = load_all_datasets(DATASET_DIR)
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "scheduler" not in st.session_state:
        st.session_state.scheduler = TimetableScheduler(output_dir=OUTPUT_DIR)


init_state()


# --------------------------------------------------------------------------
# Theming (dark / light mode via CSS injection)
# --------------------------------------------------------------------------
def inject_theme_css(theme: str):
    if theme == "Dark":
        css = """
        <style>
        .stApp { background-color: #0E1117; color: #E6E6E6; }
        section[data-testid="stSidebar"] { background-color: #161A23; }
        .metric-card {
            background: linear-gradient(135deg, #1B2436 0%, #232E45 100%);
            border-radius: 12px; padding: 18px; border: 1px solid #2C3A55;
        }
        .metric-card h2 { color: #7FB2FF; margin: 0; }
        .metric-card p { color: #A9B4C9; margin: 0; font-size: 0.85rem; }
        </style>
        """
    else:
        css = """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #F4F7FF 0%, #EAF0FF 100%);
            border-radius: 12px; padding: 18px; border: 1px solid #D8E1FA;
        }
        .metric-card h2 { color: #2E4374; margin: 0; }
        .metric-card p { color: #5B6B8C; margin: 0; font-size: 0.85rem; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


inject_theme_css(st.session_state.theme)


def metric_card(col, label, value):
    col.markdown(
        f"""<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🗓️ Smart Timetable")
st.sidebar.caption("AI-Based Timetable Generator")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Data Management",
        "Timetable Generator",
        "Timetable Views",
        "Analytics",
        "Export",
        "History",
    ],
)

st.sidebar.divider()
st.session_state.theme = st.sidebar.radio("Appearance", ["Light", "Dark"], horizontal=True,
                                           index=["Light", "Dark"].index(st.session_state.theme))
inject_theme_css(st.session_state.theme)

st.sidebar.divider()
st.sidebar.caption("Final-Year MCA Project · Built with Streamlit + Google OR-Tools")

# ==========================================================================
# PAGE: Dashboard
# ==========================================================================
if page == "Dashboard":
    st.title("🗓️ AI-Based Smart Timetable Generator")
    st.write(
        "Automatically generate conflict-free, optimized timetables for schools, "
        "colleges, or universities using constraint programming (Google OR-Tools)."
    )

    datasets = st.session_state.datasets
    n_teachers = len(datasets.get("teachers", pd.DataFrame()))
    n_subjects = len(datasets.get("subjects", pd.DataFrame()))
    n_rooms = len(datasets.get("rooms", pd.DataFrame()))
    n_classes = len(datasets.get("classes", pd.DataFrame()))

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Teachers", n_teachers)
    metric_card(c2, "Total Subjects", n_subjects)
    metric_card(c3, "Total Classrooms", n_rooms)
    metric_card(c4, "Total Classes", n_classes)

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Project Overview")
        st.markdown(
            """
            This system replaces manual, error-prone scheduling with an
            AI-driven optimizer that:
            - Assigns every subject its exact required weekly hours
            - Never double-books a teacher, room, or class
            - Respects teacher availability, room capacity and room type
            - Spreads lectures evenly and minimizes free periods
            - Automatically reserves lunch and break slots
            - Balances workload across teachers

            Use **Data Management** to upload/edit your datasets, then head to
            **Timetable Generator** to produce a new schedule with one click.
            """
        )

    with right:
        st.subheader("Current Dataset Validation")
        results = validate_all(datasets)
        for name, res in results.items():
            icon = "✅" if res.ok and not res.warnings else ("⚠️" if res.ok else "❌")
            st.write(f"{icon} **{name}** — {res.row_count} rows")
            if res.missing_columns:
                st.caption(f"Missing columns: {', '.join(res.missing_columns)}")
            for w in res.warnings:
                st.caption(f"⚠️ {w}")

    if st.session_state.last_result is not None:
        st.divider()
        st.subheader("Last Generated Timetable")
        res = st.session_state.last_result
        c1, c2, c3 = st.columns(3)
        metric_card(c1, "Status", res.status)
        metric_card(c2, "Optimization Score", f"{res.optimization_score}%")
        metric_card(c3, "Lectures Scheduled", len(res.timetable))

# ==========================================================================
# PAGE: Data Management
# ==========================================================================
elif page == "Data Management":
    st.title("📂 Data Management")
    st.write("Upload, preview, validate, and edit each dataset the optimizer needs.")

    dataset_names = list(DATASET_FILENAMES.keys())
    tabs = st.tabs([n.replace("_", " ").title() for n in dataset_names])

    for tab, name in zip(tabs, dataset_names):
        with tab:
            filename = DATASET_FILENAMES[name]
            st.caption(f"Expected file: `{filename}`")

            uploaded = st.file_uploader(
                f"Upload {name}.csv/.xlsx", type=["csv", "xlsx", "xls"], key=f"upload_{name}"
            )
            if uploaded is not None:
                try:
                    new_df = load_csv_or_excel(uploaded)
                    st.session_state.datasets[name] = new_df
                    st.success(f"Loaded {len(new_df)} rows for {name}.")
                except ValueError as e:
                    st.error(str(e))

            df = st.session_state.datasets.get(name, pd.DataFrame())
            validation = validate_dataset(name, df)

            if validation.missing_columns:
                st.error(f"Missing required columns: {', '.join(validation.missing_columns)}")
            for w in validation.warnings:
                st.warning(w)

            st.write(f"**Preview & edit** ({len(df)} rows)")
            edited = st.data_editor(
                df, num_rows="dynamic", use_container_width=True, key=f"editor_{name}"
            )
            col_a, col_b, col_c = st.columns([1, 1, 3])
            if col_a.button("💾 Save changes", key=f"save_{name}"):
                st.session_state.datasets[name] = edited
                save_dataframe(edited, os.path.join(DATASET_DIR, filename))
                st.success(f"Saved {filename}.")
            if col_b.button("↩️ Reload from disk", key=f"reload_{name}"):
                path = os.path.join(DATASET_DIR, filename)
                if os.path.exists(path):
                    st.session_state.datasets[name] = load_csv_or_excel(path)
                    st.rerun()
                else:
                    st.info("No saved file on disk yet.")

# ==========================================================================
# PAGE: Timetable Generator
# ==========================================================================
elif page == "Timetable Generator":
    st.title("⚙️ Timetable Generator")
    st.write("Configure constraints, then generate an optimized (or random) timetable.")

    datasets = st.session_state.datasets
    required = ["subjects", "teachers", "rooms", "timeslots", "classes"]
    missing = [r for r in required if r not in datasets or datasets[r].empty]

    if missing:
        st.error(
            f"Missing or empty datasets: {', '.join(missing)}. "
            "Please add them under Data Management before generating."
        )
    else:
        with st.expander("⚙️ Constraint configuration", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                enforce_capacity = st.checkbox("Enforce room capacity", value=True)
                enforce_lab = st.checkbox("Enforce lab-room matching for lab subjects", value=True)
                enforce_availability = st.checkbox("Enforce teacher availability", value=True)
                enforce_max_day = st.checkbox("Enforce max lectures/day per teacher", value=True)
            with c2:
                w_even = st.slider("Weight: even distribution across week", 0, 10, 5)
                w_consec = st.slider("Weight: avoid consecutive same-subject lectures", 0, 10, 3)
                w_gaps = st.slider("Weight: minimize free periods", 0, 10, 4)
                w_balance = st.slider("Weight: balance teacher workload", 0, 10, 2)

            time_limit = st.slider("Solver time limit (seconds)", 5, 120, 30)
            mode = st.radio(
                "Generation mode",
                ["Optimized (AI constraint solver)", "Random (quick feasible schedule)"],
                horizontal=True,
            )

        config = ConstraintConfig(
            enforce_room_capacity=enforce_capacity,
            enforce_lab_rooms=enforce_lab,
            enforce_teacher_availability=enforce_availability,
            enforce_max_lectures_per_day=enforce_max_day,
            weight_even_distribution=w_even,
            weight_avoid_consecutive_same_subject=w_consec,
            weight_minimize_gaps=w_gaps,
            weight_balance_teacher_load=w_balance,
        )

        with st.expander("Active constraints"):
            for line in describe_constraints(config):
                st.write(f"- {line}")

        if st.button("🚀 Generate Timetable", type="primary", use_container_width=True):
            progress = st.progress(0, text="Building constraint model...")
            progress.progress(30, text="Solving with Google OR-Tools CP-SAT...")

            request = GenerationRequest(
                subjects=datasets["subjects"],
                teachers=datasets["teachers"],
                rooms=datasets["rooms"],
                timeslots=datasets["timeslots"],
                classes=datasets["classes"],
                teacher_availability=datasets.get("teacher_availability", pd.DataFrame()),
                config=config,
                time_limit_seconds=time_limit,
                random_mode=mode.startswith("Random"),
            )
            result = st.session_state.scheduler.generate(request)
            progress.progress(100, text="Done.")

            st.session_state.last_result = result

            if result.status in ("INFEASIBLE", "NO_SOLUTION"):
                st.error("No feasible timetable could be generated with the current data/constraints.")
                st.write("**Conflict / diagnostic report:**")
                for c in result.conflict_report:
                    st.write(f"- {c}")
            else:
                st.success(
                    f"Timetable generated! Status: {result.status} · "
                    f"Optimization score: {result.optimization_score}% · "
                    f"Solved in {result.solve_seconds:.1f}s"
                )
                c1, c2, c3 = st.columns(3)
                metric_card(c1, "Optimization Score", f"{result.optimization_score}%")
                metric_card(c2, "Lectures Scheduled", len(result.timetable))
                metric_card(c3, "Solve Time", f"{result.solve_seconds:.1f}s")

                if result.conflict_report:
                    st.warning("Some items could not be scheduled — see diagnostics below.")
                    with st.expander("Diagnostics"):
                        for c in result.conflict_report:
                            st.write(f"- {c}")
                else:
                    st.info("✅ No conflicts detected.")

                st.dataframe(result.timetable, use_container_width=True, height=350)

                label = st.text_input("Save this run to history as (optional name)", "")
                if st.button("💾 Save to history"):
                    path = st.session_state.scheduler.save_to_history(
                        result, label=label.strip() or None
                    )
                    st.success(f"Saved to {path}")

# ==========================================================================
# PAGE: Timetable Views
# ==========================================================================
elif page == "Timetable Views":
    st.title("📋 Timetable Views")

    result = st.session_state.last_result
    if result is None or result.timetable.empty:
        st.info("Generate a timetable first on the **Timetable Generator** page.")
    else:
        tt = result.timetable
        scheduler = st.session_state.scheduler

        view_type = st.selectbox(
            "View by",
            ["Department-wise", "Semester-wise", "Teacher-wise", "Classroom-wise", "Daily"],
        )

        search = st.text_input("🔎 Search/filter (subject, teacher, room, day...)", "")

        if view_type == "Department-wise":
            dept = st.selectbox("Department", sorted(tt["Department"].unique()))
            view_df = scheduler.department_view(tt, dept)
        elif view_type == "Semester-wise":
            dept = st.selectbox("Department", sorted(tt["Department"].unique()))
            sems = sorted(tt[tt["Department"] == dept]["Semester"].unique())
            sem = st.selectbox("Semester", sems)
            view_df = scheduler.semester_view(tt, dept, sem)
        elif view_type == "Teacher-wise":
            teacher = st.selectbox("Teacher", sorted(tt["TeacherName"].unique()))
            view_df = scheduler.teacher_view(tt, teacher)
        elif view_type == "Classroom-wise":
            room = st.selectbox("Room", sorted(tt["RoomNumber"].unique()))
            view_df = scheduler.room_view(tt, room)
        else:  # Daily
            day = st.selectbox("Day", [d for d in DAY_ORDER if d in tt["Day"].unique()])
            view_df = scheduler.daily_view(tt, day)

        if search:
            mask = view_df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
            )
            view_df = view_df[mask]

        st.write(f"**{len(view_df)} lecture(s)** matching this view")

        grid = scheduler.as_grid(view_df) if view_type != "Daily" else None
        if grid is not None and not grid.empty:
            st.write("**Weekly grid**")
            st.dataframe(grid.fillna(""), use_container_width=True)

        st.write("**Detailed list**")
        st.dataframe(view_df, use_container_width=True, height=350)

# ==========================================================================
# PAGE: Analytics
# ==========================================================================
elif page == "Analytics":
    st.title("📊 Analytics")

    result = st.session_state.last_result
    if result is None or result.timetable.empty:
        st.info("Generate a timetable first on the **Timetable Generator** page.")
    else:
        tt = result.timetable
        scheduler = st.session_state.scheduler
        rooms_df = st.session_state.datasets.get("rooms", pd.DataFrame())
        timeslots_df = st.session_state.datasets.get("timeslots", pd.DataFrame())
        total_lecture_slots = (
            len(timeslots_df[timeslots_df["SlotType"] == "Lecture"]) if not timeslots_df.empty else 1
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Teacher Workload")
            workload = scheduler.teacher_workload(tt)
            fig = px.bar(workload, x="TeacherName", y="LecturesPerWeek", color="LecturesPerWeek",
                         color_continuous_scale="Blues")
            fig.update_layout(xaxis_title="", yaxis_title="Lectures / week", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Room Utilization")
            util = scheduler.room_utilization(tt, total_lecture_slots)
            fig = px.bar(util, x="RoomNumber", y="UtilizationPercent", color="UtilizationPercent",
                         color_continuous_scale="Purples")
            fig.update_layout(xaxis_title="", yaxis_title="Utilization %", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Subject Distribution")
            dist = scheduler.subject_distribution(tt)
            fig = px.pie(dist, names="SubjectName", values="LecturesPerWeek", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.subheader("Weekly Schedule Summary")
            weekly = scheduler.weekly_summary(tt)
            fig = px.line(weekly, x="Day", y="LecturesScheduled", markers=True)
            fig.update_layout(xaxis_title="", yaxis_title="Lectures scheduled")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Conflict Report")
        if result.conflict_report:
            for c in result.conflict_report:
                st.write(f"- {c}")
        else:
            st.success("No conflicts detected in the generated timetable.")

# ==========================================================================
# PAGE: Export
# ==========================================================================
elif page == "Export":
    st.title("⬇️ Export Timetable")

    result = st.session_state.last_result
    if result is None or result.timetable.empty:
        st.info("Generate a timetable first on the **Timetable Generator** page.")
    else:
        tt = result.timetable
        scheduler = st.session_state.scheduler
        st.write(f"Exporting **{len(tt)}** scheduled lectures (status: {result.status}).")

        scope = st.selectbox(
            "What would you like to export?",
            ["Full timetable"] + [f"Department: {d}" for d in sorted(tt["Department"].unique())],
        )
        export_df = tt if scope == "Full timetable" else scheduler.department_view(
            tt, scope.replace("Department: ", "")
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📄 Download CSV",
                data=to_csv_bytes(export_df),
                file_name="timetable.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            sheets = {
                "Timetable": export_df,
                "Teacher Workload": scheduler.teacher_workload(tt),
                "Subject Distribution": scheduler.subject_distribution(tt),
            }
            st.download_button(
                "📊 Download Excel",
                data=to_excel_bytes(sheets),
                file_name="timetable.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col3:
            pdf_bytes = to_pdf_bytes(
                title="Timetable",
                df=export_df,
                subtitle=f"Generated {datetime.now().strftime('%d %b %Y %H:%M')} · "
                f"Optimization score: {result.optimization_score}%",
            )
            st.download_button(
                "🖨️ Download Printable PDF",
                data=pdf_bytes,
                file_name="timetable.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Preview")
        st.dataframe(export_df, use_container_width=True, height=350)

# ==========================================================================
# PAGE: History
# ==========================================================================
elif page == "History":
    st.title("🕘 Timetable History")
    scheduler = st.session_state.scheduler
    entries = scheduler.list_history()

    if not entries:
        st.info("No saved timetables yet. Generate one and click **Save to history**.")
    else:
        for entry in entries:
            with st.expander(
                f"{entry['name']} — {entry['num_lectures']} lectures · "
                f"score {entry['optimization_score']}% · {entry['status']}"
            ):
                st.write(f"Generated at: {entry['generated_at']}")
                st.write(f"Solve time: {entry['solve_seconds']:.1f}s")
                if os.path.exists(entry["csv_path"]):
                    df = scheduler.load_history_entry(entry["csv_path"])
                    st.dataframe(df, use_container_width=True, height=250)
                    st.download_button(
                        "Download this run (CSV)",
                        data=to_csv_bytes(df),
                        file_name=f"{entry['name']}.csv",
                        mime="text/csv",
                        key=f"dl_{entry['name']}",
                    )
                    if st.button("Load into current session", key=f"load_{entry['name']}"):
                        from optimizer import OptimizationResult

                        st.session_state.last_result = OptimizationResult(
                            status=entry["status"],
                            timetable=df,
                            conflict_report=[],
                            optimization_score=entry["optimization_score"],
                            solve_seconds=entry["solve_seconds"],
                        )
                        st.success("Loaded into current session. Go to Timetable Views/Analytics to inspect it.")






