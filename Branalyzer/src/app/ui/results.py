from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


#_LABEL_NAMES_RAW = {1: "Rest", 2: "Left Fist", 3: "Right Fist"} OLD HARDCODED LABEL MAP
#_LABEL_NAMES_BINARY = {0: "Left Fist", 1: "Right Fist"} OLD HARDCODED LABEL MAP

def _get_class_labels() -> tuple[dict[int, str], dict[int, str]]:
    """Return label maps for raw event codes and binary-mapped classes based on session state."""
    assignments = st.session_state.get("user_assignments", {}) # If upload path exists, i.e., if user uploaded file

    display_names = st.session_state.get("user_display_names", {})

    if display_names:
        class_a_desc = display_names.get("class_a", "Class A")
        class_b_desc = display_names.get("class_b", "Class B")
        rest_desc = display_names.get("rest", "Rest")
    elif st.session_state.get("user_assignments"):
        # Fallback to raw annotation descriptions if no display names set
        assignments = st.session_state["user_assignments"]
        class_a_desc = next((d for d, r in assignments.items() if r == "Class A"), "Class A")
        class_b_desc = next((d for d, r in assignments.items() if r == "Class B"), "Class B")
        rest_desc = next((d for d, r in assignments.items() if r == "Rest"), "Rest")
    else:
        # Fallback to PhysioNet EEGMMIDB default descriptions if no user assignments (i.e., for demo subjects)
        class_a_desc = "Left Fist"
        class_b_desc = "Right Fist"
        rest_desc = "Rest"

    raw_labels = {1: rest_desc, 2: class_a_desc, 3: class_b_desc}
    binary_labels = {0: class_a_desc, 1: class_b_desc}
    return raw_labels, binary_labels 



def render_model_benchmarking() -> None:
    """Render summary metrics, the model comparison table, and confusion matrices.

    This section expects finished model results in session state and stays hidden
    until the model stage has completed.
    """
    st.divider()
    st.header("Model Benchmarking")

    if not st.session_state["models_ready"] or not st.session_state["results"]:
        st.info("Run models after preprocessing to view benchmarking results.")
        return

    results = st.session_state["results"]
    best_performer = max(results, key=lambda result: result.accuracy)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Best Model", best_performer.name)
    with cols[1]:
        st.metric("Accuracy", f"{best_performer.accuracy:.4f}")
    with cols[2]:
        st.metric("F1 Score", f"{best_performer.f1_score:.4f}")
    with cols[3]:
        st.metric("Std Dev", f"{best_performer.std_dev:.4f}")

    rows = []
    for result in results:
        rows.append(
            {
                "Model": result.name + " + CSP",
                "Accuracy": result.accuracy,
                "F1 Score": result.f1_score,
                "Std Dev": result.std_dev,
                "Inference Time (s)": result.inference_time_s,
            }
        )

    dataframe = pd.DataFrame(rows).round(
        {"Accuracy": 4, "F1 Score": 4, "Std Dev": 4, "Inference Time (s)": 8}
    )
    st.dataframe(dataframe, use_container_width=True, hide_index=True)

    max_cm_value = max(
        (result.confusion_matrix.max() if result.confusion_matrix is not None else 0 for result in results),
        default=0,
    )

    st.subheader("Confusion Matrix")
    confusion_columns = st.columns(len(results))
    for column, result in zip(confusion_columns, results):
        with column:
            render_confusion_matrix(result, max_cm_value)


def render_confusion_matrix(result, max_cm_value: int) -> None:
    """Render one confusion matrix heatmap for a model result."""
    confusion_matrix = result.confusion_matrix
    if confusion_matrix is None:
        st.info(f"{result.name}: No confusion matrix available.")
        return
    
    raw_labels, binary_labels = _get_class_labels()
    n_classes = confusion_matrix.shape[0]
    if n_classes <= 2:
        class_labels = [binary_labels[0], binary_labels[1]][:n_classes]
    else:
        class_labels = [raw_labels[1], raw_labels[2], raw_labels[3]][:n_classes]

    figure = go.Figure(
        go.Heatmap(
            z=confusion_matrix,
            x=class_labels,
            y=class_labels,
            colorscale="teal",
            text=confusion_matrix,
            texttemplate="%{text}",
            showscale=False,
            zmin=0,
            zmax=max_cm_value,
        )
    )
    figure.update_layout(
        title=f"{result.name} - Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=280,
        width=280,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(figure, use_container_width=True)


def render_event_log() -> None:
    """Render the combined epoch-by-epoch prediction log for all selected models.

    The table uses cell color to show whether each model prediction matches the
    ground truth for that epoch.
    """
    st.divider()
    st.header("Event Log")

    if not st.session_state["models_ready"] or not st.session_state["results"]:
        st.info("Run models to inspect ground truth vs. prediction timelines.")
        return

    st.write(
        "Ground truth vs. prediction timeline. Each model column is colored per epoch: "
        "green for a correct prediction and red for a mismatch."
    )

    results_with_predictions = [
        result for result in st.session_state["results"] if result.predictions is not None
    ]
    if not results_with_predictions:
        st.info("No predictions available yet.")
        return

    comparison_table, match_lookup = build_event_log_comparison_table(results_with_predictions)
    styled_table = comparison_table.style.apply(
        lambda row: highlight_model_prediction_cells(row, match_lookup),
        axis=1,
    )
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    pipeline_out = st.session_state["pipeline_out"]
    if pipeline_out is not None:
        with st.expander("Pipeline Notes"):
            st.write(pipeline_out.notes)
            st.write({"Subjects": pipeline_out.n_subjects, "Epochs": pipeline_out.n_epochs})


def build_event_log_comparison_table(results) -> tuple[pd.DataFrame, dict[tuple[int, str], bool | None]]:
    """Build the shared event log table and per-cell match lookup.

    The returned lookup is used later for styling model prediction cells without
    altering the display values in the dataframe itself.
    """
    first_result = results[0]
    ground_truth = first_result.ground_truth
    epoch_count = len(first_result.predictions)
    raw_labels, binary_labels = _get_class_labels()

    truth_label_map = raw_labels
    if ground_truth is not None and set(np.unique(ground_truth).tolist()) <= {0, 1}:
        truth_label_map = binary_labels

    rows: list[dict[str, str]] = []
    match_lookup: dict[tuple[int, str], bool | None] = {}

    for epoch_index in range(epoch_count):
        truth_value = ground_truth[epoch_index] if ground_truth is not None else None
        row = {
            "Epoch #": epoch_index + 1,
        }

        for result in results:
            prediction_value = result.predictions[epoch_index]
            prediction_label_map = (
                binary_labels
                if set(np.unique(result.predictions).tolist()) <= {0, 1}
                else raw_labels
            )
            column_name = result.name
            row[column_name] = prediction_label_map.get(int(prediction_value), str(prediction_value))
            match_lookup[(epoch_index, column_name)] = (
                None if truth_value is None else int(prediction_value) == int(truth_value)
            )

        row["Ground Truth"] = truth_label_map.get(int(truth_value), str(truth_value)) if truth_value is not None else "-"
        rows.append(row)

    return pd.DataFrame(rows), match_lookup


def highlight_model_prediction_cells(row, match_lookup: dict[tuple[int, str], bool | None]):
    """Return per-cell styles for one event log row based on match status."""
    epoch_index = int(row["Epoch #"]) - 1
    styles: list[str] = []

    for column_name in row.index:
        if column_name in {"Epoch #", "Ground Truth"}:
            styles.append("")
            continue

        match = match_lookup.get((epoch_index, column_name))
        if match is True:
            styles.append("background-color: #153b22")
        elif match is False:
            styles.append("background-color: #4c1616")
        else:
            styles.append("")

    return styles
