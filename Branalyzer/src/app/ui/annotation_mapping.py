import streamlit as st
from data_ingest import build_maps

def _sync_class_a_display():
    checked_value = st.session_state["anno_class_a"]
    st.session_state["class_a_display_name"] = checked_value if checked_value != "—" else "—"

def _sync_class_b_display():
    checked_value = st.session_state["anno_class_b"]
    st.session_state["class_b_display_name"] = checked_value if checked_value != "—" else "—"

@st.fragment
def render_annotation_mapping():
    annotations = st.session_state.get("user_annotations", None)

    # File information summary
    file_info = st.session_state.get("loaded_file_info", {})
    if file_info:
        st.subheader("Loaded File Information")
        info_columns = st.columns(3)
        with info_columns[0]:
            st.metric("Channels", file_info.get("Channels", "—"))
        with info_columns[1]:
            st.metric("Sampling Frequency", file_info.get("Sampling Frequency", "—"))
        with info_columns[2]:
            st.metric("Duration", file_info.get("Duration", "—"))

    if not annotations:
        st.warning(
            "WARNING: No annotations found."
            "Classification requires at least two event types.")
        return

    st.subheader("Map Annotations to Classes")
    st.caption(
        "Assign a class to each comparison target from the discovered annotations."
        " 'Class A' and 'Class B' are required. 'Rest' is optional and will be treated as the baseline class."
    )

    annotation_list = list(annotations.keys())

    # # Gets classes with their respective counts — NOT CURRENTLY USED
    # def _label_count(desc):
    #     return f"{desc} ({annotations[desc]})"
    # options_with_count = [_label_count(desc) for desc in annotation_list]

    none_option = ["—"]
    options_with_none = none_option + annotation_list

    # Class A Selection
    col_a_select, col_a_name = st.columns(2)
    with col_a_select:
        class_a_pick = st.selectbox(
            "Class A",
            options=options_with_none,
            index=0,
            key="anno_class_a",
            on_change=_sync_class_a_display,
        )

    # if st.session_state.get("_prev_class_a") != class_a_pick:
    #     st.session_state["class_a_display_name"] = class_a_pick
    #     st.session_state["_prev_class_a"] = class_a_pick
    with col_a_name:
        class_a_display = st.text_input(
            "Display name",
            key="class_a_display_name",
        )

    # Class B Selection
    col_b_select, col_b_name = st.columns(2)
    with col_b_select:
        class_b_pick = st.selectbox(
            "Class B",
            options=options_with_none,
            index=0,
            key="anno_class_b",
            on_change=_sync_class_b_display,
        )

    # if st.session_state.get("_prev_class_b") != class_b_pick:
    #     st.session_state["class_b_display_name"] = class_b_pick
    #     st.session_state["_prev_class_b"] = class_b_pick
    with col_b_name:
        class_b_display = st.text_input(
            "Display name",
            key="class_b_display_name",
        )

    # Rest (optional)
    col_r_pick, col_filler = st.columns(2)
    with col_r_pick:
        rest_pick = st.selectbox(
            "Rest (optional)",
            options=options_with_none,
            index=0,
            key="anno_rest",
        )
    with col_filler:
        if rest_pick != "—":
            st.info(f'"{rest_pick}" is marked as Rest and will be treated as the baseline class.')
        else:
            st.empty()

    # Validation
    if class_a_pick == class_b_pick and (class_a_pick != "—" and class_b_pick != "—"):
        st.error("Class A and Class B must be different annotations.")
    elif rest_pick != "—" and rest_pick in (class_a_pick, class_b_pick):
        st.error("Rest must be a different annotation from Class A and Class B.")
    elif class_a_display.strip() == "" or class_b_display.strip() == "":
        st.error("Display names cannot be empty or whitespace.")
    elif class_a_display.strip() == class_b_display.strip() and (class_a_display.strip() != "—" and class_b_display.strip() != "—"):
        st.error("Provide different display names for Class A and Class B before confirming.")
    elif class_a_pick != "—" and class_b_pick != "—":

        if st.button("Confirm Mapping"):
                
                if class_a_pick == "—" and class_b_pick == "—":
                    st.error("Please assign both Class A and Class B to annotations before confirming.")
                elif class_a_pick == "—":
                    st.error("Class A must be assigned to an annotation.")
                elif class_b_pick == "—":
                    st.error("Class B must be assigned to an annotation.")
                # Build the assignments dict the rest of the pipeline expects
                else:
                    assignments = {}
                    for desc in annotation_list:
                        if desc == class_a_pick:
                            assignments[desc] = "Class A"
                        elif desc == class_b_pick:
                            assignments[desc] = "Class B"
                        elif desc == rest_pick:
                            assignments[desc] = "Rest"
                        else:
                            assignments[desc] = "Ignore"

                    event_map, code_map = build_maps(assignments)
                    st.session_state["user_assignments"] = assignments
                    st.session_state["user_event_map"] = event_map
                    st.session_state["user_code_map"] = code_map
                    st.session_state["user_display_names"] = {
                        "class_a": class_a_display or class_a_pick,
                        "class_b": class_b_display or class_b_pick,
                    }
                    st.session_state["annotation_mapping_complete"] = True
                    st.rerun()
    else:
        st.info("Please assign both Class A and Class B to annotations.")