from __future__ import annotations

import time

import streamlit as st

from ui.waveform import plot_waveforms_plotly



@st.fragment
def render_signal_inspection(picks: list[str]) -> None:
    """Render the signal inspection area and keep waveform interactions fragment-scoped.

    The fragment reads loaded and preprocessed signal views from session state so
    waveform navigation does not require a full-page rerun.
    """
    st.divider()
    st.header("Signal Inspection")

    if not st.session_state["data_loaded"] or st.session_state["raw_view"] is None:
        st.info("Click Load Data to inspect the raw EEG signal.")
        return

    st.write("EEG channel viewer. Raw is available after loading; filtered and ICA-cleaned views appear after preprocessing.")
    slider_col_ds, slider_col_frame, slider_col_window = st.columns(3)
    with slider_col_ds:
        st.session_state["waveform_ds_factor"] = int(
            st.select_slider(
                "Downsample factor",
                options=[1, 2, 3, 4, 5, 6, 8, 10, 12],
                value=int(st.session_state["waveform_ds_factor"]),
                key="waveform_ds_factor_slider",
            )
        )
    with slider_col_frame:
        st.session_state["waveform_frame_speed"] = int(
            st.select_slider(
                "Frame speed (ms)",
                options=[50, 100, 150, 200, 300, 500],
                value=int(st.session_state["waveform_frame_speed"]),
                key="waveform_frame_speed_slider",
            )
        )
    with slider_col_window:
        st.session_state["waveform_window_size"] = int(
            st.slider(
                "Window size (s)",
                min_value=10,
                max_value=30,
                value=int(st.session_state["waveform_window_size"]),
                step=1,
                key="waveform_window_size_slider",
            )
        )

    raw_view = st.session_state["raw_view"]
    filtered_view = st.session_state["filtered_view"]
    ica_view = st.session_state["ica_view"]

    if st.session_state["preprocessing_ready"] and filtered_view is not None and ica_view is not None:
        tab_raw, tab_filt, tab_ica = st.tabs(["Raw", "Filtered", "ICA-Cleaned"])

        with tab_raw:
            render_waveform_plot(
                raw_view,
                picks,
                st.session_state["waveform_window_start"],
                st.session_state["waveform_window_size"],
                st.session_state["waveform_ds_factor"],
                "Raw (avg ref + selected channels)",
                st.session_state["waveform_frame_speed"],
                plot_key="raw",
            )

        with tab_filt:
            render_waveform_plot(
                filtered_view,
                picks,
                st.session_state["waveform_window_start"],
                st.session_state["waveform_window_size"],
                st.session_state["waveform_ds_factor"],
                "Filtered (1 Hz highpass + 8-30 Hz bandpass)",
                st.session_state["waveform_frame_speed"],
                plot_key="filtered",
            )

        with tab_ica:
            render_waveform_plot(
                ica_view,
                picks,
                st.session_state["waveform_window_start"],
                st.session_state["waveform_window_size"],
                st.session_state["waveform_ds_factor"],
                "ICA-Cleaned",
                st.session_state["waveform_frame_speed"],
                plot_key="ica",
            )
    else:
        render_waveform_plot(
            raw_view,
            picks,
            st.session_state["waveform_window_start"],
            st.session_state["waveform_window_size"],
            st.session_state["waveform_ds_factor"],
            "Raw (avg ref + selected channels)",
            st.session_state["waveform_frame_speed"],
            plot_key="raw",
            subtitle="Run preprocessing to unlock filtered and ICA-cleaned views.",
        )

    render_waveform_navigation(st.session_state["waveform_window_size"])
    maybe_autoplay_waveform(
        st.session_state["waveform_window_size"],
        st.session_state["waveform_frame_speed"],
    )


def render_waveform_navigation(duration: int) -> None:
    """Render waveform navigation controls and sync them with session state.

    This keeps the current window start, step size, and jump input aligned across
    button clicks, manual edits, and fragment reruns.
    """
    raw_view = st.session_state["raw_view"]
    if raw_view is None:
        return

    total_time = raw_view.n_times / float(raw_view.info["sfreq"])
    max_window_start = max(0.0, total_time - float(duration))
    current_start = min(max(0.0, float(st.session_state["waveform_window_start"])), max_window_start)
    st.session_state["waveform_window_start"] = current_start

    st.session_state["waveform_jump_to"] = min(
        max(0, int(st.session_state["waveform_jump_to"])),
        int(max_window_start),
    )
    if st.session_state["waveform_sync_jump_input"]:
        st.session_state["waveform_jump_to_input"] = int(st.session_state["waveform_jump_to"])
        st.session_state["waveform_sync_jump_input"] = False

    label_left, label_step, label_jump, label_right = st.columns([1, 1, 1, 1])
    with label_step:
        st.caption("Step (s)")
    with label_jump:
        st.caption("Jump to (s)")

    col_left, col_step, col_jump, col_right = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
    with col_left:
        if st.button("<", key="waveform_step_left", use_container_width=True):
            st.session_state["waveform_window_start"] = max(
                0.0,
                current_start - float(st.session_state["waveform_step_size"]),
            )
            st.session_state["waveform_jump_to"] = int(st.session_state["waveform_window_start"])
            st.session_state["waveform_sync_jump_input"] = True
            try:
                st.rerun(scope="fragment")
            except st.errors.StreamlitAPIException:
                pass
    with col_step:
        step_value = int(
            st.number_input(
                "Step",
                min_value=1,
                max_value=max(1, int(max_window_start) if max_window_start >= 1 else 1),
                value=int(st.session_state["waveform_step_size"]),
                step=1,
                key="waveform_step_size_input",
                label_visibility="collapsed",
            )
        )
        if step_value != int(st.session_state["waveform_step_size"]):
            st.session_state["waveform_step_size"] = step_value
    with col_jump:
        jump_to = int(
            st.number_input(
                "Jump to",
                min_value=0,
                max_value=max(0, int(max_window_start)),
                step=1,
                key="waveform_jump_to_input",
                label_visibility="collapsed",
            )
        )
        if jump_to != int(current_start):
            st.session_state["waveform_jump_to"] = jump_to
            st.session_state["waveform_window_start"] = float(jump_to)
            st.session_state["waveform_sync_jump_input"] = False
            try:
                st.rerun(scope="fragment")
            except st.errors.StreamlitAPIException:
                pass
    with col_right:
        if st.button(">", key="waveform_step_right", use_container_width=True):
            st.session_state["waveform_window_start"] = min(
                max_window_start,
                current_start + float(st.session_state["waveform_step_size"]),
            )
            st.session_state["waveform_jump_to"] = int(st.session_state["waveform_window_start"])
            st.session_state["waveform_sync_jump_input"] = True
            try:
                st.rerun(scope="fragment")
            except st.errors.StreamlitAPIException:
                pass


def maybe_autoplay_waveform(duration: int, frame_speed: int) -> None:
    """Advance the waveform window while autoplay is enabled.

    The function sleeps for the requested frame duration, updates navigation state,
    and reruns only the signal inspection fragment.
    """
    if not st.session_state["waveform_autoplay"]:
        return

    raw_view = st.session_state["raw_view"]
    if raw_view is None:
        st.session_state["waveform_autoplay"] = False
        return

    total_time = raw_view.n_times / float(raw_view.info["sfreq"])
    max_window_start = max(0.0, total_time - float(duration))
    current_start = float(st.session_state["waveform_window_start"])
    step_size = float(st.session_state["waveform_step_size"])
    next_start = min(max_window_start, current_start + step_size)

    if next_start <= current_start:
        st.session_state["waveform_autoplay"] = False
        return

    time.sleep(max(0.05, float(frame_speed) / 1000.0))
    st.session_state["waveform_window_start"] = next_start
    st.session_state["waveform_jump_to"] = int(next_start)
    st.session_state["waveform_sync_jump_input"] = True
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        pass


def render_waveform_plot(
    raw,
    picks: list[str],
    t_start: float,
    duration: int,
    ds_factor: int,
    title: str,
    frame_speed: int,
    plot_key: str,
    subtitle: str | None = None,
) -> None:
    """Render one waveform chart block for the supplied signal view.

    This draws the title row, builds the Plotly figure, and shows a warning if the
    current channel selection cannot produce a plot.
    """
    render_waveform_title_row(title, plot_key, subtitle)

    figure = plot_waveforms_plotly(
        raw,
        picks,
        t_start,
        float(duration),
        int(ds_factor),
        title,
        frame_duration_ms=int(frame_speed),
    )
    if figure is None:
        st.warning("Nothing to plot (check channel selection).")
    else:
        st.plotly_chart(figure, use_container_width=True)


def render_waveform_title_row(title: str, plot_key: str, subtitle: str | None = None) -> None:
    """Render the waveform title, optional subtitle, and play or pause control.

    The autoplay toggle is stored in session state and triggers a fragment rerun so
    playback updates stay localized to signal inspection.
    """
    title_col, button_col = st.columns([5, 1])
    with title_col:
        st.markdown(f"#### {title}")
        if subtitle:
            st.caption(subtitle)
    with button_col:
        button_label = "Pause" if st.session_state["waveform_autoplay"] else "Play"
        if st.button(button_label, key=f"waveform_autoplay_{plot_key}", use_container_width=True):
            st.session_state["waveform_autoplay"] = not st.session_state["waveform_autoplay"]
            st.rerun(scope="fragment")
