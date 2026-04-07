from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class WaveformPlotConfig:
    """Caller-facing options for one waveform plot window."""
    picks: list[str]
    t_start: float
    duration: float
    ds_factor: int
    title: str
    scroll_step: float = 0.5
    scroll_seconds: float = 30.0
    frame_duration_ms: int = 200


@dataclass(frozen=True)
class PreparedWaveformData:
    """Normalized waveform data and derived layout values used for plotting."""
    channel_names: list[str]
    channel_indices: list[int]
    data: np.ndarray
    times: np.ndarray
    sfreq: float
    total_time: float
    offsets: np.ndarray
    y_min: float
    y_max: float


@dataclass(frozen=True)
class WindowSlice:
    """A single time window sliced from the prepared waveform data."""
    start_time: float
    end_time: float
    times: np.ndarray
    data: np.ndarray


def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Return every nth sample for each channel when downsampling is requested."""
    if factor <= 1:
        return data
    return data[:, ::factor]


def normalize_ds_factor(ds_factor: int) -> int:
    """Clamp the downsample factor to a valid positive integer."""
    return max(1, int(ds_factor))


def compute_channel_offsets(data: np.ndarray) -> np.ndarray:
    """Compute vertical offsets so stacked channels do not overlap visually."""
    if data.size == 0:
        return np.array([], dtype=float)

    global_scale = float(np.nanstd(data))
    if global_scale <= 0 or np.isnan(global_scale):
        global_scale = 1.0

    return np.arange(data.shape[0])[::-1] * global_scale * 4.0


def compute_y_range(data: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    """Compute a fixed y-axis range for the full prepared waveform view."""
    if data.size == 0 or len(offsets) == 0:
        return 0.0, 1.0

    y_min = float(np.min(data) + offsets[-1])
    y_max = float(np.max(data) + offsets[0])
    if y_max <= y_min:
        y_max = y_min + 1.0

    return y_min, y_max


def prepare_waveform_data(raw, config: WaveformPlotConfig) -> PreparedWaveformData | None:
    """Select channels, downsample the signal, and derive plot-ready waveform state."""
    if raw is None:
        return None

    channel_names = list(raw.ch_names)
    channel_indices = [channel_names.index(channel) for channel in config.picks if channel in channel_names]
    if not channel_indices:
        return None

    ds_factor = normalize_ds_factor(config.ds_factor)
    sfreq = float(raw.info["sfreq"])
    total_time = raw.n_times / sfreq

    all_data, all_times = raw[channel_indices, :]
    all_data = _downsample(all_data, ds_factor)
    all_times = all_times[::ds_factor]

    selected_channel_names = [channel_names[index] for index in channel_indices]
    offsets = compute_channel_offsets(all_data)
    y_min, y_max = compute_y_range(all_data, offsets)

    return PreparedWaveformData(
        channel_names=selected_channel_names,
        channel_indices=channel_indices,
        data=all_data,
        times=all_times,
        sfreq=sfreq / ds_factor,
        total_time=total_time,
        offsets=offsets,
        y_min=y_min,
        y_max=y_max,
    )


def slice_window(
    prepared: PreparedWaveformData,
    start_time: float,
    duration: float,
) -> WindowSlice:
    """Slice one visible window from the prepared waveform state."""
    clamped_start = max(0.0, float(start_time))
    clamped_duration = max(0.0, float(duration))

    start_index = max(0, int(clamped_start * prepared.sfreq))
    end_index = min(len(prepared.times), int((clamped_start + clamped_duration) * prepared.sfreq))
    if end_index <= start_index:
        end_index = min(len(prepared.times), start_index + 1)

    window_times = prepared.times[start_index:end_index]
    window_data = prepared.data[:, start_index:end_index]

    if window_times.size == 0 and prepared.times.size > 0:
        last_index = min(len(prepared.times) - 1, start_index)
        window_times = prepared.times[last_index:last_index + 1]
        window_data = prepared.data[:, last_index:last_index + 1]

    display_end = clamped_start + clamped_duration
    if window_times.size > 0:
        display_end = max(display_end, float(window_times[-1]))

    return WindowSlice(
        start_time=clamped_start,
        end_time=display_end,
        times=window_times,
        data=window_data,
    )


class WaveformOverlay(Protocol):
    """Protocol for pluggable trace builders that share the waveform data pipeline."""

    def build_initial_traces(
        self,
        window_slice: WindowSlice,
        prepared: PreparedWaveformData,
    ) -> list[go.Scatter]:
        """Build the traces shown when the figure is first rendered."""
        ...

    def build_frame_traces(
        self,
        window_slice: WindowSlice,
        prepared: PreparedWaveformData,
    ) -> list[go.Scatter]:
        """Build traces for later windows using the same prepared waveform state."""
        ...


class WaveformLineOverlay:
    """Render the stacked channel lines for the base waveform view."""

    def build_initial_traces(
        self,
        window_slice: WindowSlice,
        prepared: PreparedWaveformData,
    ) -> list[go.Scatter]:
        """Build the initial line traces with legend entries enabled."""
        return self._build_traces(window_slice, prepared, include_legend=True)

    def build_frame_traces(
        self,
        window_slice: WindowSlice,
        prepared: PreparedWaveformData,
    ) -> list[go.Scatter]:
        """Build line traces for subsequent windows without repeating legend entries."""
        return self._build_traces(window_slice, prepared, include_legend=False)

    def _build_traces(
        self,
        window_slice: WindowSlice,
        prepared: PreparedWaveformData,
        include_legend: bool,
    ) -> list[go.Scatter]:
        """Build one Plotly trace per selected channel for the supplied window."""
        traces: list[go.Scatter] = []

        for channel_index, channel_name in enumerate(prepared.channel_names):
            raw_amplitude = window_slice.data[channel_index]
            display_amplitude = raw_amplitude + prepared.offsets[channel_index]
            customdata = np.column_stack((raw_amplitude, np.full(raw_amplitude.shape, prepared.offsets[channel_index])))

            traces.append(
                go.Scatter(
                    x=window_slice.times,
                    y=display_amplitude,
                    mode="lines",
                    name=channel_name,
                    showlegend=include_legend,
                    customdata=customdata,
                    hovertemplate=(
                        "t=%{x:.3f}s"
                        "<br>amp=%{customdata[0]:.3f}"
                        "<br>offset=%{customdata[1]:.3f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        return traces


class WaveformFigureBuilder:
    """Assemble a waveform figure from prepared data and a list of overlays."""

    def __init__(
        self,
        prepared: PreparedWaveformData,
        config: WaveformPlotConfig,
        overlays: list[WaveformOverlay],
    ) -> None:
        """Store the prepared waveform state and overlay set for figure creation."""
        self.prepared = prepared
        self.config = config
        self.overlays = overlays

    def build_figure(self) -> go.Figure:
        """Build the final Plotly figure for the current waveform window."""
        figure = go.Figure()
        initial_window = self.build_initial_window()

        for trace in self.build_initial_traces(initial_window):
            figure.add_trace(trace)

        figure.update_layout(self.build_layout(initial_window))

        return figure

    def build_initial_window(self) -> WindowSlice:
        """Compute the initial visible window after clamping the requested start time."""
        max_window_start = max(0.0, self.prepared.total_time - max(float(self.config.duration), 0.0))
        initial_start = min(max(0.0, float(self.config.t_start)), max_window_start)
        return slice_window(self.prepared, initial_start, self.config.duration)

    def build_initial_traces(self, window_slice: WindowSlice) -> list[go.Scatter]:
        """Collect initial traces from all registered overlays in stable order."""
        traces: list[go.Scatter] = []
        for overlay in self.overlays:
            traces.extend(overlay.build_initial_traces(window_slice, self.prepared))
        return traces

    def build_layout(self, initial_window: WindowSlice) -> dict:
        """Build the shared Plotly layout for the waveform figure."""
        y_padding = (self.prepared.y_max - self.prepared.y_min) * 0.05

        return {
            "height": 500,
            "margin": {"l": 20, "r": 20, "t": 24, "b": 40},
            "legend": {"orientation": "h"},
            "xaxis": {
                "range": [float(initial_window.start_time), float(initial_window.end_time)],
            },
            "yaxis": {
                "range": [self.prepared.y_min - y_padding, self.prepared.y_max + y_padding],
            },
            "annotations": [
                {
                    "text": "Time (s)",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.0,
                    "y": -0.12,
                    "showarrow": False,
                    "xanchor": "left",
                    "yanchor": "top",
                }
            ],
        }


def plot_waveforms_plotly(
    raw,
    picks: list[str],
    t_start: float,
    duration: float,
    ds_factor: int,
    title: str,
    scroll_step: float = 0.5,
    scroll_seconds: float = 30.0,
    frame_duration_ms: int = 200,
):
    """Compatibility wrapper that prepares data and returns a Plotly waveform figure."""
    config = WaveformPlotConfig(
        picks=picks,
        t_start=t_start,
        duration=duration,
        ds_factor=ds_factor,
        title=title,
        scroll_step=scroll_step,
        scroll_seconds=scroll_seconds,
        frame_duration_ms=frame_duration_ms,
    )

    prepared = prepare_waveform_data(raw, config)
    if prepared is None:
        return None

    builder = WaveformFigureBuilder(
        prepared=prepared,
        config=config,
        overlays=[WaveformLineOverlay()],
    )
    return builder.build_figure()
