# Waveform Plotting Pipeline

This page documents the waveform plotting pipeline used by the signal inspection UI in `Branalyzer/src/app/ui/waveform.py`.

## Core Data Structures

`WaveformPlotConfig`
Stores plot options such as selected channels, window start, window duration, and downsample factor.

`PreparedWaveformData`
Stores downsampled waveform data and derived values needed for plotting, including offsets and axis bounds.

Key fields:
- `data`: `(n_channels, n_times)` float array
- `times`: `(n_times,)` float array
- `offsets`: per-channel vertical offsets for stacking
- `y_min`, `y_max`: fixed bounds to stabilize the axis

`WindowSlice`
Represents a single time window extracted from prepared waveform data.

## Overlays

`WaveformOverlay`
Protocol for pluggable trace builders that share the waveform data pipeline.

`WaveformLineOverlay`
Builds the stacked waveform line traces used in the current UI.

## Builder

`WaveformFigureBuilder`
Assembles the Plotly figure from prepared data and registered overlays.

## Helper Functions

`prepare_waveform_data(raw, config)`
Selects channels, downsamples signals, and prepares waveform data for plotting.

`slice_window(prepared, start_time, duration)`
Extracts the visible time window for the current plot.

`plot_waveforms_plotly(...)`
Public wrapper used by the UI layer to build a waveform figure.

## Technical Notes

- Downsampling keeps every nth sample (`ds_factor`) to speed up plotting.
- Vertical offsets are scaled by global standard deviation to avoid overlap.
- Hover data includes raw amplitude and applied offset for each channel.
