# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import DrawingArea, AnnotationBbox

from lazyfmri import plotting

import calinet.core.io as cio
from calinet.config import stim_colors
from calinet.logger import current_subject
from calinet.utils import find_available_modalities, update_kwargs

from typing import Union, Optional, Sequence, Any, Tuple, List, Dict


import logging
logger = logging.getLogger(__name__)


def add_subject_legend(
        fig,
        axes,
        ref_ax: Optional[Any]=None,
        order: Optional[Sequence[str]]=None,
        pad_x: float=0.0,
        pad_y: float=0.0,
        **legend_kwargs: Any
    ) -> Optional[Any]:
    """
    Add a deduplicated figure-level legend anchored to the top-right.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to which the legend is added.
    axes : sequence of matplotlib.axes.Axes
        Axes to scan for labeled artists.
    ref_ax : matplotlib.axes.Axes or None, optional
        Axes whose top-right corner is used as anchor. If ``None``, the
        union of all supplied axes is used.
    order : sequence of str or None, optional
        Optional label order (e.g., ``("CSpr", "CSpu")``). Only labels present
        in the plot are included.
    pad_x : float, default=0.0
        Horizontal offset in figure coordinates.
    pad_y : float, default=0.0
        Vertical offset in figure coordinates.
    legend_kwargs : dict of str to Any
        Additional keyword arguments passed to ``fig.legend``.

    Returns
    -------
    legend : matplotlib.legend.Legend or None
        Created legend object, or ``None`` if no labeled artists are found.

    Notes
    -----
    This function inspects axes for labeled artists, deduplicates labels, and
    adds a single figure-level legend.

    It modifies the provided ``fig`` in place and does not write files.
    """

    fig.canvas.draw()

    # collect first handle for each unique label
    by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if not l or l.startswith("_"):
                continue
            if l not in by_label:
                by_label[l] = h

    if not by_label:
        return None

    if order is None:
        labels = list(by_label.keys())
    else:
        labels = [lab for lab in order if lab in by_label]

    handles = [by_label[lab] for lab in labels]

    if ref_ax is not None:
        pos = ref_ax.get_position()
        x = pos.x1 + pad_x
        y = pos.y1 + pad_y
    else:
        x = max(ax.get_position().x1 for ax in axes) + pad_x
        y = max(ax.get_position().y1 for ax in axes) + pad_y

    return fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(x, y),
        bbox_transform=fig.transFigure,
        frameon=False,
        **legend_kwargs,
    )


def _generate_single_qa_plot(
        dataset_name: str,
        subject: str,
        qa_dir: Union[str, Path],
        **kwargs
    ) -> Tuple[str, str]:
    """
    Generate and save a single QA plot for a subject.

    Parameters
    ----------
    dataset_name : str
        Dataset or lab name used to locate subject data.
    subject : str
        Subject identifier.
    qa_dir : str or pathlib.Path
        Directory where the QA plot image is saved.

    Returns
    -------
    result : tuple of (str, str)
        Tuple containing ``subject`` and the output file path.

    Notes
    -----
    The output file is named ``"<subject>_desc-overview.png"``.

    This function creates a matplotlib figure using
    ``plot_modalities_per_subject``, writes an image file to disk, and closes
    the figure.

    This function performs file I/O and logging.
    """

    token = current_subject.set(subject)
    logger = logging.getLogger(__name__)

    f = plot_modalities_per_subject(
        dataset_name,
        subject,
        **kwargs
    )

    fname = os.path.join(qa_dir, f"{subject}_desc-overview.png")
    try:
        f.savefig(
            fname,
            bbox_inches="tight",
            dpi=300
        )
        logger.info(f"Image saved as '{fname}'")
    finally:
        plt.close(f)

    return subject, fname


def add_subject_title(
        fig,
        ref_ax,
        text: str,
        pad: float=0.02,
        **kwargs: Any
    ) -> Any:
    """
    Add a centered subject title above a reference axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to which the title is added.
    ref_ax : matplotlib.axes.Axes
        Reference axis used to determine horizontal alignment and vertical
        position.
    text : str
        Title text to display.
    pad : float, default=0.02
        Vertical offset in figure coordinates above ``ref_ax``.
    kwargs : dict of str to Any
        Additional keyword arguments passed to ``fig.text``.

    Returns
    -------
    text_artist : matplotlib.text.Text
        Created text artist.

    Notes
    -----
    This function modifies the provided ``fig`` in place and does not perform
    file I/O.
    """

    fig.canvas.draw()
    pos = ref_ax.get_position()
    xcenter = (pos.x0 + pos.x1) / 2
    y = pos.y1 + pad
    return fig.text(xcenter, y, text, ha="center", va="bottom", **kwargs)


def plot_modalities_per_subject(
        lab_name: str,
        subject: str,
        root_path: Union[Path, str]="Z:\\CALINET2\\converted",
        task_names: List[str]=["acquisition", "extinction"],
        **kwargs: Any
    ) -> matplotlib.figure.Figure:
    """
    Plot all available physiology modalities for a subject across tasks.

    Parameters
    ----------
    lab_name : str
        Dataset or lab name.
    subject : str
        Subject identifier.
    root_path : pathlib.Path or str, default="Z:\\CALINET2\\converted"
        Root directory containing subject data.
    task_names : list of str or None, optional
        Task names to include (e.g., ``["acquisition", "extinction"]``).
        If ``None``, default tasks are used.
    kwargs : dict of str to Any
        Additional plotting keyword arguments forwarded to
        ``plot_physio_with_events``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing modality plots arranged by task.

    Notes
    -----
    This function scans for available modality files, creates a multi-panel
    figure, and plots waveform data with events.

    Missing modality files are skipped with warnings.

    This function modifies matplotlib state and returns a figure but does not
    write files.
    """

    root_path = Path(root_path)
    subj_path = root_path / lab_name / subject / "physio"

    # subject-level superset, only for ordering
    available_mods = find_available_modalities(subj_path)

    # determine which modalities actually exist for each task
    task_mods = {}
    for task in task_names:
        mods_this_task = []
        for mod in available_mods:
            phys_file = subj_path / f"{subject}_task-{task}_recording-{mod}_physio.tsv.gz"
            if phys_file.exists():
                mods_this_task.append(mod)
            else:
                logger.warning(f"Missing file for task='{task}', modality='{mod}': {phys_file}. Skipping this axis.")
                
        task_mods[task] = mods_this_task

    # size based on actual number of rows
    total_rows = sum(max(1, len(task_mods[task])) for task in task_names)
    figsize = (14, total_rows * 3.54)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(nrows=len(task_names), ncols=1)

    all_plot_axes = []
    task_title_axes = []
    stim_types_last = None

    for task, sf in zip(task_names, np.atleast_1d(subfigs)):
        mods = task_mods[task]
        nmods = max(1, len(mods))  # keep at least one row so title block behaves nicely

        gs = sf.add_gridspec(
            nrows=nmods + 1,
            ncols=1,
            height_ratios=[0.18] + [1] * nmods
        )

        task_ax = sf.add_subplot(gs[0, 0])
        task_ax.axis("off")
        task_ax.text(0.5, 0.0, task, ha="center", va="bottom", fontsize=20)
        task_title_axes.append(task_ax)

        if not mods:
            empty_ax = sf.add_subplot(gs[1, 0])
            empty_ax.axis("off")
            empty_ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            continue

        axs = []
        for i, mod in enumerate(mods, start=1):
            sharex = axs[0] if axs else None
            ax = sf.add_subplot(gs[i, 0], sharex=sharex)
            axs.append(ax)

            defs = {
                "line_width": 2,
                "color": "k",
                "x_label": "time (s)",
                "plot_alpha": 0.5,
                "stim_types": ("CSpr", "CSpu", "CSm", "USp"),
                "title": {"title": mod, "style": "italic"},
                "y_dec": 2,
                "legend": False
            }

            local_kwargs = kwargs.copy()
            for key, val in defs.items():
                local_kwargs = update_kwargs(local_kwargs, key, val, force=True)

            ax_out, stim_types = plot_physio_with_events(
                lab_name,
                subject,
                task,
                mod,
                ax=ax,
                root_path=root_path,
                **local_kwargs
            )

            if ax_out is not None:
                all_plot_axes.append(ax_out)
                stim_types_last = stim_types

    fig.canvas.draw()

    if all_plot_axes and stim_types_last is not None:
        add_subject_legend(
            fig,
            all_plot_axes,
            ref_ax=task_title_axes[0],
            order=stim_types_last,
            pad_x=0.0,
            pad_y=-0.01,
            fontsize=10,
        )

    add_subject_title(
        fig,
        task_title_axes[0],
        subject,
        pad=0,
        fontsize=24,
        fontweight="bold",
    )

    return fig


def plot_physio_with_events(
        site: str,
        subject: str,
        task_name: str,
        modality: str,
        root_path: Union[str, Path]="Z:\\CALINET2\\converted",
        ax: Optional[Any]=None,
        chan_idx: int=-1,
        stim_types: Tuple[str, ...]=("CSpr", "USp", "CSpu", "CSm", "USo", "USm"),
        legend: bool=True,
        **kwargs: Any
    ) -> Tuple[Optional[Any], Tuple[str, ...]]:
    """
    Plot a physiology signal with event markers for a single modality.

    Parameters
    ----------
    site : str
        Dataset or site name.
    subject : str
        Subject identifier.
    task_name : str
        Task name.
    modality : str
        Recording modality (e.g., ``"scr"``, ``"eye"``).
    root_path : str or pathlib.Path, default="Z:\\CALINET2\\converted"
        Root directory containing data.
    ax : matplotlib.axes.Axes or None, optional
        Axis to plot into. If ``None``, a new figure and axis are created.
    chan_idx : int, default=-1
        Channel index to plot from the physiology dataframe.
    stim_types : tuple of str, default=(...)
        Event types to include in the plot.
    legend : bool, default=True
        Whether to display a legend for plotted events.
    kwargs : dict of str to Any
        Additional plotting arguments forwarded to ``LazyLine``.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        Axis containing the plot, or ``None`` if plotting was skipped.
    stim_types : tuple of str
        Stimulus types used for plotting.

    Notes
    -----
    This function reads physiology and event files, extracts metadata from a
    JSON sidecar, and plots waveform data with shaded event spans.

    Missing files or invalid configurations result in skipped plots with
    warnings.

    This function performs file I/O and modifies matplotlib state.
    """
    
    site_dir = os.path.join(root_path, site)
    subject_dir = os.path.join(site_dir, subject)
    physio_dir = os.path.join(subject_dir, "physio")

    base_name = f"{subject}_task-{task_name}"

    phys_file = os.path.join(
        physio_dir,
        f"{base_name}_recording-{modality}_physio.tsv.gz"
    )

    ev_file = os.path.join(
        physio_dir,
        f"{base_name}_events.tsv"
    )

    # create axis if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3.54))

    # skip missing physio file
    if not os.path.exists(phys_file):
        logger.warning(
            f"Missing physio file for subject={subject}, task={task_name}, "
            f"modality={modality}: {phys_file}. Skipping axis."
        )
        return None, stim_types

    # skip missing event file
    if not os.path.exists(ev_file):
        logger.warning(
            f"Missing event file for subject={subject}, task={task_name}: "
            f"{ev_file}. Skipping axis."
        )
        return None, stim_types

    units = None
    sr = None
    json_file = phys_file.replace(".tsv.gz", ".json")
    if os.path.exists(json_file):
        logger.info(f"Read SamplingFrequency from {json_file}")
        data = cio.load_json(json_file)
        sr = data.get("SamplingFrequency")
        logger.info(f"SamplingFrequency={sr}")
        
        if "eye" in modality:
            dict_w_units = data.get("pupil_size")
        else:
            dict_w_units = data.get(modality)

        if dict_w_units is not None:
            units = dict_w_units.get("Units")
            logger.info(f"Units for '{modality}'={units}")
    else:
        logger.warning(
            f"Could not find {json_file}. This is required to derive "
            f"'SamplingFrequency'. Skipping axis."
        )
        return None, stim_types

    logger.info(f"Read '{modality}'/'{task_name}' file: {phys_file}")
    df_phys = cio.read_physio_tsv_headerless(phys_file)
    df_ev = pd.read_csv(ev_file, sep="\t")

    if chan_idx > df_phys.shape[1] - 1:
        logger.warning(
            f"Column #{chan_idx+1} was requested for '{modality}', but "
            f"dataframe only has {df_phys.shape[1]} columns. Skipping axis."
        )
        return None, stim_types
    
    if sr is None:
        logger.warning(
            f"No sampling frequency found for subject={subject}, "
            f"task={task_name}, modality={modality}. Skipping axis."
        )
        return None, stim_types
    
    time_ax = np.arange(0, df_phys.shape[0]) / sr
    logger.info(f"'{modality}' channel index: {chan_idx}")

    y_lbl = "amplitude"
    if "y_label" not in kwargs:
        if units is not None:
            y_lbl = f"{y_lbl} [{units}]"
        kwargs["y_label"] = y_lbl
        
    pl = plotting.LazyLine(
        df_phys.iloc[:, chan_idx].to_numpy(),
        xx=time_ax,
        fontname="Arial",
        axs=ax,
        **kwargs
    )

    ax = pl.axs

    logger.info("Add events")
    for key, val in stim_colors.items():
        if key in stim_types:
            ev_df = df_ev.loc[df_ev["event_type"] == key]
            for j, onset in enumerate(ev_df["onset"]):
                plotting.add_axvspan(
                    ax,
                    loc=[onset, onset + 1],
                    color=val,
                    alpha=1,
                    ymax=0.3,
                    label=key if j == 0 else None
                )

    if legend:
        pl.axs.legend(frameon=False)

    return ax, stim_types


def add_screen_circle(
        ax,
        center_xy: Tuple[float, float],
        radius_px: int=60,
        **circle_kw: Any
    ) -> Any:
    """
    Add a circular overlay at a specified location in data coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to which the circle is added.
    center_xy : tuple of float
        Center of the circle in data coordinates.
    radius_px : int, default=60
        Radius of the circle in pixels.
    circle_kw : dict of str to Any
        Additional keyword arguments passed to ``matplotlib.patches.Circle``.

    Returns
    -------
    artist : matplotlib.offsetbox.AnnotationBbox
        Annotation box containing the circle.

    Notes
    -----
    The circle is rendered using a pixel-based ``DrawingArea`` to maintain
    consistent size regardless of axis scaling.

    This function modifies the provided axis and does not perform file I/O.
    """

    # A small pixel-based canvas
    da = DrawingArea(2*radius_px, 2*radius_px, 0, 0)
    c = plt.Circle((radius_px, radius_px), radius_px, fill=False, **circle_kw)
    da.add_artist(c)

    ab = AnnotationBbox(
        da, center_xy,           # anchor in data coords
        frameon=False,
        box_alignment=(0.5, 0.5),
        xycoords='data'
    )
    ax.add_artist(ab)
    return ab


def plot_gaze_hexbin_with_dva_circle(
        df: pd.DataFrame,
        screen_mm: Tuple[float, float],
        viewing_distance_mm: float,
        dva: float=5.0,
        center_mm: Optional[Tuple[float, float]]=None,
        gridsize: int=60,
        mincnt: int=1,
        figsize: Tuple[float, float]=(7.01, 5),
        ax: Optional[Any]=None,
        cbar: bool=False
    ) -> Tuple[Any, Any]:
    """
    Plot gaze data as a hexbin with a visual angle (DVA) circle overlay.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing gaze coordinates with columns
        ``"x_coordinate"`` and ``"y_coordinate"``.
    screen_mm : tuple of float
        Screen dimensions in millimeters as ``(width, height)``.
    viewing_distance_mm : float
        Viewing distance in millimeters.
    dva : float, default=5.0
        Radius of the circle in degrees of visual angle.
    center_mm : tuple of float or None, optional
        Circle center in millimeters. If ``None``, screen center is used.
    gridsize : int, default=60
        Number of hexagons in the x-direction.
    mincnt : int, default=1
        Minimum count per hexbin.
    figsize : tuple of float, default=(7.01, 5)
        Figure size if a new figure is created.
    ax : matplotlib.axes.Axes or None, optional
        Axis to plot into. If ``None``, a new figure is created.
    cbar : bool, default=False
        Whether to add a colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axis containing the plot.

    Notes
    -----
    Points outside the DVA circle are plotted as gray scatter, while points
    inside are shown as a hexbin density.

    This function modifies matplotlib state and may create a new figure.
    """

    w, h = map(float, screen_mm)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = df["x_coordinate"].to_numpy(dtype=float)
    y = df["y_coordinate"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    xv = x[valid]
    yv = y[valid]

    # ---- Circle geometry ----
    if center_mm is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = map(float, center_mm)

    radius_mm = viewing_distance_mm * np.tan(np.deg2rad(dva))

    # Distance from center
    dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)

    inside = dist <= radius_mm
    outside = ~inside

    # ---- Plot outside points in gray ----
    ax.scatter(
        xv[outside],
        yv[outside],
        color="lightgray",
        s=8,
        alpha=0.5,
        zorder=1
    )

    # ---- Hexbin only inside points ----
    hb = ax.hexbin(
        xv[inside],
        yv[inside],
        gridsize=gridsize,
        extent=[0, w, 0, h],
        mincnt=mincnt,
        zorder=2
    )

    # ---- Add circle ----
    circ = Circle(
        (cx, cy),
        radius_mm,
        fill=False,
        linestyle="--",
        linewidth=2,
        color="black",
        zorder=3
    )
    ax.add_patch(circ)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_xlabel("x (mm)")
    # ax.set_ylabel("y (mm)")

    if cbar:
        fig.colorbar(hb, ax=ax, label="count")

    return fig, ax


def qc_gaze_flag(
        df: pd.DataFrame,
        screen_size: Tuple[float, float],
        x_col: str="x_coordinate",
        y_col: str="y_coordinate",
        valid_mask: Optional[Union[np.ndarray, List[bool]]]=None,
        margin_px: int=10,
        outside_frac_thresh: float=0.05,
        central_q: float=0.95,
        central_outside_thresh: float=0.0,
        min_valid_samples: int=2000,
        center_offset_frac_thresh: float=0.25,
        spread_iqr_frac_min: float=0.05,
        spread_iqr_frac_max: float=0.95,
        swap_axes_check: bool=True,
        flip_y_check: bool=True,
        normalized_check: bool=True,
        degrees_check: bool=True,
        plot: bool=False,
        gridsize: int=70,
        title: Optional[str]=None,
        save_path: Optional[Union[str, Path]]=None,
        ax: Optional[Any]=None
    ) -> Dict[str, Any]:
    """
    Perform quality control checks on gaze coordinate data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing gaze coordinates.
    screen_size : tuple of float
        Screen size in pixels as ``(width, height)``.
    x_col : str, default="x_coordinate"
        Column name for x-coordinates.
    y_col : str, default="y_coordinate"
        Column name for y-coordinates.
    valid_mask : array-like of bool or None, optional
        Optional mask indicating valid samples.
    margin_px : int, default=10
        Pixel margin for bounds checks.
    outside_frac_thresh : float, default=0.05
        Threshold for fraction of points outside screen bounds.
    central_q : float, default=0.95
        Central quantile interval.
    central_outside_thresh : float, default=0.0
        Allowed tolerance outside bounds for central interval.
    min_valid_samples : int, default=2000
        Minimum number of valid samples required.
    center_offset_frac_thresh : float, default=0.25
        Threshold for median offset from screen center.
    spread_iqr_frac_min : float, default=0.05
        Minimum IQR fraction threshold.
    spread_iqr_frac_max : float, default=0.95
        Maximum IQR fraction threshold.
    swap_axes_check : bool, default=True
        Whether to check for swapped axes.
    flip_y_check : bool, default=True
        Whether to check for flipped y-axis.
    normalized_check : bool, default=True
        Whether to detect normalized coordinate ranges.
    degrees_check : bool, default=True
        Whether to detect degree-based coordinates.
    plot : bool, default=False
        Whether to generate a diagnostic plot.
    gridsize : int, default=70
        Hexbin grid size for plotting.
    title : str or None, optional
        Plot title.
    save_path : str or pathlib.Path or None, optional
        Path to save plot image if plotting is enabled.
    ax : matplotlib.axes.Axes or None, optional
        Axis to plot into.

    Returns
    -------
    qc : dict of str to Any
        Dictionary containing QC flag, reasons, and diagnostic metrics.

    Notes
    -----
    This function evaluates multiple heuristics including bounds violations,
    central mass, spread, normalization, and coordinate transformations.

    If ``plot=True``, a diagnostic figure is created and optionally written to
    disk.

    This function performs optional plotting and may write image files.
    """

    w, h = map(float, screen_size)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    if valid_mask is None:
        valid = finite
    else:
        valid_mask = np.asarray(valid_mask).astype(bool)
        valid = finite & valid_mask

    n_valid = int(valid.sum())
    if n_valid < min_valid_samples:
        out = {"flag": False, "reason": ["insufficient_data"], "n_valid": n_valid}
        if plot:
            print("Not enough valid samples to plot reliably.")
        return out

    xv = x[valid]
    yv = y[valid]

    # --- bounds checks ---
    xmin, xmax = 0 - margin_px, (w - 1) + margin_px
    ymin, ymax = 0 - margin_px, (h - 1) + margin_px

    outside = (xv < xmin) | (xv > xmax) | (yv < ymin) | (yv > ymax)
    outside_frac = float(outside.mean())

    # --- central mass interval ---
    lo = (1.0 - central_q) / 2.0
    hi = 1.0 - lo
    x_lo, x_hi = np.quantile(xv, [lo, hi])
    y_lo, y_hi = np.quantile(yv, [lo, hi])

    central_outside = (
        (x_lo < xmin - central_outside_thresh) or
        (x_hi > xmax + central_outside_thresh) or
        (y_lo < ymin - central_outside_thresh) or
        (y_hi > ymax + central_outside_thresh)
    )

    # --- center offset check ---
    cx, cy = w / 2.0, h / 2.0
    medx, medy = float(np.median(xv)), float(np.median(yv))
    center_offset = float(np.hypot(medx - cx, medy - cy))
    center_offset_thresh = center_offset_frac_thresh * min(w, h)
    center_far = center_offset > center_offset_thresh

    # --- spread checks (IQR) ---
    iqr_x = float(np.quantile(xv, 0.75) - np.quantile(xv, 0.25))
    iqr_y = float(np.quantile(yv, 0.75) - np.quantile(yv, 0.25))
    iqr_x_frac = iqr_x / w
    iqr_y_frac = iqr_y / h

    spread_too_small = (iqr_x_frac < spread_iqr_frac_min) or (iqr_y_frac < spread_iqr_frac_min)
    spread_too_large = (iqr_x_frac > spread_iqr_frac_max) or (iqr_y_frac > spread_iqr_frac_max)

    # --- unit/scale heuristics ---
    # normalized: most samples in [0,1] or [0,1] +/- tiny noise
    norm_like = False
    if normalized_check:
        in01 = ((xv >= -0.05) & (xv <= 1.05) & (yv >= -0.05) & (yv <= 1.05)).mean()
        norm_like = in01 > 0.95

    # degrees-like: ranges of only a few tens (common for degrees)
    deg_like = False
    if degrees_check:
        xr = float(np.nanpercentile(xv, 99) - np.nanpercentile(xv, 1))
        yr = float(np.nanpercentile(yv, 99) - np.nanpercentile(yv, 1))
        deg_like = (xr < 100.0 and yr < 100.0 and (w > 300 and h > 300))

    # --- swap / flip diagnostics ---
    swap_axes_suspect = False
    flip_y_suspect = False

    if swap_axes_check:
        # if x range fits height better and y fits width better (common swap signature)
        x_range = float(np.nanpercentile(xv, 99.5) - np.nanpercentile(xv, 0.5))
        y_range = float(np.nanpercentile(yv, 99.5) - np.nanpercentile(yv, 0.5))
        # compare normalized by respective screen dims
        x_fit_w = abs((x_range / w) - 0.5)
        x_fit_h = abs((x_range / h) - 0.5)
        y_fit_h = abs((y_range / h) - 0.5)
        y_fit_w = abs((y_range / w) - 0.5)
        # heuristic: x "fits" height better and y "fits" width better
        swap_axes_suspect = (x_fit_h + y_fit_w) < (x_fit_w + y_fit_h) - 0.2

    if flip_y_check:
        # If median y is very close to top edge while task expects center-ish.
        # This is only a weak heuristic; keep it as "suspect", not definitive.
        flip_y_suspect = (abs(medy - (h - 1 - cy)) < abs(medy - cy)) and center_far

    # --- decide flag ---
    reasons = []
    if outside_frac > outside_frac_thresh:
        reasons.append(f"outside_frac>{outside_frac_thresh:.3f}")
    if central_outside:
        reasons.append(f"central_{int(central_q*100)}%_outside")
    if center_far:
        reasons.append(f"median_far_from_center>{center_offset_thresh:.1f}px")
    if spread_too_small:
        reasons.append(f"spread_iqr_too_small<{spread_iqr_frac_min:.2f}")
    if spread_too_large:
        reasons.append(f"spread_iqr_too_large>{spread_iqr_frac_max:.2f}")
    if norm_like:
        reasons.append("coords_look_normalized_0to1")
    if deg_like:
        reasons.append("coords_look_like_degrees")
    if swap_axes_suspect:
        reasons.append("swap_axes_suspect")
    if flip_y_suspect:
        reasons.append("flip_y_suspect")

    flag = len(reasons) > 0

    # --- plotting ---
    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        hb = ax.hexbin(
            xv, yv,
            gridsize=gridsize,
            extent=[0, w, 0, h],
            mincnt=1
        )

        # screen bounds
        ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, linewidth=2))

        # central interval rectangle
        ax.add_patch(
            plt.Rectangle(
                (x_lo, y_lo),
                max(1e-6, x_hi - x_lo),
                max(1e-6, y_hi - y_lo),
                fill=False,
                linestyle="--",
                linewidth=2
            )
        )

        # median point
        ax.scatter([medx], [medy], s=40, marker="x")

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

        main_title = title if title is not None else "Gaze QC"
        subtitle = (
            f"outside={outside_frac:.3f} | "
            f"center_offset={center_offset:.1f}px | "
            f"IQRx={iqr_x_frac:.3f}, IQRy={iqr_y_frac:.3f}"
        )
        ax.set_title(main_title + ("\n" + subtitle), color=("red" if flag else "black"))

        # reason box
        txt = "OK" if not reasons else "FLAG:\n- " + "\n- ".join(reasons)
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.15)
        )

        plt.colorbar(hb, ax=ax, label="Count")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close()

    return {
        "flag": bool(flag),
        "reason": reasons if reasons else ["ok"],
        "n_valid": n_valid,
        "outside_frac": outside_frac,
        "bounds_used": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        "central_interval_x": (float(x_lo), float(x_hi)),
        "central_interval_y": (float(y_lo), float(y_hi)),
        "median_xy": (medx, medy),
        "center_xy": (cx, cy),
        "center_offset_px": center_offset,
        "iqr_frac_xy": (iqr_x_frac, iqr_y_frac),
        "normalized_like": bool(norm_like),
        "degrees_like": bool(deg_like),
        "swap_axes_suspect": bool(swap_axes_suspect),
        "flip_y_suspect": bool(flip_y_suspect),
        "minmax_x": (float(np.min(xv)), float(np.max(xv))),
        "minmax_y": (float(np.min(yv)), float(np.max(yv))),
    }
