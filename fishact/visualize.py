import numpy as np
import pandas as pd

import tsplot

def get_y_axis_label(df, signal):
    """
    Generate y-label for visualizations.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame as loaded from parse.load_data() or returned
        from parse.resample().
    signal : string
        String for what is on the y-axis

    Returns
    -------
    output : string
        y-axis label.
    """

    # Get approximate time interval of averages
    inds = df['fish']==df['fish'].unique()[0]
    exp_time = np.sort(df.loc[inds, 'exp_time'].values)
    dt = np.median(np.diff(exp_time)) * 60

    # Make y-axis label
    if 0.05 <= abs(dt - int(dt)) <= 0.95:
        return 'sec. of {0:s} in {1:.2f} min.'.format(signal, dt)
    else:
        return 'sec. of {0:s} in {1:d} min.'.format(signal, int(np.round(dt)))


def all_traces(df, signal='activity', summary_trace='mean', time_shift='center',
               alpha=0.75, hover_color='#535353', height=350, width=650,
               colors=None):
    """
    Generate a set of plots for each genotype.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame as loaded from parse.load_data() or returned
        from parse.resample().
    signal : string, default 'activity'
        Column of `df` that is used for the y-values in the plot.
    summary_trace : string, float, or None, default 'mean'
        Which summary statistic to use to make summary trace. If a
        string, can one of 'mean', 'median', 'max', or 'min'. If
        None, no summary trace is generated. If a float between
        0 and 1, denotes which quantile to show.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    alpha : float, default 0.75
        alpha value for individual time traces
    hover_color : string, default '#535353'
        Hex value for color when hovering over a curve
    height : int, default 200
        Height of each subplot plot in pixels.
    width : int, default 650
        Width of each subplot in pixels.
    colors : dict, default None
        colors[cat] is a 2-list containg, for category `cat`:
            colors[cat][0]: hex value for color of all time series
            colors[cat][1]: hex value for color of summary trace
        If none, colors are generated using paired ColorBrewer colors,
        with a maximum of six categories.

    Returns
    -------
    output : Bokeh grid plot
        Bokeh figure with subplots of all time series
    """
    # Make y-axis label
    y_axis_label = get_y_axis_label(df, signal)

    # Make plots
    p = tsplot.all_traces(
            df, 'exp_time', 'activity', 'fish', time_ind='exp_ind',
            light='light', summary_trace='mean', time_shift=time_shift,
            alpha=0.75, x_axis_label='time (hr)', y_axis_label=y_axis_label)

    return p


def grid(df, signal='activity', summary_trace='mean', time_shift='center',
         alpha=0.75, hover_color='#535353', height=200, width=650, colors=None):
    """
    Generate a set of plots for each genotype.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame as loaded from parse.load_data() or returned
        from parse.resample().
    signal : string, default 'activity'
        Column of `df` that is used for the y-values in the plot.
    summary_trace : string, float, or None, default 'mean'
        Which summary statistic to use to make summary trace. If a
        string, can one of 'mean', 'median', 'max', or 'min'. If
        None, no summary trace is generated. If a float between
        0 and 1, denotes which quantile to show.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    alpha : float, default 0.75
        alpha value for individual time traces
    hover_color : string, default '#535353'
        Hex value for color when hovering over a curve
    height : int, default 200
        Height of each subplot plot in pixels.
    width : int, default 650
        Width of each subplot in pixels.
    colors : dict, default None
        colors[cat] is a 2-list containg, for category `cat`:
            colors[cat][0]: hex value for color of all time series
            colors[cat][1]: hex value for color of summary trace
        If none, colors are generated using paired ColorBrewer colors,
        with a maximum of six categories.

    Returns
    -------
    output : Bokeh grid plot
        Bokeh figure with subplots of all time series
    """

    # Make y-axis label
    y_axis_label = get_y_axis_label(df, signal)

    # Make plots
    p = tsplot.grid(
            df, 'exp_time', 'activity', 'genotype', 'fish',
            time_ind='exp_ind', light='light', summary_trace=summary_trace,
            time_shift=time_shift, height=height, width=width,
            x_axis_label='time (hr)', y_axis_label=y_axis_label, colors=colors)

    return p


def summary(df, signal='activity', summary_trace='mean', time_shift='center',
            confint=True, ptiles=(2.5, 97.5), n_bs_reps=1000, alpha=0.35,
            height=350, width=650, colors=None, legend=True):
    """
    Generate a summary plot of the time courses.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame as loaded from parse.load_data() or returned
        from parse.resample().
    signal : string, default 'activity'
        Column of `df` that is used for the y-values in the plot.
    summary_trace : string, float, or None, default 'mean'
        Which summary statistic to use to make summary trace. If a
        string, can one of 'mean', 'median', 'max', or 'min'. If
        None, no summary trace is generated. If a float between
        0 and 1, denotes which quantile to show.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    confint : bool, default True
        If True, also display confidence interval.
    ptiles : list or tuple of length two, default (2.5, 97.5)
        Percentiles for confidence intervals; ignored if
        `confint` is False.
    n_bs_reps : int, default 1000
        Number of bootstrap replicates to use in conf. int. Ignored if
        `confint` is False.
    alpha : float, default 0.75
        alpha value for individual time traces
    hover_color : string, default '#535353'
        Hex value for color when hovering over a curve
    height : int, default 200
        Height of each subplot plot in pixels.
    width : int, default 650
        Width of each subplot in pixels.
    colors : dict, default None
        colors[cat] is a 2-list containg, for category `cat`:
            colors[cat][0]: hex value for color of all time series
            colors[cat][1]: hex value for color of summary trace
        If none, colors are generated using paired ColorBrewer colors,
        with a maximum of six categories.
    legend : bool, default True
        If True, show legend.

    Returns
    -------
    output : Bokleh plot
        Bokeh figure with summary plots
    """

    # Make y-axis label
    y_axis_label = get_y_axis_label(df, signal)

    p = tsplot.summary(
            df, 'exp_time', 'activity', 'genotype', 'fish',
            time_ind='exp_ind', light='light', summary_trace=summary_trace,
            time_shift=time_shift, confint=confint, ptiles=ptiles,
            n_bs_reps=n_bs_reps, alpha=0.25, height=height, width=width,
            x_axis_label='time', y_axis_label=y_axis_label, colors=colors,
            legend=legend)

    return p
