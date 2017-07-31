import numpy as np
import pandas as pd

import tsplot

def ecdf(data, formal=False, buff=0.1, min_x=None, max_x=None):
    """
    Generate `x` and `y` values for plotting an ECDF.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.
    formal : bool, default False
        If True, generate `x` and `y` values for formal ECDF.
        Otherwise, generate `x` and `y` values for "dot" style ECDF.
    buff : float, default 0.1
        How long the tails at y = 0 and y = 1 should extend as a
        fraction of the total range of the data. Ignored if
        `formal` is False.
    min_x : float, default None
        Minimum value of `x` to include on plot. Overrides `buff`.
        Ignored if `formal` is False.
    max_x : float, default None
        Maximum value of `x` to include on plot. Overrides `buff`.
        Ignored if `formal` is False.

    Returns
    -------
    x : array
        `x` values for plotting
    y : array
        `y` values for plotting
    """

    if formal:
        return _ecdf_formal(data, buff=buff, min_x=min_x, max_x=max_x)
    else:
        return _ecdf_dots(data)


def _ecdf_dots(data):
    """
    Compute `x` and `y` values for plotting an ECDF.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.

    Returns
    -------
    x : array
        `x` values for plotting
    y : array
        `y` values for plotting
    """
    return np.sort(data), np.arange(1, len(data)+1) / len(data)


def _ecdf_formal(data, buff=0.1, min_x=None, max_x=None):
    """
    Generate `x` and `y` values for plotting a formal ECDF.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.
    buff : float, default 0.1
        How long the tails at y = 0 and y = 1 should extend as a fraction
        of the total range of the data.
    min_x : float, default None
        Minimum value of `x` to include on plot. Overrides `buff`.
    max_x : float, default None
        Maximum value of `x` to include on plot. Overrides `buff`.

    Returns
    -------
    x : array
        `x` values for plotting
    y : array
        `y` values for plotting
    """
    # Get x and y values for data points
    x, y = _ecdf_dots(data)

    # Set defaults for min and max tails
    if min_x is None:
        min_x = x[0] - (x[-1] - x[0])*buff
    if max_x is None:
        max_x = x[-1] + (x[-1] - x[0])*buff

    # Set up output arrays
    x_formal = np.empty(2*(len(x) + 1))
    y_formal = np.empty(2*(len(x) + 1))

    # y-values for steps
    y_formal[:2] = 0
    y_formal[2::2] = y
    y_formal[3::2] = y

    # x- values for steps
    x_formal[0] = min_x
    x_formal[1] = x[0]
    x_formal[2::2] = x
    x_formal[3:-1:2] = x[1:]
    x_formal[-1] = max_x

    return x_formal, y_formal


def get_y_axis_label(df, signal, time_unit=None):
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

    if time_unit is None:
        if signal == 'sleep':
            time_unit = 'min.'
        elif signal == 'activity':
            time_unit = 'sec.'

    # Get approximate time interval of averages
    inds = df['fish']==df['fish'].unique()[0]
    exp_time = np.sort(df.loc[inds, 'exp_time'].values)
    dt = np.median(np.diff(exp_time)) * 60

    # Make y-axis label
    if 0.05 <= abs(dt - int(dt)) <= 0.95:
        return '{0:s} of {1:s} in {2:.2f} min.'.format(time_unit, signal, dt)
    else:
        return '{0:s} of {1:s} in {2:d} min.'.format(time_unit, signal,
                                                     int(np.round(dt)))


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
            df, 'exp_time', signal, 'fish', time_ind='exp_ind',
            light='light', summary_trace='mean', time_shift=time_shift,
            alpha=0.75, x_axis_label='time (hr)', y_axis_label=y_axis_label)

    return p


def grid(df, signal='activity', summary_trace='mean',  gtype_order=None,
         time_shift='center', alpha=0.75, hover_color='#535353', height=200,
         width=650, colors=None):
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
    gtype_order : list or tuple, default None
        A list of the order of the genotypes to use in the plots. Each
        entry must be in df['genotype']. If None,
        df['genotype'].unique() is used.
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
            df, 'exp_time', signal, 'genotype', 'fish', cats=gtype_order,
            time_ind='exp_ind', light='light', summary_trace=summary_trace,
            time_shift=time_shift, height=height, width=width,
            x_axis_label='time (hr)', y_axis_label=y_axis_label, colors=colors)

    return p


def summary(df, signal='activity', summary_trace='mean', gtype_order=None,
            time_shift='center', confint=True, ptiles=(2.5, 97.5),
            n_bs_reps=1000, alpha=0.35, height=350, width=650, colors=None,
            legend=True):
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
    gtype_order : list or tuple, default None
        A list of the order of the genotypes to use in the plots. Each
        entry must be in df['genotype']. If None,
        df['genotype'].unique() is used.
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
            df, 'exp_time', signal, 'genotype', 'fish', cats=gtype_order,
            time_ind='exp_ind', light='light', summary_trace=summary_trace,
            time_shift=time_shift, confint=confint, ptiles=ptiles,
            n_bs_reps=n_bs_reps, alpha=0.25, height=height, width=width,
            x_axis_label='time', y_axis_label=y_axis_label, colors=colors,
            legend=legend)

    return p
