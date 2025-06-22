import os
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import gaussian_kde
from math import cos, sin, radians

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import plotly.graph_objects as go
from plotly.subplots import make_subplots


## Loading and shaping data functions
## ==================================

def get_dpath(mouse, session, frame):

    exp_path = '../Behavior/'
    dpath = os.path.join(exp_path, '{s}/{m}_{s}_{f}.dat'.format(m=mouse, s=session, f=frame))

    # exp_path = '/Users/joezaki/Desktop/BehaviorData/'
    # cohort = mouse[0]
    # dpath = os.path.join(exp_path, '{c}/DATfiles/{m}_{s}_{f}.dat'.format(c=cohort, m=mouse, s=session, f=frame))
    
    return dpath

# -----------------------------------------

def parse_data_file(
        dpath,
        header_rows=32,
        limit=None,
        sampling_rate=1/30,
        plot_dropped_frame_histogram=False,
        plot_bad_frame_histogram=False
        ):
    '''
    Loads in a dat file from Tracker and parses it into a pandas dataframe.

    Parameters
    ==========
    dpath : str
        Path to file of interest
    header_rows : int
        Number of intial rows comprise the header. These will be excluded. Default is 32.
    limit : int
        Number of consecutive NaN frames to tolerate. Default is None.
    sampling_rate : float
        Sampling rate of the time series recording, in seconds. Default is 1/30.
    plot_bad_frame_histogram : bool
        Whether or not to plot a histogram of the consecutive bad frame chunks. Default is False.
    plot_dropped_frame_histogram : bool
        Whether or not to plot a histogram of the consecutive dropped frame chunks. Default is False.
    '''

    data = pd.read_csv(dpath, sep='\t', skiprows=header_rows, header=None)
    colnames = ['Frame', 'Timestamp', 'X', 'Y', 'Sectors', 'State', 'CurrentLevel', 'MotorState', 'Flags', 'FrameInfo']
    data.columns = colnames

    if plot_dropped_frame_histogram:
        ms_delays, ms_delay_freq = np.unique((data['Timestamp'].diff().values[1:] * sampling_rate).astype(int), return_counts=True)
        fig = go.Figure(data=[go.Bar(x=ms_delays, y=ms_delay_freq)])
        fig.update_layout(template='simple_white', width=600, height=400, title_text=os.path.basename(dpath),
                        xaxis=dict(title_text='Number of Consecutive<br>Dropped Frames', dtick=1),
                        yaxis=dict(title_text='Frequency'))
        fig.show()

    # find lost frames and add them back in as nans
    idealized_frames = np.arange(1,data['Frame'].values[-1]+1)
    missing_frame_idx = idealized_frames[~np.isin(idealized_frames, data['Frame'].values)]
    missing_df = pd.DataFrame(columns = colnames)
    missing_df['Frame'] = missing_frame_idx
    data = pd.concat([data, missing_df])
    data = data.sort_values('Frame').reset_index()
    data.drop(columns='index', inplace=True)

    # find instances where tracking is not good
    bad_frame_bool = data['FrameInfo'] != 3
    data.loc[bad_frame_bool,'X'] = np.nan
    data.loc[bad_frame_bool,'Y'] = np.nan
    data.loc[bad_frame_bool,'State'] = np.nan

    # interpolate linearly for position and timestamp, fill with nearest for rest of data
    data['X'] = data['X'].infer_objects(copy=False).interpolate(method='linear', limit=limit)
    data['Y'] = data['Y'].infer_objects(copy=False).interpolate(method='linear', limit=limit)
    data['Timestamp'] = data['Timestamp'].infer_objects(copy=False).interpolate(method='linear', limit=limit)
    data['State'] = data['State'].infer_objects(copy=False).interpolate(method='nearest', limit=limit)
    data = data.infer_objects(copy=False).ffill(limit=limit).bfill(limit=limit) # in case first and/or last frames are invalid, it will fill with nearest good ones

    if plot_bad_frame_histogram:
        # plot a histogram representing the duration of each lapse of tracking (when state does not equal 3)
        num_consecutive, freq_consecutive = np.unique(np.unique(label(bad_frame_bool.values)[0], return_counts=True)[1][1:], return_counts=True)
        fig = go.Figure(data=[go.Bar(x=num_consecutive, y=freq_consecutive)])
        if limit is not None:
            fig.add_vline(x=limit, line_width=2, line_color='black', line_dash='dath', opacity=1)
        fig.update_layout(template='simple_white', width=400, height=400, title_text=os.path.basename(dpath),
                        yaxis_title='Frequency', xaxis_title='Number of Consecutive<br>Bad Frames')
        fig.show()

    data['Dist'] = np.append(0, np.sqrt((np.diff(data['Y']))**2 + (np.diff(data['X']))**2))
    
    return data

# -----------------------------------------

def getMeanBinnedVector(vector, bins, return_format='array'):
    """
    Breaks a vector up into bins, takes the mean of each bin, and returns that new vector
    """
    binned_vector = np.array_split(vector, bins)
    means = [np.mean(bin) for bin in binned_vector]

    if return_format == "list":
        return means
    elif return_format == "array":
        return np.array(means)
    else:
        raise Exception(
            "Invalid 'return_format' argument. Must be one of 'list' or 'array'."
        )

# -----------------------------------------

def getMeanBinnedVector_2d(matrix, bins, return_format="array"):
    """
    Breaks up a matrix into bins along the time axis (axis=1). Takes the mean of each bin along the same axis and returns the new matrix
    """
    binned_matrices = np.array_split(matrix, bins, axis=1)
    means = [np.mean(bin, axis=1) for bin in binned_matrices]
    if return_format == "list":
        return means
    elif return_format == "array":
        return np.array(means).T
    else:
        raise Exception(
            "Invalid 'return_format' argument. Must be one of 'list' or 'array'."
        )

# -----------------------------------------

## Plotting Functions
## ==================

# from: https://community.plotly.com/t/how-to-draw-a-filled-circle-segment/59583/2
# for plotting shock zone in occupancy plots
def degree2rad(degrees):
    return degrees*np.pi/180
def disk_part(center=[0,0], radius=1, start_angle=0, end_angle=90, n=50, seg=True):
    sta = degree2rad(start_angle)
    ea = degree2rad(end_angle)
    t = np.linspace(sta, ea, n)
    x = center[0] + radius*np.cos(t)
    y = center[1] +radius*np.sin(t)
    path = f"M {x[0]},{y[0]}"
    for xc, yc in zip(x[1:], y[1:]):
        path += f" L{xc},{yc}"
    if seg: #segment
        return path + " Z"
    else: #disk sector
        return path + f" L{center[0]},{center[1]} Z" #sector


def is_inside_shape(width, height, shape, shape_dims):
    """
    Checks if a point (width, height) is inside a shape (circle or rectangle).

    Args:
        width: x-coordinate of the point
        height: y-coordinate of the point
        shape: str of 'circle' or 'rectangle'
        shape_dims: dict of {x_center:_, y_center:_, circle_radius:_} if circle, or
                            {x0:_, y0:_, x1:_, y1:_} if rectangle
    
    Returns:
        True if the point is inside the shape, False otherwise.
    """

    if shape == 'circle':
        x_center = shape_dims['x_center']
        y_center = shape_dims['y_center']
        circle_radius = shape_dims['circle_radius']
        distance_squared = (width - x_center) ** 2 + (height - y_center) ** 2
        return distance_squared < circle_radius ** 2
    elif shape == 'rectangle':
        x0 = shape_dims['x0']
        y0 = shape_dims['y0']
        x1 = shape_dims['x1']
        y1 = shape_dims['y1']
        return x0 <= height <= x1 and y0 <= width <= y1


# -----------------------------------------

def draw_arena(
        start_angle,
        end_angle,
        radius=127.5,
        plot_type='heatmap',
        x0=0,
        y0=0,
        x1=255,
        y1=255,
        ):
    '''
    Creates plotly shapes for the circle of the arena and a wedge of the shock zone.

    Parameters
    ==========
    start_angle, end_angle : int or float
        Beginning and end angles of the shock zone wedge.
    radius : int or float
        Radius of the circle to be drawn. Default is 127.5.
    plot_type : str
        Type of plot this will be used for, to change whether shock_zone is plotted above or below.
        Should be one of 'heatmap', 'line', or 'scatter'. Default is 'heatmap'.
    x0, y0, x1, y1 : int or float
        The edges of the circle to be drawn. Defaults are 0, 0, 255, 255, respectively.
    '''
    arena = dict(type="circle", layer='above',
                  x0=x0, y0=y0, x1=x1, y1=y1, fillcolor="rgba(0,0,0,0)",
                  line=dict(color="black", width=1), opacity=1)
    shock_zone_shape = disk_part(center=[radius,radius], radius=radius, start_angle=start_angle, end_angle=end_angle, seg=False)
    shock_zone = dict(type="path", layer='above' if plot_type=='heatmap' else 'below',
                      path=shock_zone_shape,
                      fillcolor="lightpink", line=dict(color='crimson', width=2), opacity=0.5)

    return arena, shock_zone


def get_point_on_circle(
        angle,
        center=[0,0],
        radius=100
        ):
    '''
        Find the x,y coordinates on circle, based on given angle
    '''
    angle = radians(angle)
    x = center[0] + (radius * cos(angle))
    y = center[1] + (radius * sin(angle))

    return x,y

# -----------------------------------------

def get_polar_histogram(
        x,
        y,
        radius,
        hist_bars=50
        ):
    
    assert len(x) == len(y), 'x and y must be the same length.'
    
    angle_list = np.linspace(0, 360, hist_bars+1)
    coords_list = np.array([get_point_on_circle(center=[radius,radius], radius=radius, angle=angle) for angle in angle_list])
    polygon_list = [Polygon([(radius,radius), coord_from, coord_to]) for coord_from, coord_to in zip(coords_list[:-1], coords_list[1:])]

    points_df = pd.DataFrame()
    points_df['coords'] = list(zip(x,y))
    points_df['coords'] = points_df['coords'].apply(Point)
    points = gpd.GeoDataFrame(points_df, geometry='coords')

    polygons_df = pd.DataFrame({'angle':angle_list[:-1],
                                'polygon':polygon_list})
    polygons_df = pd.concat([polygons_df, polygons_df]).reset_index()
    polygons = gpd.GeoDataFrame(polygons_df, geometry='polygon')

    points_in_polygons_df = gpd.tools.sjoin(points, polygons, predicate="within", how='left')
    occupancy_counts = points_in_polygons_df['angle'].value_counts().sort_index()
    angles = occupancy_counts.index.values
    wedge_occupancy_fractions = occupancy_counts.values / len(x)

    return angles, wedge_occupancy_fractions

# -----------------------------------------

def get_occupancy_trace(
        x,
        y,
        plot_type,
        line_color='black',
        scatter_color='slategrey',
        scatter_size=5,
        heatmap_bins=100,
        hist_bars=50,
        bar_color='#FFAA70',
        bar_scale=1,
        colorscale='deep_r',
        radius=127.5,
        kde_radius=15
        ):
    '''
    Get occupancy trace of the mouse to feed into occupancy_plot().

    Parameters
    ==========
    x, y : 1d arrays
        1d arrays representing the x- and y-coordinates of the mouse across time.
    plot_type : str
        One of 'line', 'scatter', 'heatmap', or 'polar' to plot either a time series line,
        scatterpoints with a line connecting them, heatmap plot, or polar histogram of
        occupancy, respectively. Default is 'line'.
    line_color : str
        Color of the time series. Only used if plot_type=='line' or plot_type=='scatter'. Default is 'black'.
    scatter_color : str or list or 1d array
        Color(s) of the scatterpoints. Only used if plot_type=='scatter'. Default is 'slategrey'.
    scatter_size : int or float
        Size of the scatterpoints. Only used if plot_type=='scatter'. Default is 5.
    heatmap_bins : int
        Number of bins to break up the heatmap into. Only used if plot_type=='heatmap'. Default is 100.
    hist_bars : int
        Number of bars to break up the polar histogram into. Only used if plot_type=='polar'. Default is 50.
    bar_color : str or list
        Color(s) of the histogram bars. Only used if plot_type=='polar'.
    bar_scale : int or float
        How much spacing between bars in polar plot, with 1 being no spacing and larger numbers are proportionately
        more spacing. Only used if plot_type=='polar'. Default is 1.
    colorscale : str
        Colorscale of the heatmap or for scatter_colors. Only used if plot_type=='heatmap' or plot_type=='scatter'. Default is 'deep_r'.
    radius : float
        Number representing the radius of the whole circle. Default is 127.5.
    kde_radius : int or float
        Size of radius of circle to trim the kde matrix to fit into the circle where the mouse can freely roam. Default is 15.
    '''

    if (plot_type == 'line') | (plot_type == 'scatter'):
        plot_mode = 'lines' if plot_type=='line' else 'lines+markers'
        data_trace = go.Scattergl(
            x=x,
            y=y,
            mode=plot_mode,
            line=dict(width=1, color=line_color),
            marker=dict(color=scatter_color, colorscale=colorscale, size=scatter_size,
                        line=dict(color='black', width=0.5)))

    if plot_type == 'heatmap':
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min() : x.max() : heatmap_bins * 1j, y.min() : y.max() : heatmap_bins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # remove kde outside of the circle border
        circle_dims = {'x_center':radius, 'y_center':radius, 'circle_radius':radius-kde_radius}
        inside_circle_bool = [is_inside_shape(width, height, shape='circle', shape_dims=circle_dims) for \
                              width, height in zip(xi.flatten(), yi.flatten())]
        zi[np.invert(inside_circle_bool)] = np.nan

        data_trace = go.Heatmap(
            x=xi.flatten(),
            y=yi.flatten(),
            z=zi,
            colorscale=colorscale,
            showscale=False
            )
    
    if plot_type == 'polar':
        angles, wedge_occupancy_fractions = get_polar_histogram(x=x, y=y,
                                                                radius=radius, hist_bars=hist_bars)
        data_trace = go.Barpolar(
                r=wedge_occupancy_fractions,
                theta=angles,
                width=np.repeat((angles[1]-angles[0]), len(angles))/bar_scale,
                marker_color=bar_color,
                marker_line_color="black",
                marker_line_width=0.5)

    return data_trace


def occupancy_plot(
        data,
        start_angle,
        end_angle,
        radius,
        beg=0,
        end=None,
        sampling_rate=1/30,
        title=None,
        plot_type='line',
        hist_bars=50,
        save_path=None,
        **kwargs
        ):
    '''
    Plot occupancy of the mouse across the session, either as a line representing the time series, or
    as a heatmap of occupancy aggregated across time.

    Parameters
    ==========
    data : pandas dataframe
        data to be plotted, which must have columns named X and Y representing the coordinates sorted across time
    start_angle, end_angle : int or float
        Beginning and end angles of the shock zone wedge.
    radius : float
        Number representing the radius of the whole circle.
    beg, end : int
        The beginning and end of the total time to be plotted, in seconds, if subsetting is desired. By default,
        no subsetting is done.
    sampling_rate : float
        Sampling rate of the recording. Default is 1/30.
    title : str
        Title for the entire plot. Default is None.
    plot_type : str
        One of 'line', 'scatter', 'heatmap', or 'polar' to plot either a time series line, scatterpoints with a line connecting them,
        heatmap plot, or polar histogram plot, respectively. Default is 'line'.
    hist_bars : int
        Number of bars to break up the polar histogram into. Only used if plot_type=='polar'. Default is 50.
    save_path : str
        bsolute path to the directory where the file should be stored, including filename and extension. If this
        is not None, the plot will be saved; otherwise it won't be saved. Default is None.
    **kwargs
        Extra keyword arguments to be fed into the get_occupancy_plot() func above.
    '''

    # optionally subset time
    if end == None:
        time_range = np.arange(int(beg / sampling_rate), data.shape[0])
    else:
        time_range = np.arange(int(beg / sampling_rate), int(end / sampling_rate))

    x = data.X[time_range].values
    y = data.Y[time_range].values

    fig = go.Figure()

    # plot data
    data_trace = get_occupancy_trace(
        x=x,
        y=y,
        plot_type=plot_type,
        **kwargs
        )
    fig.add_trace(data_trace)

    # plot outline of box and shock zone
    if plot_type != 'polar':
        arena, shock_zone = draw_arena(radius=radius, start_angle=start_angle, end_angle=end_angle,
                                    plot_type=plot_type)
        fig.add_shape(arena)
        fig.add_shape(shock_zone)
    else:
        _, wedge_occupancy_fractions = get_polar_histogram(x=x, y=y, radius=radius, hist_bars=hist_bars)
        fig.add_trace(go.Barpolar(r=[np.max(wedge_occupancy_fractions)],
                                  theta=[(end_angle-start_angle)/2 + start_angle],
                                  width=[(end_angle-start_angle)],
                                  marker_color='lightpink',
                                  marker_line_color="crimson",
                                  marker_line_width=2,
                                  opacity=0.5))

    # configure plot
    fig.update_layout(template='simple_white', width=800, height=800,
                      title_text=title, title_x=0.5, showlegend=False,
                      yaxis=dict(visible=False, autorange='reversed', scaleanchor='x', scaleratio=1),
                      xaxis=dict(visible=False),
                      polar = dict(
                          radialaxis = dict(showline=False, showticklabels=False, ticks=''),
                          angularaxis = dict(direction='clockwise', rotation=0, showticklabels=False, ticks='')
                          ))
    
    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fig.write_image(save_path, scale=4)

    fig.show()

# -----------------------------------------

def plot_occupancy_list(
        dpath_list,
        frame,
        colors,
        subplot_titles,
        radius,
        start_angle,
        end_angle,
        title=None,
        sampling_rate = 1/30,
        plot_type = 'line',
        beg=0,
        end=None,
        cols=4,
        subplot_spacing=0.05,
        plot_width=1000,
        plot_height=1000,
        save_path=None,
        **kwargs
        ):
    """
    Creates a group of subplots of occupancy plots for a list of data paths provided.

    Parameters
    ==========
    dpath_list : list
        List of file paths to where the data of interest are stored.
    frame : str
        Which frame of reference the plot uses. One of 'Room' or 'Arena'.
    colors : list or str
        What color to plot each subplot. Must be the same length as dpath_list or a string.
    subplot_titles : list
        list of titles for each subplot.
    radius : float
        Number representing the radius of the whole circle.
    start_angle, end_angle : int or float
        Beginning and end angles of the shock zone wedge.
    title : str
        Title for the entire plot. Default is None.
    sampling_rate : float
        Sampling rate of the recording. Default is 1/30.
    plot_type : str
        One of 'line', 'scatter', or 'heatmap' to plot either a time series line, scatterpoints with a line connecting them,
        or heatmap plot, respectively. Default is 'line'.
    beg, end : int
        The beginning and end of the total time to be plotted, in seconds, if subsetting is desired. By default,
        no subsetting is done.
    cols : int
        Number of columns to separate the subplots into. Rows are calculated by default based on total
        number of subplots. Default is 4.
    subplot_spacing : float
        Vertical and horizontal spacing between subplots. Default is 0.05.
    plot_width, plot_height : int
        Width and height of the entire plot. Defaults are 1000.
    save_path : str
        bsolute path to the directory where the file should be stored, including filename and extension. If this
        is not None, the plot will be saved; otherwise it won't be saved. Default is None.
    **kwargs
        Extra keyword arguments to be fed into the get_occupancy_plot() func above.
    """


    rows = int(np.ceil(len(dpath_list) / cols))

    specs = [[{'type': 'polar'} for row in np.arange(cols)] for col in np.arange(rows)] if plot_type == 'polar' else None
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                        horizontal_spacing=subplot_spacing, vertical_spacing=subplot_spacing, specs=specs)
        
    for i, dpath in enumerate(dpath_list):
        if ((type(start_angle) == int)   & (type(start_angle) == int)) | \
           ((type(end_angle)   == float) & (type(start_angle) == float)):
            angles = {'start_angle':start_angle,
                      'end_angle':end_angle}
        elif (type(start_angle) == list) & (type(end_angle) == list):
            angles = {'start_angle':start_angle[i],
                      'end_angle':end_angle[i]}
        else:
            raise Exception('Invalid data types for "start_angle" and "end_angle". Both must either be lists or scalar values.')
        
        row = int(i / cols) + 1
        col = (i % cols) + 1

        data = parse_data_file(dpath)

        if end == None:
            time_range = np.arange(int(beg / sampling_rate), data.shape[0])
        else:
            time_range = np.arange(int(beg / sampling_rate), int(end / sampling_rate))
        
        x = data['X'][time_range].values
        y = data['Y'][time_range].values
        kwargs['line_color'] = colors if type(colors) == str else colors[i] if type(colors) == list else \
            Exception('Invalid "colors" argument. Must be a string representing a color or a list of colors.')
        kwargs['bar_color'] = colors if type(colors) == str else colors[i] if type(colors) == list else \
            Exception('Invalid "colors" argument. Must be a string representing a color or a list of colors.')

        data_trace = get_occupancy_trace(
            x=x,
            y=y,
            plot_type=plot_type,
            radius=radius,
            **kwargs
            )
        fig.add_trace(data_trace, row=row, col=col)

        # plot outline of box and shock zone
        if plot_type != 'polar':
            arena, shock_zone = draw_arena(plot_type=plot_type,
                                           start_angle=angles['start_angle'],
                                           end_angle=angles['end_angle'])
            fig.add_shape(arena, row=row, col=col)
            fig.add_shape(shock_zone, row=row, col=col)
            fig.update_xaxes(visible=False,
                             scaleanchor='x', scaleratio=1, row=row, col=col)
            fig.update_yaxes(autorange='reversed', visible=False, 
                             scaleanchor='x', scaleratio=1, row=row, col=col)
        else:
            _, wedge_occupancy_fractions = get_polar_histogram(x=x, y=y, radius=radius)
            fig.add_trace(go.Barpolar(r=[np.max(wedge_occupancy_fractions)],
                                    theta=[(angles['end_angle']-angles['start_angle'])/2 + angles['start_angle']],
                                    width=[(angles['end_angle']-angles['start_angle'])],
                                    marker_color='lightpink',
                                    marker_line_color="crimson",
                                    marker_line_width=2,
                                    opacity=0.5),
                                    row=row, col=col)
            fig.update_polars(
                radialaxis = dict(showline=False, showticklabels=False, ticks=''),
                angularaxis = dict(direction='clockwise', rotation=0, showticklabels=False, ticks=''),
                row=row, col=col)

    if title is None:
        title = 'Occupancy trace from {f} frame'.format(f=frame)
    fig.update_layout(template='simple_white', width=plot_width, height=plot_height,
                    showlegend=False, title_x=0.5,
                    title_text=title)
    
    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fig.write_image(save_path, scale=4)
    
    fig.show(renderer='notebook_connected')