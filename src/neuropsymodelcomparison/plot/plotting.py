"""
This module contains functions for creating figures.

This file originated from the online analysis project at:
https://github.com/OlafHaag/UCM-WebApp
"""
import itertools
import string

import pandas as pd
import numpy as np
from plotly import express as px, graph_objs as go, figure_factory as ff
from plotly.subplots import make_subplots
import scipy

from ..dataprocessing import analysis

theme = {'graph_margins': {'l': 40, 'b': 40, 't': 40, 'r': 10},
         # Use colors consistently to quickly grasp what is what.
         'df1': 'cornflowerblue',
         'df2': 'palevioletred',
         'df1|df2': 'mediumpurple',
         'sum': 'peru',
         'parallel': 'lightgreen',
         'orthogonal': 'salmon',
         'pre': 'mediumseagreen',
         'post': 'sandybrown',
         'colors': px.colors.qualitative.Plotly,
         }

task_order = pd.Series({'pre': 0, 'df1': 1, 'df2': 2, 'df1|df2': 3, 'post': 4})


def get_ellipse_coordinates(x_center=0, y_center=0, axis=(1, 0), a=1, b=1, n=100):
    """ Helper function to generate coordinates for drawing an ellipse.
    
    :param x_center: X coordinates of ellipse center.
    :param y_center: Y coordinates of ellipse center.
    :param axis: Ellipse main axis direction.
    :param a: Ellipse parameter for axis.
    :param b: Ellipse parameter for orthogonal axis.
    :param n: Resolution.
    :return: X and y path coordinates for ellipse. Plot with scatter and mode set to lines.
    """
    # We work with unit vectors.
    if not np.isclose(np.linalg.norm(axis), 1):
        axis = np.array(axis)/np.linalg.norm(axis)
    axis2 = analysis.get_orthogonal_vec2d(axis)
    t = np.linspace(0, 2 * np.pi, n)
    # Ellipse parameterization with respect to a system of axes of directions defined by axis.
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    # Construct rotation matrix.
    rot_mat = np.array([axis, axis2]).T
    # Coordinates of the points with respect to the cartesian system with basis vectors [1, 0], [0,1].
    xp, yp = np.dot(rot_mat, [xs, ys])
    # Move to center coordinates.
    x = xp + x_center
    y = yp + y_center
    return x, y


def generate_trials_figure(df, marker_opacity=0.7, marker_size=10, contour_data=None, **layout_kwargs):
    """ Scatter plot of data.
    
    :param df: Data
    :type df: pandas.DataFrame
    :param contour_data: outlier visualisation data.
    :type: numpy.ndarray
    :rtype: plotly.graph_objs.figure
    """
    if df.empty:
        data = []
    else:
        data = [go.Scatter(
            x=df[df['task'] == t]['df1'],
            y=df[df['task'] == t]['df2'],
            text=[f"df1={j['df1']:.2f}<br />df2={j['df2']:.2f}<br />Sum={j['sum']:.2f}<br />Participant {j['user']}"
                  f"<br />Session {j['session']}<br />Block {t}<br />Trial {j['trial']}"
                  for _, j in df[df['task'] == t].iterrows()],
            hoverinfo='text',
            mode='markers',
            opacity=marker_opacity,
            marker={'size': marker_size,
                    'color': theme[t],
                    'line': {'width': 0.5, 'color': 'white'}
                    },
            name=f"Task {t}",
        ) for t in task_order[df['task'].unique()].sort_values().index]

    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            xaxis={'title': 'Degree of Freedom 1', 'range': [0, 100], 'constrain': 'domain'},
            yaxis={'title': 'Degree of Freedom 2', 'range': [0, 100], 'scaleanchor': 'x', 'scaleratio': 1},
            margin=theme['graph_margins'],
            legend=legend,
            hovermode='closest',
        )
    )
    # Task goal 1 visualization.
    fig.add_trace(go.Scatter(
        x=[25, 100],
        y=[100, 25],
        mode='lines',
        name="task goal 1",
        marker={'color': 'black',
                },
        hovertemplate="df1+df2=125",
    ))
    # Task goal 2 (DoF constrained) visualization.
    fig.add_scatter(y=[62.5], x=[62.5],
                    name="task goal 2",
                    text=["task goal 2 (df1=df2)"],
                    hoverinfo='text',
                    mode='markers',
                    marker_symbol='x',
                    opacity=0.7,
                    marker={'size': 25,
                            'color': 'red'})

    # Add visualisation for outlier detection.
    if isinstance(contour_data, np.ndarray):
        fig.add_trace(go.Contour(z=contour_data,
                                 name='outlier threshold',
                                 line_color='black',
                                 contours_type='constraint',
                                 contours_operation='=',
                                 contours_value=-1,
                                 contours_showlines=False,
                                 line_width=1,
                                 opacity=0.25,
                                 showscale=False,
                                 showlegend=True,
                                 hoverinfo='skip',
                                 ))

    fig.update_xaxes(hoverformat='.2f')
    fig.update_yaxes(hoverformat='.2f')
    fig.update_layout(**layout_kwargs)
    return fig


def get_pca_annotations(pca_dataframe):
    """ Generate display properties of principal components for graphs.
    
    :param pca_dataframe: Results of PCA.
    :type pca_dataframe: pandas.DataFrame
    :return: List of properties for displaying arrows.
    :rtype: list
    """
    # Visualize the principal components as vectors over the input data.
    arrows = list()
    # Each block displays its principal components.
    try:
        for name, group in pca_dataframe.groupby('task'):
            vectors = analysis.get_pca_vectors(group)  # origin->destination pairs.
            
            arrows.extend([dict(
                ax=v0[0],
                ay=v0[1],
                axref='x',
                ayref='y',
                x=v1[0],
                y=v1[1],
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=theme[name]  # Match arrow color to block.
            )
                for v0, v1 in vectors])
    except KeyError:
        pass
    return arrows


def add_pca_ellipses(fig, pca_dataframe, size=2.0):
    """ Get data for drawing ellipses around data for each block.
    Ellipses are only scaled by explained variance, not by spread of the actual data.
    
    :param fig: Figure to add ellipses to.
    :param pca_dataframe: Tabular results of PCA.
    :param float size: Size of ellipses is determined setting the semi-major & semi-minor axes to the the square root
                       of explained variance by the principal components and then multiplying them with this size value.
                       Default is 2.0.
    """
    # Each block displays its principal components.
    try:
        for name, group in pca_dataframe.groupby('task'):
            x, y = get_ellipse_coordinates(*group[['meanx', 'meany']].iloc[0],
                                           axis=group[['x', 'y']].iloc[0],
                                           a=np.sqrt(group['var_expl'].iloc[0])*size,
                                           b=np.sqrt(group['var_expl'].iloc[1])*size,
                                           )
            fig.add_scatte(x=x,
                           y=y,
                           mode='lines',
                           line_color=theme[name],
                           showlegend=False,
                           hoverinfo='skip',
                          )
    except (KeyError, IndexError):
        pass


def generate_histograms(dataframe, by=None, x_title="", legend_title=None, **layout_kwargs):
    """ Plot distribution of data to visually check for normal distribution.
    
    :param dataframe: Data for binning. When by is given this must be only 2 columns with one of them being the grouper.
    :type dataframe: pandas.DataFrame
    :param by: dataframe column to group data by.
    :type by: str
    :param legend_title:
    :type legend_title: str
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title=legend_title,
    )
    
    if dataframe.empty:
        fig = go.Figure()
        fig.update_xaxes(range=[0, 100])
    else:
        if not by:
            # Columns we want to plot histograms for. Display order is reversed.
            data = [dataframe[c] for c in dataframe.columns]
            try:
                colors = [theme[c] for c in dataframe.columns]
            except KeyError:
                colors = [theme['colors'][i + 1] for i in range(len(dataframe.colums))]
            # Create distplot with curve_type set to 'normal', overriding default 'kde'.
            fig = ff.create_distplot(data,  dataframe.columns, colors=colors, curve_type='normal')
        else:
            data = list()
            labels = list()
            colors = list()
            grouped = dataframe.groupby(by)
            for i, (name, df) in enumerate(grouped):
                data.append(df.drop(columns=by).squeeze(axis='columns'))
                labels.append(f"{by.capitalize()} {name}")  # Potential risk when 'by' is a list.
                # Set theme colors for traces.
                try:
                    color = theme[name]
                except KeyError:
                    color = theme['colors'][i+1]
                colors.append(color)
            fig = ff.create_distplot(data,  labels, colors=colors, curve_type='normal')  # Override default 'kde'.

    if not x_title:
        x_title = "Value"

    fig.layout.update(legend=legend,
                      yaxis={'title': 'Probability Density'},
                      xaxis={'title': x_title},
                      margin=theme['graph_margins'],
                      **layout_kwargs,
                      )
    return fig


def generate_histogram_rows(dataframe, x_title="", rows='task', legend_title=None, **layout_kwargs):
    """ Plot distributions of data by value given by 'rows'.

    :param dataframe: Data.
    :type dataframe: pandas.DataFrame
    :param x_title: X-axis title, unit of measurement, defaults to "".
    :type x_title: str, optional
    :param rows: Column name in data by which to split into rows, defaults to "task".
    :type rows: str, optional
    :param legend_title: Title of the legend, defaults to None.
    :type legend_title: [None, str], optional
    """
    fig = px.histogram(dataframe,
                       barmode='overlay',
                       histnorm='probability density',
                       facet_row=rows,
                       opacity=0.7,
                       category_orders={'condition':['df1', 'df2', 'df1|df2'],
                                        'task':task_order.index.tolist()},
                       color_discrete_map=theme)
    legend = go.layout.Legend(
                            xanchor='left',
                            yanchor='top',
                            orientation='v',
                            title=legend_title
                            )
    fig.update_layout(xaxis_title=x_title, legend=legend, margin=theme['graph_margins'])
    fig.update_xaxes(hoverformat='.2f')
    fig.update_yaxes(hoverformat='.2f', title="")
    fig.update_layout(
                      # keep the original annotations and add a list of new annotations:
                      annotations = list(fig.layout.annotations) + 
                      [go.layout.Annotation(x=-0.1,
                                            y=0.5,
                                            font=dict(
                                                size=14
                                            ),
                                            showarrow=False,
                                            text="Probalitiy Density",
                                            textangle=-90,
                                            xref="paper",
                                            yref="paper"
                                           )
                      ]
                      )
    return fig


def generate_means_scatterplot(dataframe, **layout_kwargs):
    """[summary]

    :param dataframe: [description]
    :type dataframe: [type]
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title="Task",
    )
    
    if dataframe.empty:
        fig = go.Figure()
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])
    else:
        fig = px.scatter(dataframe, x='meanx', y='meany', symbol='task', color='user',
                         hover_data=['task', 'user', 'meanx', 'meany'])

    fig.update_xaxes(hoverformat='.2f')
    fig.update_yaxes(hoverformat='.2f')
    fig.update_layout(yaxis_scaleanchor='x',
                      yaxis_scaleratio=1,
                      **layout_kwargs)
    
    return fig

def generate_pca_figure(dataframe, value='var_expl_ratio', error=None, **layout_kwargs):
    """ Plot explained variance by principal components as Bar plot with cumulative explained variance.
    
    :param dataframe: Results of PCA.
    :type dataframe: pandas.DataFrame
    :return: Properties of plot as a dictionary.
    :rtype: dict
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title={'text': 'PC'},
    )
    
    layout = dict(
        legend=legend,
        yaxis={'title': 'Explained Variance in Percent'},
        xaxis={'title': 'Task'},
        margin={'l': 60, 'b': 40, 't': 40, 'r': 10},
        #hovermode=False,
    )
    
    try:
        fig = px.bar(dataframe, x='task', y=value, error_y=error, barmode='group', color='PC',
                     category_orders={
                         'task': list(task_order[dataframe['task'].unique()].sort_values().index)
                     })
    except (KeyError, ValueError):
        fig = go.Figure()
    else:
        fig.update_yaxes(hoverformat='.2f')
        #fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    finally:
        fig.layout.update(**layout, **layout_kwargs)
    
    return fig


def generate_means_figure(dataframe, variables=None, **layout_kwargs):
    """ Barplots for variables grouped by block.
    Variable for each user is plotted as well as mean over all users.
    
    :param dataframe: Data with variables.
    :type dataframe: pandas.DataFrame
    :param variables: Variables to plot by block. List of dicts with 'label' and 'var' keys.
    :type variables: list[dict]
    :return: Figure object.
    """
    if not variables:
        variables = [{'label': 'Sum Variance', 'var': 'sum variance'},
                     {'label': 'Sum Mean', 'var': 'sum mean'}]
        
    fig = make_subplots(rows=len(variables), cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_titles=list(string.ascii_uppercase[:len(variables)]))  # A, B, C, ...
    # Handle empty data.
    if dataframe.empty:
        fig.layout.update(xaxis2_title='Task',
                          margin=theme['graph_margins'],
                          )
        # Empty dummy traces.
        for i, v in enumerate(variables):
            fig.add_trace({}, row=i+1, col=1)
            fig.update_yaxes(title_text=v['label'], hoverformat='.2f', row=i+1, col=1)
        fig.update_xaxes(tickvals=task_order.values, ticktext=task_order.index,
                         row=len(variables), col=1)
        return fig
    
    # Subplots for variables.
    grouped = dataframe.groupby('user')
    # Variance plot.
    for i, v in enumerate(variables):
        for name, group in grouped:
            fig.add_trace(go.Bar(
                x=group['task'].map(task_order),
                y=group[v['var']],
                name=f'Participant {name}',
                showlegend=False,
                marker={'color': group['task'].map(theme),
                        'opacity': 0.5},
                hovertemplate="%{text}: %{y:.2f}",
                text=[v['label']] * len(group),
            ),
                row=i+1, col=1,
            )
    
        # Add mean across participants by block
        means = analysis.get_mean(dataframe, column=v['var'], by='task')
        for task, value in means.iteritems():
            fig.add_trace(go.Scatter(
                x=[task_order[task] - 0.5, task_order[task], task_order[task] + 0.5],
                y=[value, value, value],
                name=f"Task {task}",
                showlegend=(not i),  # Show legend only for first trace to prevent duplicates.
                hovertext=f'Task {task}',
                hoverinfo='y',
                textposition='top center',
                mode='lines',
                marker={'color': theme[task]},
                hovertemplate=f"Mean {v['label']}: {value:.2f}",
            ), row=i+1, col=1)
        fig.update_yaxes(title_text=v['label'], hoverformat='.2f', row=i+1, col=1)
    fig.update_xaxes(tickvals=task_order[dataframe['task'].unique()],
                     ticktext=task_order[dataframe['task'].unique()].index,
                     title_text="Task", row=len(variables), col=1)

    # Layout
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        itemsizing='constant',
        title={'text': 'Mean'},
    )
    
    fig.update_layout(
        barmode='group',
        bargap=0.15,  # Gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # Gap between bars of the same location coordinate.
        margin=theme['graph_margins'],
        legend=legend,
        hovermode='closest',
        **layout_kwargs,
    )
    return fig


def generate_violin_figure(dataframe, columns, ytitle, legend_title=None, **layout_kwargs):
    """ Plot 2 columns of data as violin plot, grouped by block.

    :param dataframe: Variance of projections.
    :type dataframe: pandas.DataFrame
    :param columns: 2 columns for the negative and the positive side of the violins.
    :type columns: list
    :param ytitle: Title of Y-axis. What is being plotted? What are the units of the data?
    :type ytitle: str
    :param legend_title: What's the common denominator of the columns?
    :type legend_title: str

    :return: Figure object of graph.
    :rtype: plotly.graph_objs.Figure
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title=legend_title,
    )
    
    fig = go.Figure()
    fig.layout.update(xaxis_title='Task',
                      yaxis_title=ytitle,
                      legend=legend,
                      margin=theme['graph_margins'])
    if dataframe.empty:
        return fig
    
    # Make sure we plot only 2 columns, left and right.
    columns = columns[:2]
    sides = ('negative', 'positive')
    grouped = dataframe.groupby('task')
    for name, group_df in grouped:
        for i, col in enumerate(columns):
            fig.add_trace(go.Violin(x=group_df['task'].map(task_order),
                                    y=group_df[col],
                                    legendgroup=col, scalegroup=col, name=col,
                                    side=sides[i],
                                    pointpos=i - 0.5,
                                    line_color=theme[col],
                                    text=[f"{col}<br />participant: {j['user']}<br />"
                                          f"block: {j['block']}<br />condition: {j['condition']}"
                                          for _, j in group_df.iterrows()],
                                    hoverinfo='y+text',
                                    spanmode='hard',
                                    showlegend=bool(name == dataframe['task'].unique()[0]),  # Only 1 legend.
                                    )
                          )
    
    # update characteristics shared by all traces
    fig.update_traces(meanline={'visible': True, 'color': 'dimgray'},
                      box={'visible': True, 'width': 0.5, 'line_color': 'dimgray'},
                      points='all',  # Show all points.
                      jitter=0.1,  # Add some jitter on points for better visibility.
                      scalemode='count')  # Scale violin plot area with total count.

    fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay', hovermode='closest', **layout_kwargs)
    fig.update_xaxes(tickvals=task_order[dataframe['task'].unique()],
                     ticktext=task_order[dataframe['task'].unique()].index)
    fig.update_yaxes(hoverformat='.2f', zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
    return fig


def generate_lines_plot(dataframe, y_var, errors=None, by='user', facets='condition', color_col=None, jitter=False,
                        **layout_kwargs):
    """ Intended for use with either df1/df2 or parallel/orthogonal.
    
    :param dataframe: Values to plot
    :type dataframe: pandas.DataFrame
    :param y_var: Y-axis variable. What is being plotted?
    :type y_var: str
    :param errors: column name for standard deviations.
    :type errors: str|None
    :param by: Line group. What any column in dataframe will be grouped by.
    :type by: str
    :param facets: Separate into several subplots by this variable.
    :type facets: str
    :param color_col: Column containing keys for colors from theme.
    :type color_col: str
    :param jitter: Make some horizontal space between line plots.
    :type jitter: bool

    :return: Figure object of graph.
    :rtype: plotly.graph_objs.Figure
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title=color_col.upper(),
    )
    
    try:
        x_range = [dataframe['block'].unique().astype(float).min() - 0.5,
                   dataframe['block'].unique().astype(float).max() + 0.5]
        
        # When all error bars overlap they just clutter the graph without being able to distinguish them.
        # Therefore, introduce some jitter in the x-direction. Isn't perfect, but must suffice for now.
        if jitter:
            # Separate the colored groups.
            i = itertools.count(1)  # Counter for current group.
            d = 0.4  # Spread to the left and right of each block.
            s = d / (dataframe[color_col].nunique() + 1)  # Incremental distance for each group from the min (block -d).
            n_groups = dataframe[color_col].nunique()
            dataframe['x'] = dataframe['block'].astype(float) \
                + s * dataframe.groupby(color_col)['block'].transform(lambda _: -n_groups - 2 + 2 * next(i))
            # Now spread the users apart a bit within each group.
            by_idx = pd.Series(np.arange(dataframe.user.nunique())+1, index=dataframe.user.unique())
            try:
                dataframe['x'] += s * (1 + dataframe[by].astype(int).map(by_idx)) / dataframe[by].nunique()
            except ValueError:  # Division by 0.
                dataframe['x'] = dataframe['block']
        else:
            dataframe['x'] = dataframe['block']

        hover_data = {by: False,
                      y_var: ':.3f',
                      errors: ':.3f',
                      'participant': (True, dataframe['user'].values),
                      'block_': (True, dataframe['block'].values),
                      'task_': (True, dataframe['task'].values),
                      'x': False}
        if errors is None:
            del hover_data[errors]
        fig = px.line(data_frame=dataframe, x='x', y=y_var, error_y=errors,
                      line_group=by, facet_col=facets, facet_col_wrap=0, color=color_col, color_discrete_map=theme,
                      hover_data=hover_data, labels={'block_': 'block', 'task_': 'task'},
                      range_x=x_range,
                      render_mode='svg', 
                      **layout_kwargs)
        fig.update_xaxes(title="Block", tickvals=dataframe['block'].unique())
    except (KeyError, ValueError) as e:
        fig = go.Figure()
    
    # We have a variable number of subplots. Capitalize all x-axis titles.
    xaxes = [fig.layout[e] for e in fig.layout if e[0:5] == 'xaxis']
    try:
        for ax in xaxes:
            ax['title']['text'] = ax['title']['text'].capitalize()
    except AttributeError:
        pass
    fig.layout.update(yaxis_title=y_var.capitalize(),
                      legend=legend,
                      margin=theme['graph_margins'],
                      
                      )
    return fig


def generate_qq_plot(dataframe, vars_, dist='norm', **layout_kwargs):
    """ Plotting sample distribution quantiles against theoretical quantiles.
    
    :param dataframe:
    :type dataframe: pandas.DataFrame
    :param vars_: Which column's or columns' distribution to to plot against theoretical distribution.
    :type vars_: str|list
    :param dist: Which theoretical distribution to compare to.
    :return: Figure object of graph.
    :rtype: plotly.graph_objs.Figure
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title="Distribution",
    )
    if isinstance(vars_, str):
        vars_ = [vars_]
        
    try:
        fig = make_subplots(cols=dataframe['task'].nunique(), rows=1,
                            x_title="Theoretical Quantiles", y_title="Standardized Residuals",
                            shared_yaxes=True,
                            horizontal_spacing=0.04,
                            column_titles=task_order[dataframe['task'].unique()].sort_values().index.to_list())
    except (KeyError, ValueError) as e:
        fig = go.Figure()
    else:
        # Subplots for conditions.
        grouped = dataframe.groupby('task')
        col_order = pd.Series(np.arange(dataframe['task'].nunique())+1,
                              index=task_order[dataframe['task'].unique()].sort_values().index)
        show_legend = True
        for name, df in grouped:
            z_scores = pd.DataFrame(scipy.stats.zscore(df[vars_], axis=0, ddof=0), columns=vars_)
            for var_ in vars_:
                theoretical_qs = scipy.stats.probplot(z_scores[var_], dist=dist)
                x = np.array([theoretical_qs[0][0][0], theoretical_qs[0][0][-1]])
            
                fig.add_scatter(x=theoretical_qs[0][0], y=theoretical_qs[0][1], showlegend=show_legend,
                                mode='markers', name=var_, marker_color=theme[var_], opacity=0.7,
                                row=1, col=col_order[name])
                fig.add_scatter(x=x, y=theoretical_qs[1][1] + theoretical_qs[1][0] * x, mode='lines',
                                showlegend=False, marker_color=theme[var_], row=1, col=col_order[name])
            show_legend = False
        
    fig.layout.update(legend=legend,
                      hovermode=False,
                      margin={'l': 60, 'b': 60, 't': 40, 'r': 40},
                      **layout_kwargs
                      )
    return fig
