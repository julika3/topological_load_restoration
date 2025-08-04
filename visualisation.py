import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from numpy import arange

from constants import *
from helper_functions import save_figure, save_figures_to_file


def versorgungsgrad(load_sum, generation_series, scenario):
    gen_sum = generation_series.sum()
    vg = gen_sum / load_sum
    generation_series.sort_values(ascending=False, inplace=True)

    fig = make_subplots(rows=1,
                        cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=("", ""))

    if vg < 1:
        fig.add_trace(go.Pie(labels=['', ''],
                             values=[gen_sum, load_sum-gen_sum], hole=.55, textinfo='none',
                             marker_colors=['rgb(113,209,145)', 'rgb(240,240,240)'],
                             sort=False,
                             showlegend=False, hoverinfo="none"
                             ), row=1, col=1)
    else:
        fig.add_trace(go.Pie(labels=['', ''],
                             values=[vg-1, 2-vg], hole=.55, textinfo='none',
                             marker_colors=['rgb(80, 133, 97)', 'rgb(113,209,145)'],
                             sort=False,
                             showlegend=False,hoverinfo="none"
                             ), row=1, col=1)
    fig.add_annotation(x=0.12, y=0.5,
                       text=f'Versorgungsgrad:<br>{round(vg * 100, 2)} %',
                       showarrow=False,
                       font_size=38)

    labels = [ps for ps in generation_series.index if ps != 'Jaenschwalde'] + ['Jaenschwalde']
    values = [gen for gen in generation_series.values]
    values_jw = values.pop(0)
    values = values + [values_jw]

    fig.add_trace(go.Pie(labels=labels,
                         values=values, hole=.55,
                         marker_colors=[ASSIGNED_COLORS.get(ps, 'black') for ps in labels],
                         sort=False, direction ='clockwise',
                         text=generation_series.map(lambda x: f'{int(round(x, 0))} MW'),
                         textfont_size=30, textinfo='label',#textfont_color='White',
                         showlegend=False, textposition='outside', hoverinfo="none"
                         ), row=1, col=2)
    fig.add_trace(go.Pie(labels=labels,
                         values=values, hole=.55,
                         marker_colors=[ASSIGNED_COLORS.get(ps, 'black') for ps in labels],
                         sort=False, direction ='clockwise',
                         text=[f'{int(round(x, 0))} MW' for x in values],
                         textfont_size=30, textinfo='text', textfont_color='White',
                         showlegend=False, textposition='inside', hoverinfo="none"
                         ), row=1, col=2)
    fig.add_annotation(x=0.89, y=0.5,
                       text=f'Gesamterzeugung:<br>{round(gen_sum/1000, 2)} GW',
                       showarrow=False,
                       font_size=38)

    fig.update_layout(margin_t=100, legend_bordercolor='Black', legend_borderwidth=2,
                      title=dict(text=f'Versorgungssituation: {scenario.replace("_", " ")}', font=dict(size=40)),
                      title_x=0.5, font_size=22)
    fig.update_layout(height=900, width=1800)

    fig.show()
    return fig


def versorgungsgrad_vergleich(scenario_vg_dict):
    fig = make_subplots(rows=2,
                        cols=4,
                        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
                               [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=[scenario.split('_')[1] for scenario in scenario_vg_dict.keys()])
    fig.update_annotations(font_size=28)

    fig.update_layout(margin_t=150, legend_bordercolor='Black', legend_borderwidth=2,
                      width=1200, height=800,
                      title=f'Versorgungsgrade nach Abschaltung', title_x=0.5, font=dict(size=25))

    yaxis_row1 = 0.845
    yaxis_row2 = 0.155
    annotation_placer = {0: (0.055, yaxis_row1),
                         1: (0.315, yaxis_row1),
                         2: (0.633, yaxis_row1),
                         3: (0.945, yaxis_row1),
                         4: (0.055, yaxis_row2),
                         5: (0.315, yaxis_row2),
                         6: (0.633, yaxis_row2),
                         7: (0.945, yaxis_row2)
                         }

    for n, vg in enumerate(scenario_vg_dict.values()):
        if vg < 1:
            fig.add_trace(go.Pie(labels=['', ''],
                                 values=[vg, 1-vg], hole=.6, textinfo='none',
                                 marker_colors=['rgb(113,209,145)', 'rgb(240,240,240)'],
                                 sort=False,
                                 showlegend=False
                                 ), row=1 if n < 4 else 2, col=n+1 if n<4 else n-3)
        else:
            fig.add_trace(go.Pie(labels=['', ''],
                                 values=[vg-1, 2-vg], hole=.6, textinfo='none',
                                 marker_colors=['rgb(80, 133, 97)', 'rgb(113,209,145)'],
                                 sort=False,
                                 showlegend=False
                                 ), row=1 if n < 4 else 2, col=n+1 if n<4 else n-3)
        fig.add_annotation(x=annotation_placer[n][0], y=annotation_placer[n][1],
                           text=f'{round(vg * 100, 2)} %',
                           showarrow=False,
                           font_size=25)

    fig.show()
    return fig


def shortest_distance_matrix_heatmap(sdm_df, ranked_lengths=False, show_values=False, show_colorscale=True,
                                     show=SHOW_FIGS, save_as=False):
    """
    takes the shortest distance matrix between multiple power stations and a networks substations and turns the
    corresponding shortest path lengths or the closeness rank of the power stations into a heatmap
    :param show_values: print the corresponding matrix entry values on the heatmap
    :param show_colorscale: either hides or shows the colorscale legend
    :param ranked_lengths: use the power station ranks (from closest to farthest) instead of the path lengths
    :param sdm_df: dataframe with substation columns and power station rows detailing the shortest path
                length between them
    :param show: open figure in browser
    :param save_as: "html"/"PDF"/"JSON" save figure to export folder 'fig'
    :return: go.Figure SDM visualised as Heatmap
    """
    zmax = len(sdm_df.index) if ranked_lengths else int(sdm_df.max().max())
    sdm_df = sdm_df.sort_index(ascending=False).reindex(sorted(sdm_df.columns), axis=1)
    if show_values:
        fig = go.Figure(data=go.Heatmap(x=sdm_df.columns, y=sdm_df.index,
                                        z=sdm_df.values.tolist(), zmin=1, zmax=zmax,
                                        text=sdm_df.values.tolist(),
                                        texttemplate="%{text}", textfont={"size":13},
                                        colorscale='YLGnBu_r'))
    else:
        fig = go.Figure(data=go.Heatmap(x=sdm_df.columns, y=sdm_df.index,
                                        z=sdm_df.values.tolist(), zmin=1, zmax=zmax,
                                        colorscale='YLGnBu_r'))
    title = f'Rank des Kraftwerks als nächster Versorger des Lastknotens in {NETWORK_MODEL_NAME}' if ranked_lengths \
        else f'Pfadlängen der Geodäten zwischen Lastknoten und Kraftwerken in {NETWORK_MODEL_NAME}'
    fig.update_layout(title=title, title_x=0.5, margin_t=60,
                      xaxis_title='<b>Lastknoten</b>',
                      height=600, width=1200, font_size=14,
                      # ensures that all labels are shown
                      xaxis_dtick=1, yaxis_dtick=1)
    fig.update_traces(showscale=show_colorscale)
    fig.add_annotation(x=-0.135, y=0.5, xref="paper",
                   yref="paper", showarrow=False, yanchor='middle',
                   text='<b>Kraftwerk</b>',
                   font=dict(size=15), textangle=90)

    if show:
        fig.show()
    fig_title = f'shortest_distances_matrix_SDM'
    save_figure(fig, fig_title, save_as)

    return fig


def visualise_centrality(comp_df, mode='Betweenness', show=SHOW_FIGS, save_as=False):
    fig = go.Figure()
    if mode == 'Betweenness':
        comp_df = comp_df[comp_df[f'{mode} Centrality'] > 0]
    if mode == 'Closeness':
        comp_df = comp_df[comp_df[f'Power Transfer<br>{mode} Centrality'] > 0]
    bc_fig = comp_df.sort_values([f'{mode} Centrality'])
    colors = ['#6667AB', '#B28330']
    tickvals = [x for x in bc_fig.index]
    for col in [f'{mode} Centrality', f'Power Transfer<br>{mode} Centrality']:
        fig.add_trace(go.Scatter(x=tickvals, y=bc_fig[col].values.tolist(), name=col, marker_color=colors.pop(0)))

    fig.update_layout(xaxis_dtick=1, height=800, width=1200, font_size=18,
                      title=dict(text=f'{mode} Centrality der Umspannwerke im '#vermaschten '
                                      f'{NETWORK_MODEL_NAME} Netz', font=dict(size=25)),
                      title_x=0.5,
                      xaxis_title='<b>Umspannwerk</b>',
                      yaxis_tickvals=[x for x in arange(0, bc_fig.max(numeric_only=True).max() + 0.05, 0.05)],
                      legend_bordercolor='Black', legend_borderwidth=2,
                      legend_groupclick='toggleitem', legend_orientation='h')
    # workaround to set the required colors for tick labels
    tick_labels = []
    for tick in tickvals:
        if tick in ASSIGNED_COLORS.keys():
            label = f'<span style="color:#E64D67">{tick}</span>'
        else:
            label = str(tick)
        tick_labels.append(label)
    fig.update_xaxes(tickvals=tickvals,
                     ticktext=tick_labels,
                     tickfont_color='black')
    fig.update_layout(legend=dict(
        yanchor="top", y=0.96,
        xanchor="left", x=0.02))
    if show:
        fig.show()
    if save_as:
        save_figure(fig, f'{mode}_Centrality_UWs_vermaschtes_{NETWORK_MODEL_NAME}_Netz', save_as)
    return fig


def plot_restored_generation(dispatched_power_df, strategy='Superposition', last_data=None, load_names=LOAD_CASE,
                             relative_values=False, show=SHOW_FIGS, save_as=False):

    fig = realised_dispatch_bar_chart(dispatched_power_df, last_data, load_names, strategy, relative_values)

    if show:
        fig.show()
    if show or save_as:
        file_title = f'Deckung_nach_Versorgungswiederaufbau_{strategy}_{NETWORK_MODEL_NAME}'
        save_figure(fig, file_title, save_as)

    return fig


def realised_dispatch_bar_chart(dp_df, load_series, load_name, strategy, relative_values, show_sum=False):
    """
    Creates a stacked bar chart showing how distributed generation covers substation loads.

    Parameters:
    :param dp_df: (pd.DataFrame): Generation data per power station and load node.
    :param load_series: (pd.Series or None): Load data per substation. If None, only generation is shown.
    :param load_name: (str): Name of the load profile used (for labeling).
    :param strategy: (str): Strategy name used for generation dispatch (for labeling).
    :param relative_values: (bool or float): If True or float, normalize generation by load to show coverage ratio.
    :param show_sum: (bool): If True, adds a line showing the total generation per substation.
    :return: (plotly.graph_objects.Figure): The resulting bar chart figure.
    """
    fig = go.Figure()

    # Add load line if available and not using relative values
    if (load_series is not None) and (not relative_values):
        # Align substations between generation and load data
        last_reg = dp_df.align(load_series, join='inner', axis=1)[1]
        fig.add_trace(go.Scatter(x=last_reg.index, y=last_reg.values.tolist(), name=load_name,
                                 marker_color='darkred'))

    if relative_values:
        # Set y-axis range for relative values
        fig.update_layout(yaxis_range=[0, 1], yaxis_dtick=0.1)

        if load_series is None:
            # Normalize by a predefined threshold if no load data is given
            dp_df = dp_df.div(LOAD_ESTIMATE)
        else:
            # Align and normalize generation by load
            dp_df, load_series = dp_df.align(load_series[load_series > 0], join='inner', axis=1)
            last_dict = load_series.to_dict()
            dp_df = relative_values * dp_df.div([last_dict.get(ss, 1) for ss in dp_df.columns])

            # Add horizontal line for achieved coverage level if relative_values is a float
            if relative_values != 1:
                fig.add_hline(y=relative_values,
                              annotation_text=f"Versorgungsgrad<br>{relative_values:.4f}",
                              annotation_position="right")

    elif show_sum:
        # Add total generation per substation
        fig.add_trace(go.Scatter(x=dp_df.columns,
                                 y=dp_df.sum(axis=0).values.tolist(),
                                 name='Deckungssumme', marker_color='black'))

    # Add stacked bars for each power station
    for ps in reversed(dp_df.index):
        fig.add_trace(go.Bar(x=dp_df.columns, y=dp_df.loc[ps].values.tolist(), name=ps,
                             marker_color=ASSIGNED_COLORS.get(ps, '#2F2D30')))

    # Update layout labels and legend
    fig.update_layout(barmode='stack', xaxis_dtick=1, height=600, width=1200,
                      margin_t=75, title_x=0.45,
                      title=dict(text=f'Deckung der Lasten durch Kraftwerkserzeugung in {NETWORK_MODEL_NAME} <br>'
                            f'VWA Strategie: {strategy}, '
                            f'Angenommene Last: {LOAD_NAMES[load_name] if load_series is not None else LOAD_ESTIMATE}',
                                 font=dict(size=22)),
                      yaxis_title='Anteil der gedeckten Last' if relative_values else 'Gedeckte Last',
                      xaxis_title='Umspannwerk', font_size=16,
                      legend=dict(title='Kraftwerk:', yanchor="top", y=0.8,
    ))

    return fig


def visualise_electric_degree_centrality(edc_df, show=SHOW_FIGS, save_as=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    edc_fig = edc_df.sort_values(['DC', 'EDC'])
    fig.add_trace(go.Scatter(x=edc_fig.index, y=edc_fig['DC'].values.tolist(),
                             name='Degree Centrality', marker_color='#6667AB'))

    fig.add_trace(go.Scatter(x=edc_fig.index, y=edc_fig['EDC'].values.tolist(),
                             name='Electrical Degree Centrality', marker_color='#C27D59'), secondary_y=True)

    fig.update_layout(height=800, width=1400, barmode='relative', font_size=18,
                      title=dict(text=f'Electrical Degree Centrality',
                                 font=dict(size=25)),
                      title_x=0.5,
                      yaxis_side='right', yaxis2_side='left',
                      yaxis1_title='Degree Centrality', yaxis2_title='Electrical Degree Centrality',
                      xaxis_title='<b>Umspannwerk</b>', xaxis_dtick=1,
                      legend_bordercolor='Black', legend_borderwidth=2,
                      legend_groupclick='toggleitem', legend_orientation='h')
    fig.update_layout(legend=dict(
        yanchor="top", y=0.96,
        xanchor="left", x=0.02
    ))
    if show:
        fig.show()
    if save_as:
        save_figure(fig, f'Degree_Centrality_UWs_vermaschtes_{NETWORK_MODEL_NAME}_Netz', save_as)
    return fig


def visualise_path_deviation(scenario_case, base_case=None, optimum=None, strategy='', show=SHOW_FIGS, save_as=False):
    result1, paths1, scenario1 = scenario_case
    distances1 = paths1[result1 > 1]

    if optimum is None:
        result2, paths2, scenario2 = base_case
        distances2 = paths2[result2 > 1]
    else:
        distances2 = optimum
        scenario2 = 'Theoretisches Optimum'

    changed_path_lengths = distances2.sub(distances1, axis='index', fill_value=0)
    distances2 = distances2.sub(distances1.mask(distances1 != 0, 0), axis='index', fill_value=0)
    distances1 = distances1.sub(distances2.mask(distances1 != 0, 0), axis='index', fill_value=0)
    changed = changed_path_lengths.sum()[changed_path_lengths.sum() != 0].index.tolist()
    nodes_of_interest = changed_path_lengths[changed]
    sorter = pd.DataFrame({'max': nodes_of_interest.max(), 'sum': nodes_of_interest.sum()}).sort_values(
        ['sum', 'max'], ascending=False)
    nodes_of_interest = nodes_of_interest[sorter.index.tolist()]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    iterator = sorted(changed_path_lengths.index.tolist())
    if 'Nächstmögliches Kraftwerk' in iterator:
        iterator.remove('Nächstmögliches Kraftwerk')
        iterator.append('Nächstmögliches Kraftwerk')
    for n, ps in enumerate(iterator):
        fig.add_trace(go.Bar(x=sorter.index.tolist(), name=ps,
                             y=distances2[sorter.index.tolist()].loc[ps].values.tolist(),
                             marker_color=ASSIGNED_COLORS.get(ps, '#2F2D30'),
                             showlegend=ps=='Nächstmögliches Kraftwerk',
                             legendgroup=2),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(x=sorter.index.tolist(),  name=ps,
                             y=(-distances1)[sorter.index.tolist()].loc[ps].values.tolist(),
                             marker_color=ASSIGNED_COLORS.get(ps, '#2F2D30'),
                             showlegend=ps!='Nächstmögliches Kraftwerk',
                             legendgroup=n<4),
                      row=1, col=1, secondary_y=True)

    fig.add_hline(y=0, line_color='black', line_width=2)

    path_difference = nodes_of_interest.sum()
    fig.add_trace(go.Scatter(x=path_difference[path_difference > 0].index,
                             y=path_difference[path_difference > 0].values,
                             marker_color='#F7B718',
                             legendgroup=2,
                             name=f'Zusätzliche Distanz für<br>{scenario2.replace("_", " ")}'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=path_difference[path_difference < 0].index,
                             y=path_difference[path_difference < 0].values,
                             marker_color='#fedd00',
                             legendgroup=2,
                             name=f'Zusätzliche Distanz für<br>{scenario1.replace("_", " ")}'),
                  secondary_y=True)

    title_name_dict = {'superposition': 'Superposition', 'resilienzindikatoren': 'Resilienzindikatoren',
                       'optimierung': 'Optimierung'}

    yaxis_bottom_range = int(-distances1[changed].sum().max()) - 1
    yaxis_top_range = max(int(distances2[changed].sum().max()) + 1, 7) if optimum is None \
        else int(distances2[changed].sum().max()) + 1
    yaxis2_startval = yaxis_bottom_range + 2 if yaxis_bottom_range % 2 == 0 else yaxis_bottom_range + 1
    yaxis_endval = yaxis_top_range - 1 if yaxis_top_range % 2 == 0 else yaxis_top_range
    fig.update_layout(barmode='relative', height=600, width=1200, margin_t=80,
                      xaxis_dtick=1, font_size=14,
                      title=dict(text=f'Pfadlängen zu den versorgenden Kraftwerken Region 50Hertz <br>'
                            f'VWA Strategie: {title_name_dict.get(strategy, strategy)}', font=dict(size=22)),
                      title_x=0.5,
                      xaxis_title='<b>Umspannwerk</b>',
                      yaxis_range=[yaxis_bottom_range, yaxis_top_range],
                      yaxis_tickvals=[x for x in range(0, yaxis_endval, 2)],
                      yaxis2_range=[yaxis_bottom_range, yaxis_top_range],
                      yaxis2_tickvals=[x for x in range(yaxis2_startval, 2, 2)],
                      yaxis2_ticktext = [-x for x in range(yaxis2_startval, 2, 2)],
                      legend_bordercolor='Black', legend_borderwidth=2,
                      legend_groupclick='toggleitem', legend_orientation='h')
    if optimum is not None:
        fig.update_layout(legend=dict(
            yanchor="bottom", y=1-0.99,
            xanchor="left", x=0.01))
    else:
        fig.update_layout(legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.93))

    fig.add_annotation(x=-0.04, y=yaxis_top_range, xref="paper",
                       yref="y", showarrow=False, yanchor='top',
                       text=scenario2.replace("_", " "),
                       font=dict(size=18), textangle=-90)
    fig.add_annotation(x=0.99, y=yaxis_bottom_range, xref="paper",
                       yref="y", showarrow=False, yanchor='bottom',
                       text=scenario1.replace("_", " "),
                       font=dict(size=18), textangle=-90)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    if show:
        fig.show()
    if save_as:
        save_figure(fig, f'vwa_{scenario1}_{scenario2}_{strategy}_{NETWORK_MODEL_NAME}_Netz', save_as)
    return fig


if __name__ == "__main__":
    import networkx as nx
    from RestorationNetwork import *
    from network_evaluation import electrical_degree_centrality
    my_restoration_network = RestorationNetwork('50Hertz')
    my_restoration_network.set_load_case(LOAD_CASE)
    load_sum = my_restoration_network.load_series.sum()

    generation_series = my_restoration_network.generation_series
    vg_fig = versorgungsgrad(load_sum, generation_series, my_restoration_network.scenario_applied)
    vg_fig.show()
    save_figures_to_file([vg_fig], f'figs/versorgungsgrad_basisszenario.json')

    compare_shutdowns = False
    if compare_shutdowns:
        my_dict = {}
        for ss in my_restoration_network.generators:
            shutdown_scenario = {SCENARIO_NAME: f'Abschaltung_{ss}',
                                 SCENARIO_LOCATION: ss,
                                 SCENARIO_GENERATION_VALUE: 0,
                                 SCENARIO_DESCRIPTION: f'Kohleausstieg {ss}'}
            my_restoration_network.apply_scenario(shutdown_scenario, overwrite_prev=True)
            my_dict.update({my_restoration_network.scenario_applied: my_restoration_network.load_factor})

        versorgungsgrad_vergleich(my_dict)

