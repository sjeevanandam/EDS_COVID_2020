import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import dash_daq as daq
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
print(os.getcwd())
df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')
df_input_recov=pd.read_csv('data/processed/COVID_final_recov_set.csv',sep=';')
df_input_daily=pd.read_csv('data/processed/COVID_final_daily_set.csv',sep=';')
df_analyse = pd.read_csv('data/processed/COVID_full_flat_table.csv',sep=';')
df_SIR_data = pd.read_csv('data/processed/COVID_SIR_Model_Data.csv',sep=';')







app = dash.Dash(external_stylesheets=[dbc.themes.LUX, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'])
app.layout = html.Div([

    dcc.Markdown('''
    #  Applied Data Science on COVID-19 data

    Goal of the project is to teach data science by applying a cross industry standard process,
    it covers the full walkthrough of: automated data gathering, data transformations,
    filtering and machine learning to approximating the doubling time, and
    (static) deployment of responsive dashboard.

    ''',style={'text-align':'center','border-style': 'solid'}),


 
    html.Br(),html.Br(),html.Br(),



    dbc.Row([
        ### input + panel
        dbc.Col(md=5, children=[
            dcc.Markdown('''
            ## Select Country for Country level Stats
            '''),


            dcc.Dropdown(
                id='country_drop_down_stats',
                options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
                value='India', # which is pre-selected
                multi=False
            ),
            html.Br(),html.Br(),
            html.Div([
                dcc.Tabs(id='tabs-example', value='tab-1', children=[
                    dcc.Tab(   label='Cummulative', value='tab-1', children=[
                        html.Div([
                            html.Br(),html.Br(),
                            dbc.Row([
                                dbc.Col(md=3, children=[
                                dcc.Markdown('''
                                **Scale Modes:**
                                ''',style={"font-weight": "900","font-size": "17px"}),
                                ]),
                                dbc.Col(md=4, children=[
                                    dcc.RadioItems(
                                        options=[
                                            {'label': 'Uniform', 'value': 'linear'},
                                            {'label': 'Logarithmic', 'value': 'log'},
                                            
                                        ],
                                        value='linear',
                                        id='scale_type',
                                        labelStyle={'display': 'inline-block'}
                                    ),
                                ])
                            ])
                        
                        ]),
                        dcc.Graph( id='multi_graph'),
                        
                    ]),
                    dcc.Tab(   label='Daily', value='tab-2', children=[
                        
                        
                        dcc.Graph( id='multi_graph_daily'),
                        
                    
                    ]),
                ]),




            ]),
            html.Br(),html.Br(),html.Br(),
            #dcc.Graph( id='multi_graph'),

        ]),
        ### plots
        dbc.Col(md=5, children=[
            
            html.Div(id="output-panel", style={'marginLeft': 250, 'marginTop': 250}),
            
        ])
    ]),


    html.Br(),html.Br(),html.Br(),
    html.Div([
        dcc.Markdown('''
        ## Multi-Select Country for Doubling Rate Visualisation
        ''',style={'text-align':'center'}),

        html.Br(),html.Br(),
        dcc.Dropdown(
            id='country_drop_down',
            options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
            value=['US', 'Germany','India'], # which are pre-selected
            multi=True,
            style=dict(
                        width='60%',
                        verticalAlign="middle"
                    )
        ),

        # dcc.Markdown('''
        #     ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        #     '''),
        
        html.Br(),

        dcc.Dropdown(
            id='doubling_time',
            options=[
                
                {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
                {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
            ],
            value='confirmed_DR',
            multi=False,
            style=dict(
                        width='60%',
                        verticalAlign="middle"
                    )
        ),
        html.Br(),html.Br(),
        html.Div([
            dcc.Graph( id='main_window_slope'),
        ], style=dict(position="relative", width='60%',
                    marginLeft= 250)),

    html.Br(),html.Br(),html.Br(),
    html.Div([
        dcc.Markdown('''
        ## Multi-Select Country for SIR Visualisation
        ''',style={'text-align':'center'}),

        html.Br(),html.Br(),
        dcc.Dropdown(
            id='country_drop_down_sir',
            options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
            value=['US','India'], # which are pre-selected
            multi=True,
            style=dict(
                        width='60%',
                        verticalAlign="middle"
                    )
        ),

        # dcc.Markdown('''
        #     ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        #     '''),
        
        
        html.Br(),html.Br(),
        html.Div([
            dcc.Graph( id='sir_chart'),
        ], style=dict(position="relative", width='60%',
                    marginLeft= 250)),

    ],style={'border-style': 'solid', 'marginRight': 50, 'padding': '10px'})

    



    ], style={'marginLeft': 50, 'marginTop': 25})
])


@app.callback(
    Output('multi_graph', 'figure'),
    [Input('country_drop_down_stats', 'value'),
    Input('scale_type', 'value')]
)
def update_cummulative_stacked_plot(country,scale_type):
    


    traces = []
 
    df_plot=df_input_large[df_input_large['country'] == country]

        
    df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date','deaths']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)

    df_plot_recov = df_input_recov[df_input_recov['country'] == country]
    df_plot_recov=df_plot_recov[['state','country','recovered','date']].groupby(['country','date']).agg(np.sum).reset_index()


    fig=make_subplots(rows=4, cols=1,
                subplot_titles=("Total Confirmed", "Total Active ", "Total Recovered", 'Total Deaths'),
                shared_xaxes=False, 
                vertical_spacing=0.1,
    )
    fig.add_trace(go.Scatter(
                        x=df_plot.date,
                        y=df_plot['confirmed'],
                        mode='markers+lines',
                        opacity=0.9,
                        fill='tozeroy',
                 ), row=1,col=1
    )
    fig.add_trace(go.Scatter(
                        x=df_plot.date,
                        y=df_plot['confirmed']- (df_plot_recov['recovered']+df_plot['deaths']),
                        mode='markers+lines',
                        opacity=0.9,
                        fill='tozeroy',
                 ), row=2,col=1
    )
    fig.add_trace(go.Scatter(
                        x=df_plot_recov.date,
                        y=df_plot_recov['recovered'],
                        mode='markers+lines',
                        opacity=0.9,
                        fill='tozeroy',
                        
                 ), row=3,col=1
    )
    fig.add_trace(go.Scatter(
                        x=df_plot.date,
                        y=df_plot['deaths'],
                        mode='markers+lines',
                        opacity=0.9,
                        fill='tozeroy',
                 ), row=4,col=1
    )
    
    
    
    
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=1, col=1)
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=2, col=1)
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=3, col=1)
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=4, col=1)

    fig.update_yaxes(type=scale_type, row=1, col=1, title='Confirmed infected people')
    fig.update_yaxes(type=scale_type, row=2, col=1, title='Active infected people')
    fig.update_yaxes(type=scale_type, row=3, col=1, title='Recovered people')
    fig.update_yaxes(type=scale_type, row=3, col=1, title='Deaths')

    fig.update_layout(dict (

                width=900,
                height=1500,
                template="plotly_dark",
                showlegend=False



                
        ))
    return fig



@app.callback(
    Output('multi_graph_daily', 'figure'),
    [Input('country_drop_down_stats', 'value'),
    ]
)
def update_cummulative_stacked_plot(country):

    scale_type = 'linear'
    traces = []
 
    df_plot=df_input_daily[df_input_daily['country'] == country]

        
    #df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date','deaths']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)

    df_plot_recov = df_input_recov[df_input_recov['country'] == country]
    #df_plot_recov=df_plot_recov[['state','country','recovered','date']].groupby(['country','date']).agg(np.sum).reset_index()


    fig=make_subplots(rows=4, cols=1,
                subplot_titles=("Daily Confirmed", "Daily Active ", "Daily Recovered", "Daily Deaths"),
                shared_xaxes=False, 
                vertical_spacing=0.1,
    )
    fig.add_trace(go.Bar(
                        x=df_plot.date,
                        y=df_plot['daily_confirmed'],
                       
                        
                       
                 ), row=1,col=1
    )
    fig.add_trace(go.Bar(
                        x=df_plot.date,
                        y=df_plot['daily_confirmed'] - (df_plot['daily_recovered']+df_plot['daily_deaths']),
                       
                        
                        
                 ), row=2,col=1
    )
    fig.add_trace(go.Bar(
                        x=df_plot.date,
                        y=df_plot['daily_recovered'],
                       
                        
                        
                 ), row=3,col=1
    )
    fig.add_trace(go.Bar(
                        x=df_plot.date,
                        y=df_plot['daily_deaths'],
                        
                        
                        
                 ), row=4,col=1
    )
    
    

    fig.update_xaxes(type="date", row=1, col=1)
    fig.update_xaxes(type="date", row=2, col=1)
    fig.update_xaxes(type="date", row=3, col=1)
    fig.update_xaxes(type="date", row=4, col=1)

    fig.update_yaxes(type=scale_type, row=1, col=1, title='Confirmed infected people')
    fig.update_yaxes(type=scale_type, row=2, col=1, title='Active infected people')
    fig.update_yaxes(type=scale_type, row=3, col=1, title='Recovered people')
    fig.update_yaxes(type=scale_type, row=3, col=1, title='Deaths')
    
    
    fig.update_layout(dict (

               width=900,
                height=1500,
                template="plotly_dark",
                showlegend=False

        ))
    return fig



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')]
    )
def update_figure(country_list, show_doubling):
    
    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }


    traces = []
    fig1 = go.Figure()

    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)


        fig1.add_trace(go.Scatter(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each,
                                
                        ))
                )
    

    fig1.update_layout(dict (
                width=1280,
                height=720,
                template="plotly_dark",
                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis,
                
                
        ))
    return fig1



# Python function to render output panel
@app.callback(output=Output("output-panel","children"), inputs=[Input('country_drop_down_stats', 'value'),])
def render_output_panel(country):
    df_card_plot = df_input_large[df_input_large['country'] == country]
    df_card_plot_recov = df_input_recov[df_input_recov['country'] == country]

    df_card_plot=df_card_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date','deaths']].groupby(['country','date']).agg(np.sum).reset_index()
    df_card_plot_recov=df_card_plot_recov[['state','country','recovered','date']].groupby(['country','date']).agg(np.sum).reset_index()
    


    total_cases_until_today, active_cases_today, total_deaths, total_recovered = int(df_card_plot[-1:].confirmed), int(df_card_plot[-1:].confirmed) - (int(df_card_plot_recov[-1:].recovered) + int(df_card_plot[-1:].deaths)), \
                                                                                    int(df_card_plot[-1:].deaths), int(df_card_plot_recov[-1:].recovered)
    print(df_card_plot[-1:])
    print(df_card_plot[-2:-1])
    #increases over previous day
    total_cases_increase, active_cases_increase, total_deaths_increase, total_recovered_increase = \
        int(df_card_plot[-1:].confirmed) - int(df_card_plot[-2:-1].confirmed), \
        (int(df_card_plot[-1:].confirmed) - (int(df_card_plot_recov[-1:].recovered) + int(df_card_plot[-1:].deaths)))    -    (int(df_card_plot[-2:-1].confirmed) - (int(df_card_plot_recov[-2:-1].recovered) + int(df_card_plot[-2:-1].deaths))), \
        int(df_card_plot[-1:].deaths) -int(df_card_plot[-2:-1].deaths), int(df_card_plot_recov[-1:].recovered) -int(df_card_plot_recov[-2:-1].recovered)
    
    peak_color = "white"
    panel = html.Div([
        html.H4(country),
        dbc.Card(body=True, className="text-white bg-primary", children=[
            
            html.H6("Total cases until today:", style={"color":"white"}),
            html.Div([
            html.H3("{:,.0f}".format(total_cases_until_today), style={"color":"white", "float":"left","width":"35%"}),
            html.H4('[+' + "{:,.0f}".format(total_cases_increase) + ']', style={"color":"yellow", "float":"right","width":"65%"}),
            ]),


            
            html.H6("Active cases today:", style={"color":"white"}),
            html.Div([
            html.H3("{:,.0f}".format(active_cases_today), style={"color":"white", "float":"left","width":"35%"}),
            html.H4('[+' + "{:,.0f}".format(active_cases_increase) + ']', style={"color":"yellow", "float":"right","width":"65%"}),
            ]),
            
            html.H6("Recovered cases until today:", style={"color":"white"}),
            html.Div([
            html.H3("{:,.0f}".format(total_recovered), style={"color":"white", "float":"left","width":"35%"}),
            html.H4('[+' + "{:,.0f}".format(total_recovered_increase) + ']', style={"color":"yellow", "float":"right","width":"65%"}),
            ]),

            html.H6("Total deaths until today:", style={"color":"white"}),
            html.Div([
            html.H3("{:,.0f}".format(total_deaths), style={"color":"red", "float":"left","width":"35%"}),
            html.H4('[+' + "{:,.0f}".format(total_deaths_increase) + ']', style={"color":"red", "float":"right","width":"65%"}),
            ]),
        
        ])
    ])
    return panel



@app.callback(
    Output('sir_chart', 'figure'),
    [Input('country_drop_down_sir', 'value')])
def update_figure(country_list):
    traces = []
    fig =go.Figure()
    if(len(country_list) > 0):
        for each in country_list:
            country_data = df_analyse[each][35:]
            ydata = np.array(country_data)
            t = np.arange(len(ydata))
            fitted = np.array(df_SIR_data[each])
            #t, ydata, fitted = Handle_SIR_Modelling(ydata)
            fig.add_trace(go.Scatter(
                x = t,
                y = ydata,
                mode = 'markers+lines',
                name = each+str(' - Truth'),
                opacity = 0.9
            ))
            fig.add_trace(go.Scatter(
                x = t,
                y = fitted,
                mode = 'markers+lines',
                name = each+str(' - Simulation'),
                opacity = 0.9
            ))

    fig.update_layout(dict(
            width = 1280,
            height = 720,
            title = 'Fit of SIR model for: '+', '.join(country_list),
            xaxis = {
                'title': 'Days', #'Fit of SIR model for '+str(each)+' cases',
                'tickangle': -45,
                'nticks' : 20,
                'tickfont' : dict(size = 14, color = '#7F7F7F')
            },
            yaxis = {
                'title': 'Population Infected',
                'type': 'log'
            },
            template="plotly_dark",
        ))        
    return fig




if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=True)
