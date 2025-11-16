# --------------------------------------------------------------
#  CONFECTIONARY SALES DASHBOARD – FULLY INTERACTIVE (UPDATED)
# --------------------------------------------------------------
#  Fixed: Standardized 'Choclate Chunk' to 'Chocolate Chunk' in data
#  Updated: Changed Total Revenue bar chart to stacked (barmode='stack') with formatted labels and y-axis
#  Added: Revenue Trend Line Chart (multi-line by region, over years, filtered by confectionary)
#  Added: Numerical Column Dropdown + Combined Histogram (with KDE) & Boxplot Graph (filtered by country & confectionary)
#  Run with:   python this_file.py
# --------------------------------------------------------------

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# -------------------------- 1. Load & clean data --------------------------
data = pd.read_excel('Confectionary [4564] (2).xlsx')

# Fix: Standardize 'Choclate Chunk' to 'Chocolate Chunk'
data['Confectionary'] = data['Confectionary'].replace('Choclate Chunk', 'Chocolate Chunk')

# Simple imputation (you can replace with a more sophisticated method)
for col in ['Units Sold', 'Cost(£)', 'Profit(£)']:
    data[col].fillna(data[col].mean(), inplace=True)

# Calculate additional columns
data['Selling Price(£)'] = np.where(data['Units Sold'] != 0, data['Revenue(£)'] / data['Units Sold'], np.nan)
data['Profit(%)'] = np.where(data['Revenue(£)'] != 0, (data['Profit(£)'] / data['Revenue(£)']) * 100, np.nan)
data['Total Profit(£)'] = data['Profit(£)']

# Date helpers
data['Year']  = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['MonthName'] = data['Date'].dt.strftime('%b')
data['YearMonth'] = data['Date'].dt.to_period('M').astype(str)  # For monthly grouping

# Numerical columns for distribution plots
numerical_columns = [
    'Units Sold',
    'Cost(£)',
    'Selling Price(£)',
    'Profit(£)',
    'Total Profit(£)',
    'Revenue(£)',
    'Profit(%)'
]

# -------------------------- 2. Dash app --------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Confectionary Sales Dashboard"

# -------------------------- 3. Layout --------------------------
app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'margin': '20px'},
    children=[
        html.H1("Confectionary Sales Dashboard", style={'textAlign': 'center'}),

        html.Div([
            html.Label("Select Country:"),
            dcc.Dropdown(
                id='country-dd',
                options=[{'label': c, 'value': c} for c in data['Country(UK)'].unique()],
                value='England',
                clearable=False,
                style={'width': '48%', 'display': 'inline-block'}
            ),

            html.Label("Select Confectionary:", style={'marginLeft': '4%'}),
            dcc.Dropdown(
                id='conf-dd',
                options=[{'label': c, 'value': c} for c in data['Confectionary'].unique()],
                value='Biscuit',
                clearable=False,
                style={'width': '48%', 'display': 'inline-block'}
            ),
        ], style={'marginBottom': '30px'}),

        html.Div([
            html.Label("Select Numerical Column for Distribution:"),
            dcc.Dropdown(
                id='num-col-dd',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value='Units Sold',
                clearable=False,
                style={'width': '50%'}
            ),
        ], style={'marginBottom': '20px'}),

        # ---- Row 1 -------------------------------------------------
        html.Div([
            html.Div(dcc.Graph(id='scatter-rev-units'),   style={'width': '48%'}),
            html.Div(dcc.Graph(id='box-profit'),        style={'width': '48%'})
        ], style={'display': 'flex', 'gap': '4%'}),

        # ---- Row 2 -------------------------------------------------
        html.Div([
            html.Div(dcc.Graph(id='heatmap-units'),     style={'width': '48%'}),
            html.Div(dcc.Graph(id='hbar-cost'),         style={'width': '48%'})
        ], style={'display': 'flex', 'gap': '4%', 'marginTop': '30px'}),

        # ---- Row 3 -------------------------------------------------
        html.Div([
            html.Div(dcc.Graph(id='vbar-monthly-rev'),  style={'width': '48%'}),
            html.Div(dcc.Graph(id='line-rev-time'),     style={'width': '48%'})
        ], style={'display': 'flex', 'gap': '4%', 'marginTop': '30px'}),

        # ---- Row 4 (original extra charts) -------------------------
        html.Div([
            html.Div(dcc.Graph(id='pie-profit'),        style={'width': '48%'}),
            html.Div(dcc.Graph(id='corr-heatmap'),      style={'width': '48%'})
        ], style={'display': 'flex', 'gap': '4%', 'marginTop': '30px'}),

        # ---- Row 5: Monthly Sales per Region Bar Chart -------------
        html.Div([
            html.Div(dcc.Graph(id='monthly-sales-bar'), style={'width': '100%'})
        ], style={'marginTop': '30px'}),

        # ---- Row 6: Total Revenue by Region and Confectionary ------
        html.Div([
            html.Div(dcc.Graph(id='bar-total-rev'), style={'width': '100%'})
        ], style={'marginTop': '30px'}),

        # ---- Row 7: Average Profit Margin % Bar Chart --------------
        html.Div([
            html.Div(dcc.Graph(id='profit-margin-bar'), style={'width': '100%'})
        ], style={'marginTop': '30px'}),

        # ---- Row 8: Revenue Trend Line Chart -----------------------
        html.Div([
            html.Div(dcc.Graph(id='revenue-trend-line'), style={'width': '100%'})
        ], style={'marginTop': '30px'}),

        # ---- New Row 9: Distribution & Box Plot --------------------
        html.Div([
            html.Div(dcc.Graph(id='dist-box-graph'), style={'width': '100%'})
        ], style={'marginTop': '30px'})
    ]
)

# -------------------------- 4. Callbacks --------------------------

# 1. Scatter: Revenue vs Units Sold
@app.callback(
    Output('scatter-rev-units', 'figure'),
    [Input('country-dd', 'value'), Input('conf-dd', 'value')]
)
def update_scatter(country, conf):
    df = data[(data['Country(UK)'] == country) & (data['Confectionary'] == conf)]
    fig = px.scatter(df, x='Units Sold', y='Revenue(£)',
                     color='Confectionary',
                     size='Profit(£)', hover_data=['Date'],
                     title=f'Revenue vs Units Sold – {conf} in {country}')
    return fig


# 2. Box plot: Profit per Confectionary
@app.callback(
    Output('box-profit', 'figure'),
    Input('country-dd', 'value')
)
def update_box(country):
    df = data[data['Country(UK)'] == country]
    fig = px.box(df, x='Confectionary', y='Profit(£)',
                 title=f'Profit Distribution per Confectionary – {country}')
    return fig


# 3. Heat-map (pivot): Avg Units Sold by Year × Country
@app.callback(
    Output('heatmap-units', 'figure'),
    Input('conf-dd', 'value')
)
def update_heatmap(conf):
    df = data[data['Confectionary'] == conf]
    pivot = df.pivot_table(values='Units Sold', index='Year',
                           columns='Country(UK)', aggfunc='mean').fillna(0)
    fig = px.imshow(pivot,
                    labels=dict(x='Country', y='Year', color='Avg Units Sold'),
                    title=f'Avg Units Sold ({conf}) – Year × Country')
    fig.update_layout(height=450)
    return fig


# 4. Horizontal bar: Total Cost per Confectionary (overall)
@app.callback(
    Output('hbar-cost', 'figure'),
    Input('country-dd', 'value')
)
def update_hbar(country):
    df = data[data['Country(UK)'] == country]
    agg = df.groupby('Confectionary')['Cost(£)'].sum().reset_index()
    fig = px.bar(agg, y='Confectionary', x='Cost(£)', orientation='h',
                 title=f'Total Cost per Confectionary – {country}')
    return fig


# 5. Vertical bar: Monthly Revenue (selected country, selected confectionary)
@app.callback(
    Output('vbar-monthly-rev', 'figure'),
    [Input('country-dd', 'value'), Input('conf-dd', 'value')]
)
def update_vbar_monthly(country, conf):
    df = data[(data['Country(UK)'] == country) & (data['Confectionary'] == conf)]
    monthly = df.groupby(['Year', 'Month', 'MonthName'])['Revenue(£)'].sum().reset_index()
    monthly['Period'] = monthly['MonthName'] + ' ' + monthly['Year'].astype(str)
    monthly = monthly.sort_values(['Year', 'Month'])
    fig = px.bar(monthly, x='Period', y='Revenue(£)',
                 title=f'Monthly Revenue – {conf} in {country}')
    fig.update_xaxes(tickangle=45)
    return fig


# 6. Line chart: Revenue over time (original)
@app.callback(
    Output('line-rev-time', 'figure'),
    [Input('country-dd', 'value'), Input('conf-dd', 'value')]
)
def update_line_rev(country, conf):
    df = data[(data['Country(UK)'] == country) & (data['Confectionary'] == conf)].sort_values('Date')
    fig = px.line(df, x='Date', y='Revenue(£)',
                  title=f'Revenue Over Time – {conf} in {country}')
    return fig


# 7. Pie chart: Profit share by Country (selected confectionary)
@app.callback(
    Output('pie-profit', 'figure'),
    Input('conf-dd', 'value')
)
def update_pie(conf):
    df = data[data['Confectionary'] == conf]
    agg = df.groupby('Country(UK)')['Profit(£)'].sum().reset_index()
    fig = px.pie(agg, values='Profit(£)', names='Country(UK)',
                 title=f'Profit Share by Country – {conf}')
    return fig


# 8. Correlation heatmap
@app.callback(
    Output('corr-heatmap', 'figure'),
    [Input('country-dd', 'value'), Input('conf-dd', 'value')]
)
def update_corr(country, conf):
    df = data[(data['Country(UK)'] == country) & (data['Confectionary'] == conf)]
    corr = df[['Units Sold', 'Cost(£)', 'Profit(£)', 'Revenue(£)']].corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto',
                    title=f'Correlation – {conf} in {country}')
    return fig

# 9. Monthly Units Sold per Region (grouped bar chart)
@app.callback(
    Output('monthly-sales-bar', 'figure'),
    Input('conf-dd', 'value')
)
def update_monthly_sales_bar(conf):
    df = data[data['Confectionary'] == conf]
    monthly = df.groupby(['YearMonth', 'Country(UK)'])['Units Sold'].sum().reset_index()
    monthly = monthly.sort_values('YearMonth')
    fig = px.bar(monthly, x='YearMonth', y='Units Sold', color='Country(UK)',
                 barmode='group',
                 title=f'Monthly Units Sold per Region – {conf}')
    fig.update_xaxes(tickangle=45, title='Date (Year-Month)')
    fig.update_yaxes(title='Units Sold')
    return fig

# 10. Total Revenue by Region and Confectionary (stacked bar chart)
@app.callback(
    Output('bar-total-rev', 'figure'),
    Input('conf-dd', 'value')  # Optional: Filter by selected confectionary; remove if you want overall
)
def update_total_rev_bar(conf):
    if conf:
        df = data[data['Confectionary'] == conf]
    else:
        df = data.copy()
    agg = df.groupby(['Country(UK)', 'Confectionary'])['Revenue(£)'].sum().reset_index()
    fig = px.bar(agg, x='Country(UK)', y='Revenue(£)', color='Confectionary',
                 barmode='stack', text_auto='.2s',
                 title=f'Total Revenue by Region and Confectionary (£){" (" + conf + ")" if conf else ""}')
    fig.update_xaxes(title='Region')
    fig.update_yaxes(title='Revenue (£)', tickformat='.2s')
    fig.update_layout(height=500)
    return fig

# 11. Average Profit Margin % by Region and Confectionary (grouped bar chart)
@app.callback(
    Output('profit-margin-bar', 'figure'),
    Input('conf-dd', 'value')  # Ignore input to show overall; or filter if desired
)
def update_profit_margin_bar(conf):
    df = data.copy()  # Show overall; comment and uncomment below to filter by conf
    # df = data[data['Confectionary'] == conf]  # If you want to filter
    df['ProfitMargin'] = (df['Profit(£)'] / df['Revenue(£)']) * 100
    df = df.replace([np.inf, -np.inf], np.nan)  # Handle division by zero or invalid
    agg = df.groupby(['Country(UK)', 'Confectionary'])['ProfitMargin'].mean().reset_index()
    fig = px.bar(agg, x='Country(UK)', y='ProfitMargin', color='Confectionary',
                 barmode='group',
                 title='Average Profit Margin % by Region and Confectionary')
    fig.update_xaxes(title='Region')
    fig.update_yaxes(title='Average Profit Margin (%)')
    fig.update_layout(height=500)
    return fig

# 12. Revenue Trend Over Years by Region (line chart)
@app.callback(
    Output('revenue-trend-line', 'figure'),
    Input('conf-dd', 'value')
)
def update_revenue_trend(conf):
    df = data[data['Confectionary'] == conf]
    agg = df.groupby(['Year', 'Country(UK)'])['Revenue(£)'].sum().reset_index()
    fig = px.line(agg, x='Year', y='Revenue(£)', color='Country(UK)',
                  title=f'Revenue Trend Over Years by Region – {conf}',
                  markers=True)
    fig.update_yaxes(tickformat='.2s')
    fig.update_layout(height=500)
    return fig

# 13. New: Distribution Histogram (with KDE) & Box Plot
@app.callback(
    Output('dist-box-graph', 'figure'),
    [Input('country-dd', 'value'), Input('conf-dd', 'value'), Input('num-col-dd', 'value')]
)
def update_dist_box(country, conf, col):
    filtered_data = data[(data['Country(UK)'] == country) & (data['Confectionary'] == conf)]
    df_col = filtered_data[col].dropna()
    if df_col.empty:
        return go.Figure()
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=(f'Distribution of {col}', f'Box Plot of {col}'))

    # Histogram
    hist = px.histogram(filtered_data, x=col, nbins=30)
    fig.add_trace(hist.data[0], row=1, col=1)

    # KDE (if enough data)
    if len(df_col) > 1:
        kde = gaussian_kde(df_col)
        x_range = np.linspace(df_col.min(), df_col.max(), 200)
        y_kde = kde(x_range)
        # Scale KDE to match histogram scale (approximate)
        bin_width = (df_col.max() - df_col.min()) / 30 if len(df_col) > 0 else 1
        y_kde *= len(df_col) * bin_width
        fig.add_trace(go.Scatter(x=x_range, y=y_kde, mode='lines', name='KDE', line=dict(color='red')), row=1, col=1)

    # Box plot
    box = px.box(filtered_data, y=col)
    fig.add_trace(box.data[0], row=1, col=2)

    fig.update_layout(title=f'Distribution and Box Plot for {col} in {country} - {conf}', height=500)
    return fig


# -------------------------- 5. Run server --------------------------
if __name__ == '__main__':
    app.run(debug=True)