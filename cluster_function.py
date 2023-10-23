import plotly.express as px
import pandas as pd
import pandas as pd
from sklearn.metrics import silhouette_score
import copy
from sklearn.cluster import AgglomerativeClustering



def Scaler(df,method):
    st = method
    scaler = st.fit_transform(df)
    col = df.columns
    df_scaler = pd.DataFrame(data = scaler, columns = col)
    return df_scaler

def cluster_analysis_score(cluster_method,df,n):
    clus = cluster_method(n_clusters = n)
    clus.fit(df)
    labels = clus.labels_ 
    silhouette_avg = silhouette_score(df,labels)
    return(silhouette_avg,labels)

def ajustar_data(df, variables = None):
    if 'Ciudades' in df.columns:
        df = df.drop("Ciudades", axis = 1)
    df_melt = df.melt(id_vars = variables)
    return df_melt

def group_data(df,cluster):
    im3_groups_mean = df.groupby(['variable',cluster,
    ])['value'].mean().reset_index() 
    
    return im3_groups_mean

def interactive_scatter(datos,cluster):
    
    # Custom color palette
    custom_palette = ["red", "green", "blue", "black", "gray"]

    # Convert the column to string to make it categorical
    datos[cluster] = datos[cluster].astype(str)

    # Create the scatter plot using Plotly Express
    scatter_fig = px.scatter(data_frame=datos, x="variable", y="value", color=cluster, 
                         color_discrete_map={value: color for value, color in zip(datos[cluster].unique(), custom_palette)},
                         labels={'variable': ' ', 'value': 'Factor value'}, title='Customized Plot')

    # Create traces for lines connecting the points
    lines_fig = px.line(data_frame=datos, x="variable", y="value", color=cluster, 
                    color_discrete_sequence=custom_palette, line_shape='linear')

    # Update the scatter plot to show markers and lines
    scatter_fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    scatter_fig.add_traces(lines_fig.data)  # Add lines to the scatter plot

    # Customize the appearance of the combined plot
    scatter_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig.update_layout(plot_bgcolor='white')

    # Update legend positions for both traces
    scatter_fig.update_layout(legend=dict(x=1, y=1, traceorder='normal', orientation='v'))
    #lines_fig.update_layout(legend=dict(x=1, y=1.15, traceorder='normal', orientation='h'))

    # Show the combined plot with markers and lines
    return scatter_fig.show()
    
def Sil_scatter(df,clus):
    scatter_fig = px.scatter(data_frame=df, x="n_clusters", y="Sil_Score", color='Data',
                          title=f'{clus} Plot')

    # Create traces for lines connecting the points
    lines_fig = px.line(data_frame=df, x="n_clusters", y="Sil_Score", color='Data', 
            line_shape='linear')
    # Update the scatter plot to show markers and lines
    
    scatter_fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    scatter_fig.add_traces(lines_fig.data)  # Add lines to the scatter plot

    # Customize the appearance of the combined plot
    scatter_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig.update_layout(plot_bgcolor='white')

    # Update legend positions for both traces
    scatter_fig.update_layout(legend=dict(x=1, y=1, traceorder='normal', orientation='v'))
    #lines_fig.update_layout(legend=dict(x=1, y=1.15, traceorder='normal', orientation='h'))

    # Show the combined plot with markers and lines
    return scatter_fig.show()