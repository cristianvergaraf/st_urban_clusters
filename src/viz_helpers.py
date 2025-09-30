import plotly.express as px
import streamlit as st

def interactive_scatter(datos,cluster):
    
    # Custom color palette
    custom_palette = ["red", "green", "blue", "black", "gray"]

    # Convert the column to string to make it categorical
    datos[cluster] = datos[cluster].astype(str)

    # Create the scatter plot using Plotly Express
    scatter_fig = px.scatter(data_frame=datos, x="variable", y="value", color=cluster,
                         color_discrete_map={value: color for value, color in zip(datos[cluster].unique(), custom_palette)},
                         labels={'variable': ' ', 'value': 'Factor value'}, title='Fig 1. Media de cada variable para cada cluster')

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
    st.plotly_chart(scatter_fig, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')


def create_splom_graph(data, dimensions, text, marker_color="blue", marker_size=5, colorscale='Bluered', line_width=0.5, line_color='rgb(230,230,230)', diagonal_visible=False):
    fig = go.Figure(data=go.Splom(
        dimensions=dimensions,
        text=text,
        marker=dict(
            color=marker_color,
            size=marker_size,
            colorscale=colorscale,
            line=dict(
                width=line_width,
                color=line_color
            )
        ),
        diagonal=dict(
            visible=diagonal_visible
        )
    ))
    
    return fig
