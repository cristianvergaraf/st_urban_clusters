import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import copy

st.set_page_config(layout = "wide")

# Vamos a leer la libreria cluster_function

import cluster_function as cf

# Leer datos sil_score

sil_score = pd.read_csv('sil_score.csv', encoding = "ISO-8859-1")
sil_df_kmeans = sil_score.query('Metodo == "KMeans"')
sil_df_ag = sil_score.query('Metodo == "Agglomerative"')

## Vamos a leer los df de los distintos clusters

#@st.cache_data
def load_cluster():
    dic_cluster = {}
    dic_cluster['df_Ptrans_test1_AG'] = pd.read_csv('./df_cluster_AG/df_Ptrans_test1_AG.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test1_TA_AG'] = pd.read_csv('./df_cluster_AG/df_Ptrans_test1_TA_AG.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test2_AG'] = pd.read_csv('./df_cluster_AG/df_Ptrans_test2_AG.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test2_TA_AG'] = pd.read_csv('./df_cluster_AG/df_Ptrans_test2_TA_AG.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test3_AG'] = pd.read_csv('./df_cluster_AG/df_Ptrans_test3_AG.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test3_TA_AG'] = pd.read_csv('./df_cluster_AG/df_Ptrans_test3_TA_AG.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test1_KM'] = pd.read_csv('./df_cluster_KM/df_Ptrans_test1_KM.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test1_TA_KM'] = pd.read_csv('./df_cluster_KM/df_Ptrans_test1_TA_KM.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test2_KM'] = pd.read_csv('./df_cluster_KM/df_Ptrans_test2_KM.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test2_TA_KM'] = pd.read_csv('./df_cluster_KM/df_Ptrans_test2_TA_KM.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test3_KM'] = pd.read_csv('./df_cluster_KM/df_Ptrans_test3_KM.csv',encoding = "ISO-8859-1")
    dic_cluster['df_Ptrans_test3_TA_KM'] = pd.read_csv('./df_cluster_KM/df_Ptrans_test3_TA_KM.csv',encoding = "ISO-8859-1")
    return dic_cluster

dic_cluster = load_cluster()

def ex_variables(df): 
    my_list = df.columns
    if '3' in dataset_clus:
        exclude_element = ['AREA_MN', "ED", "RES_PLU",'T_Viviendas','RES_UNI','SIDI',"RNMDP_2020"]
        variables_keep = [item for item in my_list if item not in exclude_element]
    else:
        exclude_element = ['F1','F2','F3','F4','F5']
        variables_keep = [item for item in my_list if item not in exclude_element]
    return variables_keep







key_dic_cluster = np.array(['df_Ptrans_test1_AG','df_Ptrans_test1_TA_AG','df_Ptrans_test2_AG','df_Ptrans_test2_TA_AG','df_Ptrans_test3_AG','df_Ptrans_test3_TA_AG',
                            'df_Ptrans_test1_KM','df_Ptrans_test1_TA_KM','df_Ptrans_test2_KM','df_Ptrans_test2_TA_KM','df_Ptrans_test3_KM','df_Ptrans_test3_TA_KM'])


# Def function

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
    

def grafico(df,variables,cluster):
    datos_melt_todos = cf.ajustar_data(df, variables)
    datos_group = cf.group_data(datos_melt_todos,cluster)
    return datos_group


def display_clusters(ag_clus, cluster, num_clusters): 
    for i in range(num_clusters):
        cluster_data = ag_clus[ag_clus[cluster] == str(i)]
        st.metric(label=f"Cluster {i}", value=len(cluster_data))
        st.write(f"Cluster {i}: " + ", ".join(cluster_data['Ciudades'].tolist()))
   

## Leer datos en un diccionario



#### Example usage
file_names = ['datos_metricas_socioeconomicos_porcentajes', 'df_datos_std', 'df_datos_MinMax', 'df_datos_Rscaler', 'df_datos_PTrans', 'df_datos_Normalizer', 'df_datos_Maxabs']
keys = ['Original', 'Std', 'MinMax', 'Rscaler', 'PTrans', 'Normalizer', 'Maxabs']


@st.cache_data
def load_data():
    data = {}
    data['Original'] = pd.read_csv('datos_metricas_socioeconomicos_porcentajes.csv', encoding = 'ISO-8859-1' )
    data['Std'] = pd.read_csv('df_datos_std.csv', encoding = 'ISO-8859-1')
    data['MinMax'] = pd.read_csv('df_datos_MinMax.csv', encoding = 'ISO-8859-1')
    data['Rscaler'] = pd.read_csv('df_datos_Rscaler.csv', encoding = 'ISO-8859-1')
    data['PTrans'] = pd.read_csv('df_datos_PTrans.csv', encoding = 'ISO-8859-1')
    data['Normalizer'] = pd.read_csv('df_datos_Normalizer.csv', encoding = 'ISO-8859-1')
    data['Maxabs'] = pd.read_csv('df_datos_Maxabs.csv', encoding = 'ISO-8859-1')
    return data

data = load_data()

selected_file = st.selectbox("Seleccion de datos 1", ['Original','Std','MinMax','Rscaler','PTrans','Normalizer'])
selected_file1 = st.selectbox("Seleccion de datos 2", ['Original','Std','MinMax','Rscaler','PTrans','Normalizer'])

# Leer Outliers metricas y el numero de veces que una ciudad es outliers

## Vamos a crear de igual manera dictionarios con los datos

@st.cache_data
def load_outliers_metricas():
    outliers_metricas = {}
    outliers_metricas['Original'] = pd.read_csv('df_datos_Original_outmerge.csv', index_col = [0],encoding = 'ISO-8859-1')
    outliers_metricas['Maxabs'] = pd.read_csv('df_datos_Maxabs_outmerge.csv', index_col = [0],encoding = 'ISO-8859-1')
    outliers_metricas['Std'] = pd.read_csv('df_datos_std_outmerge.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_metricas['MinMax'] = pd.read_csv('df_datos_MinMax_outmerge.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_metricas['Rscaler'] = pd.read_csv('df_datos_Rscaler_outmerge.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_metricas['PTrans'] = pd.read_csv('df_datos_PTrans_outmerge.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_metricas['Normalizer'] = pd.read_csv('df_datos_Normalizer_outmerge.csv', encoding = 'ISO-8859-1')
    return outliers_metricas

@st.cache_data
def load_outliers_ciudades():
    outliers_ciudades = {}
    outliers_ciudades['Original'] = pd.read_csv('df_datos_Original_outciudades.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_ciudades['Maxabs'] = pd.read_csv('df_datos_Maxabs_outciudades.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_ciudades['Std'] = pd.read_csv('df_datos_std_outciudades.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_ciudades['MinMax'] = pd.read_csv('df_datos_MinMax_outciudades.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_ciudades['Rscaler'] = pd.read_csv('df_datos_Rscaler_outciudades.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_ciudades['PTrans'] = pd.read_csv('df_datos_PTrans_outciudades.csv', index_col = [0], encoding = 'ISO-8859-1')
    outliers_ciudades['Normalizer'] = pd.read_csv('df_datos_Normalizer_outciudades.csv', encoding = 'ISO-8859-1')
    return outliers_ciudades


outliers_metricas = load_outliers_metricas()
outliers_ciudades = load_outliers_ciudades()


df_outmetricas = outliers_metricas[selected_file].sort_values(by = 'Outliers_Count', ascending = False)
df_outmetricas1 = outliers_metricas[selected_file1].sort_values(by = 'Outliers_Count', ascending = False)

df_outciudades = outliers_ciudades[selected_file].sort_values(by = 'Outliers_Count', ascending = False)
df_outciudades1 = outliers_ciudades[selected_file1].sort_values(by = 'Outliers_Count', ascending = False)

##########

# https://docs.streamlit.io/library/advanced-features/session-state#initialization
# https://discuss.streamlit.io/t/proper-use-of-on-change-with-st-experimental-data-editor/41704/5

def update_state():         
   st.session_state.edited_df = edit
   
def update_correlation_matrix():
        st.session_state.button_correlation = True
    
 # two dataframe in session state. An original and and edited version 
 
#1. Paso uno creamos los dataframes datos y datos1 a partir de la seleccion de los usuarios

#st.write("Elige uno de los dataset a los que se le aplicaron distintos tipos de escalamiento, check out this [https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html]")
st.markdown("Elige uno de los dataset a los que se le aplicaron distintos tipos de escalamiento: [scikit scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)")
datos = data[selected_file]
datos1 = data[selected_file1]

# Creamos dos elementos dentro de session states df y edited/df

if 'df' not in st.session_state:
    st.session_state.df = datos
    
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = datos  # Aqui coloco datos para inicializar con algo podria ser datos

      
if "button_correlation" not in st.session_state:
        st.session_state.button_correlation = False
        
if "histogram_plot" not in st.session_state:
        st.session_state.histogram_plot = False


edit = st.data_editor(datos, num_rows="dynamic")

st.button('Save changes', on_click = update_state) # Este botton me actualiza el state. De forma jerarquico el botton de datos correlacion deberia esta anidado en este.

#st.write(st.session_state.edited_df)

#####





datos_tabla = datos.loc[:,['TA', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'GYRATE_MN',
       'GYRATE_AM', 'GYRATE_MD', 'PRD', 'SHDI', 'SIDI', 'MSIDI', 'SHEI',
       'SIEI', 'MSIEI', 'NP', 'DIVISION', 'SPLIT', 'MESH', 'PAFRAC',
       'SHAPE_MN', 'SHAPE_MD', 'PARA_MN', 'PARA_MD', 'FRAC_MD',
       'SQUARE_MN', 'SQUARE_MD', 'IJI', 'LSI','TE','ED','RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias', 'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
       'RES_UNI']]

datos_tabla_editados = st.session_state.edited_df.loc[:,['TA', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'GYRATE_MN',
       'GYRATE_AM', 'GYRATE_MD', 'PRD', 'SHDI', 'SIDI', 'MSIDI', 'SHEI',
       'SIEI', 'MSIEI', 'NP', 'DIVISION', 'SPLIT', 'MESH', 'PAFRAC',
       'SHAPE_MN', 'SHAPE_MD', 'PARA_MN', 'PARA_MD', 'FRAC_MD',
       'SQUARE_MN', 'SQUARE_MD', 'IJI', 'LSI','TE','ED','RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias', 'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
       'RES_UNI']]

datos_tabla1 = datos1.loc[:,['TA', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'GYRATE_MN',
       'GYRATE_AM', 'GYRATE_MD', 'PRD', 'SHDI', 'SIDI', 'MSIDI', 'SHEI',
       'SIEI', 'MSIEI', 'NP', 'DIVISION', 'SPLIT', 'MESH', 'PAFRAC',
       'SHAPE_MN', 'SHAPE_MD', 'PARA_MN', 'PARA_MD', 'FRAC_MD',
       'SQUARE_MN', 'SQUARE_MD', 'IJI','LSI', 'TE','ED','RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias', 'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
       'RES_UNI']]
        
descripcion_metricas = pd.read_csv("metricas_descripcion.csv", index_col = 'Metric')


## Definicion de funciones

def format_value(value):
         if  (value >= 0.3) & (value <= 0.995):
             color = 'rgba(255, 0, 0, 1)'
         elif  (value <= -0.3) & (value >= -0.995):
             color = 'rgba(255, 0, 0, 1)'
         else:
             color = 'rgba(0, 0, 0, 0)'
         return f'<span style="color:{color}">{value:.2f}</span>'
     

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

## Definicion de variables con nombres de variables

variables_continuas = np.array(['TA', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'GYRATE_MN',
       'GYRATE_AM', 'GYRATE_MD', 'PRD', 'SHDI', 'SIDI', 'MSIDI', 'SHEI',
       'SIEI', 'MSIEI', 'NP', 'DIVISION', 'SPLIT', 'MESH', 'PAFRAC',
       'SHAPE_MN', 'SHAPE_MD', 'PARA_MN', 'PARA_MD', 'FRAC_MD',
       'SQUARE_MN', 'SQUARE_MD', 'IJI','LSI','TE','ED','RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias',
        'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
       'RES_UNI']) 


### Definir el sidebar

### Generamos las tabs como alternativa a una app multipage

tab1, tab2, tab3, tab4, tab5,tab6, tab7, tab8, tab9 = st.tabs(["Histograma", "Correlation matrix", "Scatterplot", "Resumen Datos", "Boxplot", "Scatterplot matrix", "Silhouette score","Cluster analysis", "Datos Clusters"])

with tab1:
   st.title("Análisis de la distribución de variables")
   col1, col2 = st.columns(2) 
   with col1:
       st.header("Selecciona una variable continua")
       opciones = st.selectbox(label ="variables_continuas", options = variables_continuas)
   with col2:
       st.header("Selecciona numero de bins")
       bins_num = int(st.slider("Selecciona numero de bins", format = r"%g", min_value = 1, max_value = 50, value = 25, step = 1))
       
   
   
   histo_button = st.button("Presiona el botón para ver los histogramas")
   
   if histo_button or st.session_state.histogram_plot:    
       hist_data = datos.loc[:,opciones]
       hist_data_editado = st.session_state.edited_df.loc[:,opciones] ## Hemos agregado aqui el data editado
       hist_data1 = datos1.loc[:,opciones]
       group_labels = [opciones]
   # Create distplot with custom bin_size
       fig = px.histogram(hist_data,x = group_labels, nbins = bins_num)
       fig_editado = px.histogram(hist_data_editado,x = group_labels, nbins = bins_num)
       fig1 = px.histogram(hist_data1,x = group_labels, nbins = bins_num)
       
  
   # Plot !!
       st.header(f"Ditribution {selected_file} de {opciones}")
       st.plotly_chart(fig, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
       
       st.header(f"Ditribution {selected_file} de {opciones} Editado")
       st.plotly_chart(fig_editado, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
       
       st.header(f"Ditribution {selected_file1} de {opciones}")
       st.plotly_chart(fig1, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')   


# CORRELACION
with tab2:
    
    
    
    cont_multi_selected = st.multiselect('Correlation Matrix', variables_continuas,
                                     default=['ED_SING','AREA_MN','ED','RES_PLU','TA','T_Viviendas','RES_UNI','SIDI', 'RNMDP_2020'])
    
    
    def style_negative_blue(val):
            
            if val == 1:
                color = 'yellow'
            elif 0.3 <= val <= 0.99:
                color = 'blue'
                
            elif -0.99 <= val <= -0.3:
                color = 'red'
            else:
                color = 'white'  # Set default color for other values
            return f'color: {color}'
        
    load = st.button('Selecciona las variables')
    # initialization
    
        
    if load or  st.session_state.button_correlation: 
        
        df_corr = datos[cont_multi_selected].corr()
        df_corr_editado = st.session_state.edited_df[cont_multi_selected].corr()
        df_corr1 = datos1[cont_multi_selected].corr()
    
        #st.write((st.session_state.edited_df).astype('object')) Este codigo es para probar la reactividad.
        #st.dataframe(datos1) Este igual es para ver la reactividad.
        
        
        st.dataframe(df_corr.style.applymap(style_negative_blue))    
        st.dataframe(df_corr_editado.style.applymap(style_negative_blue))
        st.dataframe(df_corr1.style.applymap(style_negative_blue))
                     
    
     
    #fig_corr = go.Figure([go.Heatmap(
     #       z = df_corr.values, 
      #      x=df_corr.index.values, 
       #     y=df_corr.columns.values, 
        #    colorscale= 'gray',
         #                  text=[[format_value(value) for value in row] for row in df_corr.values],
          #                  texttemplate="%{text}",
           #                 textfont={"size":10})])
     
    #fig_corr.update_layout(height=300, width=1000, margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
     
    #st.plotly_chart(fig_corr, width = 1000, height = 1000, use_container_width = True,
                   #  vertical_alignment = 'center')
     
    #fig_corr1 = go.Figure([go.Heatmap(
     #       z = df_corr1.values, 
      #      x=df_corr1.index.values, 
       #     y=df_corr1.columns.values, 
        #    colorscale= 'gray',
         #                  text=[[format_value(value) for value in row] for row in df_corr1.values],
          #                  texttemplate="%{text}",
           #                 textfont={"size":10})])
     
    #fig_corr1.update_layout(height=300, width=1000, margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
     
     
    #st.plotly_chart(fig_corr1, width = 1000, height = 1000, use_container_width = True,
     #                vertical_alignment = 'center')
     
# Scatterplot

with tab3:
    col1, col2 = st.columns(2, gap = 'small')
    with col1:
         x_axis_val = st.selectbox('Select X-Axis Value', options = variables_continuas)
    with col2:
         y_axis_val = st.selectbox('Select Y-Axis Value', options = variables_continuas)
     
    formula = f"{y_axis_val} ~ {x_axis_val}"
    modelo = smf.ols(formula = formula, data = datos).fit()
    pendiente = modelo.params[f"{x_axis_val}"]
    intercepto = modelo.params['Intercept']
    ecuacion = f'{y_axis_val} = {pendiente:.2f}x + {intercepto:.2f}'
    r2 = modelo.rsquared
    correlation = np.sqrt(r2)
    
    st.subheader("SELECCION DE DATOS 1")
    
    col1,col2,col3 = st.columns(3)
    col1.write(f'La ecuacion de la recta ajustada para {x_axis_val} y {y_axis_val} es {ecuacion}')
    col2.metric(f'Coeficiente de Correlación {x_axis_val} y {y_axis_val} ',f'{correlation:.2f}')
    col3.metric(f'Coeficiente de determinación (r2) entre {x_axis_val} y {y_axis_val}', f'{r2:.2f}')
    
    plotscat = px.scatter(datos, x=x_axis_val, y = y_axis_val, trendline="ols",trendline_color_override='darkblue',
                          opacity=0.65,trendline_scope="overall", hover_data =['Ciudades'])
    plotscat.update_layout(legend=dict(yanchor="top", y=1.10, xanchor="center", x=0.5))
    
    st.plotly_chart(plotscat, width = 1000, heigth = 800,
                    use_container_width = True,
                    vertical_alignment ='center')
    
    
    ##########################################################################
    
    formula = f"{y_axis_val} ~ {x_axis_val}"
    modelo1 = smf.ols(formula = formula, data = datos1).fit()
    pendiente1 = modelo1.params[f"{x_axis_val}"]
    intercepto1 = modelo1.params['Intercept']
    ecuacion1 = f'{y_axis_val} = {pendiente1:.2f}x + {intercepto1:.2f}'
    r2_1 = modelo1.rsquared
    correlation_1 = np.sqrt(r2_1)
    
    st.subheader("SELECCION DE DATOS 1 EDITADO")
    
    col1,col2,col3 = st.columns(3)
    col1.write(f'La ecuacion de la recta ajustada para {x_axis_val} y {y_axis_val} es {ecuacion}')
    col2.metric(f'El coeficiente de Correlación {x_axis_val} y {y_axis_val} es',f'{correlation:.2f}')
    col3.metric(f'El coeficiente de determinación (r2) entre {x_axis_val} y {y_axis_val} es', f'{r2:.2f}')
    
    plotscat = px.scatter(st.session_state.edited_df, x=x_axis_val, y = y_axis_val, trendline="ols",trendline_color_override='darkblue',
                          opacity=0.65,trendline_scope="overall", hover_data =['Ciudades'])
    plotscat.update_layout(legend=dict(yanchor="top", y=1.10, xanchor="center", x=0.5))
    
    st.plotly_chart(plotscat, width = 1000, heigth = 800,
                    use_container_width = True,
                    vertical_alignment ='center')

    
    #################################################
    
   
    
    formula = f"{y_axis_val} ~ {x_axis_val}"
    modelo1 = smf.ols(formula = formula, data = datos1).fit()
    pendiente1 = modelo1.params[f"{x_axis_val}"]
    intercepto1 = modelo1.params['Intercept']
    ecuacion1 = f'{y_axis_val} = {pendiente1:.2f}x + {intercepto1:.2f}'
    r2_1 = modelo1.rsquared
    correlation_1 = np.sqrt(r2_1)
    
    st.subheader("SELECCION DE DATOS 2")
             
    col1,col2,col3 = st.columns(3)
    col1.write(f'La ecuacion de la recta ajustada para {x_axis_val} y {y_axis_val} es {ecuacion1}')
    col2.metric(f'El coeficiente de Correlación {x_axis_val} y {y_axis_val} es',f'{correlation_1:.2f}')
    col3.metric(f'El coeficiente de determinación (r2) entre {x_axis_val} y {y_axis_val} es', f'{r2_1:.2f}')
    
    #correlacion = datos.corr().iloc[0,1]
    #st.write(f'El coeficiente de determinacion (r2) entre {x_axis_val} y {y_axis_val} es {r2_1:.2f}')
    #st.write(f'El coeficiente de correlacion entre {x_axis_val} y {y_axis_val} es {correlation_1:.2f}')
    
    plotscat1 = px.scatter(datos1, x=x_axis_val, y = y_axis_val, trendline="ols",trendline_color_override='darkblue',
                          opacity=0.65,trendline_scope="overall", hover_data =['Ciudades'])
    plotscat.update_layout(legend=dict(yanchor="top", y=1.10, xanchor="center", x=0.5))
    
    st.plotly_chart(plotscat1, width = 1000, heigth = 800,
                    use_container_width = True,
                    vertical_alignment ='center')
    
    # Esta es una parte donde te pueden explicar las metricas
    
    #col5, col6 = st.columns(2, gap = 'small')
    #with col5:
    #    st.subheader(descripcion_metricas.loc[x_axis_val,'Name'])
    #    with st.expander('Description'):
    #        st.write(descripcion_metricas.loc[x_axis_val,'Description'])
    #    with st.expander("Range"):
    #        st.write(descripcion_metricas.loc[x_axis_val,'Range'])
    #    with st.expander('Comments'):
    #        st.write(descripcion_metricas.loc[x_axis_val,'Comments'])
    #with col6:
    #    st.subheader(descripcion_metricas.loc[y_axis_val,'Name'])
    #    with st.expander('Description'):
    #        st.write(descripcion_metricas.loc[y_axis_val,'Description'])
    #    with st.expander("Range"):
    #        st.write(descripcion_metricas.loc[y_axis_val,'Range'])
    #    with st.expander('Comments'):
    #         st.write(descripcion_metricas.loc[y_axis_val,'Comments'])

                
with tab4:
    col10, col20 = st.columns(2, gap = 'small')
    with col10:
        st.header(f"Resumen Datos {selected_file}")
        describe_datos = datos_tabla.describe().T
        st.dataframe(describe_datos)
               
    with col20:
        st.header(f"Resumen Datos {selected_file1}")
        describe_datos1 = datos_tabla1.describe().T
        st.dataframe(describe_datos1)
     
    
    st.header(f"Resumen Datos Editados {selected_file}")
    describe_datos_editados = datos_tabla_editados.describe().T
    st.dataframe(describe_datos_editados)
    
    #st.dataframe(describe_datos.style.format("{:.2f}"), width = 1000, height=700)
  

## Boxplot and outliers

with tab5:  
    col30, col40,col50 = st.columns(3, gap = 'small')
         
    
    with col30:
        opciones_metricas1 = st.selectbox(label ="Boxplot Datos 1", options = variables_continuas)
        hovertemp1 = "<b>Ciudad: </b> %{text} <br>"
        hovertemp1 += "<b>Value: </b> %{y}"
        fig_box_plot1 = go.Figure()
        fig_box_plot1.add_trace(go.Box(y=datos[opciones_metricas1].values, name=datos[opciones_metricas1].name,
                                hovertemplate = hovertemp1,
                                text = datos1['Ciudades']))
        
        st.plotly_chart(fig_box_plot1, width = 1000, height = 1000, use_container_width = True,
                    vertical_alignment = 'center')
          
    with col40:
        opciones_metricas2 = st.selectbox(label ="Boxplot Datos 1 Editados", options = variables_continuas)
        hovertemp1 = "<b>Ciudad: </b> %{text} <br>"
        hovertemp1 += "<b>Value: </b> %{y}"
        fig_box_plot2 = go.Figure()
        fig_box_plot2.add_trace(go.Box(y=st.session_state.edited_df[opciones_metricas1].values, name=st.session_state.edited_df[opciones_metricas1].name,
                                hovertemplate = hovertemp1,
                                text = datos1['Ciudades']))
        
        st.plotly_chart(fig_box_plot2, width = 1000, height = 1000, use_container_width = True,
                    vertical_alignment = 'center')
        
                
         
    with col50:
        opciones_metricas3 = st.selectbox(label ="Boxplot Datos 2", options = variables_continuas)
        hovertemp = "<b>Ciudad: </b> %{text} <br>"
        hovertemp += "<b>Value: </b> %{y}"
        fig_box_plot3 = go.Figure()
        fig_box_plot3.add_trace(go.Box(y=datos1[opciones_metricas2].values, name=datos1[opciones_metricas2].name,
                                   hovertemplate = hovertemp,
                                   text = datos1['Ciudades']
                                   #
                                   ))
        
        st.plotly_chart(fig_box_plot3, width = 1000, height = 1000, use_container_width = True,
                    vertical_alignment = 'center')
        


           
    coldf1, coldf2= st.columns(2, gap = 'small')
    
    with coldf1:
        
        st.dataframe(df_outmetricas)
        st.dataframe(df_outciudades)    
       
    
    with coldf2:
        st.dataframe(df_outmetricas1)
        st.dataframe(df_outciudades1)  
        
    
    
           
with tab6:
    
    variables_seleccionadas = st.multiselect('Scatterplot matrix', variables_continuas,
                                     default=['COM','ED_SING'])
    
    
    def dim(variables_seleccionadas):
        dim = []
        for var in variables_seleccionadas:
            t = dict(label=var, values=datos[var])
            dim.append(t)
        return dim
    
    dim1 = dim(variables_seleccionadas)
    
    fig_matrix = create_splom_graph(
    data=datos,
    marker_size=5,
    dimensions= dim1,
    text=datos['Ciudades']
    )

    fig_matrix.update_layout(title="Scatterplot matrix",
                  dragmode='select',
                  width=1000,
                  height=1000,
                  hovermode='closest')

    st.plotly_chart(fig_matrix, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
    
    
with tab7:
    
    st.info('Los conjuntos de datos que terminan en TA, no tienen datos de Torrejón de Ardoz', icon="ℹ️")

    with st.expander("Variables dataset test1"):
        st.write("""
                F1 = LSI * 0.810 + TE * 0.924 + T_Viviendas * 0.979 + PobT * 0.979 + Vehiculos * 0.965 \n
                F2 = ED * 0.930 + AREA_MN * -0.878 \n
                F3 = SHEI * 0.991 + SIDI * 0.951 \n
                F4 = LPI * -0.821 + AREA_AM * -0.862 + MESH * -0.881 + SPLIT * 0.804 + DIVISION * 0.793 \n
                F5 = RES_UNI * -0.784
            """)
        
    with st.expander("variables dataset test2"):
        st.write("""
                F1 = AREA_MN * 0.815 + ED * 0.913 + RES_PLU  * 0.654 \n
                F2 = LPI * -0.848 + SPLIT * 0.834 + MESH * -0.764 \n
                F3 = LSI * 0.755 + T_Viviendas * 0.957 \n
                F4 = RES_UNI * 0.817 \n
                F5 = SIDI * 0.751 \n
                
                     """)
            
    with st.expander("variables dataset test3"):
        st.write("""
                AREA_MN, ED, RES_PLUS, T_Viviendas, RES_UNI, SIDI, RNMDP_2020
                     """)
 
  
    st.text('Fig 1. Índice de Silhouette por número de cluster para cada dataset utilizando el método Kmeans')
    scatter_fig_kmeans = px.scatter(data_frame=sil_df_kmeans, x="n_clusters", y="Sil_Score", color='Data',
                          title=f'KMeans Plot')

    # Create traces for lines connecting the points
    lines_fig_kmeans = px.line(data_frame=sil_df_kmeans, x="n_clusters", y="Sil_Score", color='Data', 
            line_shape='linear')
    # Update the scatter plot to show markers and lines
    
    scatter_fig_kmeans.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    scatter_fig_kmeans.add_traces(lines_fig_kmeans.data)  # Add lines to the scatter plot

    # Customize the appearance of the combined plot
    scatter_fig_kmeans.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig_kmeans.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig_kmeans.update_layout(plot_bgcolor='white')

    # Update legend positions for both traces
    scatter_fig_kmeans.update_layout(legend=dict(x=1, y=1, traceorder='normal', orientation='v'))
    #lines_fig.update_layout(legend=dict(x=1, y=1.15, traceorder='normal', orientation='h'))

    st.plotly_chart(scatter_fig_kmeans, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
    
    st.text("""Fig 2. Índice de Silhouette por número de cluster para cada dataset utilizando el método 
                 agrupamiento jerárquico """)
    scatter_fig_ag = px.scatter(data_frame=sil_df_ag, x="n_clusters", y="Sil_Score", color='Data',
                          title=f'Agglomerative Plot')

    # Create traces for lines connecting the points
    lines_fig_ag = px.line(data_frame=sil_df_ag, x="n_clusters", y="Sil_Score", color='Data', 
            line_shape='linear')
    # Update the scatter plot to show markers and lines
    
    scatter_fig_ag.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    scatter_fig_ag.add_traces(lines_fig_ag.data)  # Add lines to the scatter plot

    # Customize the appearance of the combined plot
    scatter_fig_ag.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig_ag.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    scatter_fig_ag.update_layout(plot_bgcolor='white')

    # Update legend positions for both traces
    scatter_fig_kmeans.update_layout(legend=dict(x=1, y=1, traceorder='normal', orientation='v'))
    #lines_fig.update_layout(legend=dict(x=1, y=1.15, traceorder='normal', orientation='h'))

    st.plotly_chart(scatter_fig_ag, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
    
with tab8:
    
    st.info('Aquí encontrarás las variables de los dintintos conjuntos de datos', icon="ℹ️")
    with st.expander("Variables dataset test1"):
        st.write("""
                F1 = LSI * 0.810 + TE * 0.924 + T_Viviendas * 0.979 + PobT * 0.979 + Vehiculos * 0.965 \n
                F2 = ED * 0.930 + AREA_MN * -0.878 \n
                F3 = SHEI * 0.991 + SIDI * 0.951 \n
                F4 = LPI * -0.821 + AREA_AM * -0.862 + MESH * -0.881 + SPLIT * 0.804 + DIVISION * 0.793 \n
                F5 = RES_UNI * -0.784 
            """)
        
    with st.expander("variables dataset test2"):
        st.write("""
                F1 = AREA_MN * 0.815 + ED * 0.913 + RES_PLU  * 0.654 \n
                F2 = LPI * -0.848 + SPLIT * 0.834 + MESH * -0.764 \n
                F3 = LSI * 0.755 + T_Viviendas * 0.957 \n
                F4 = RES_UNI * 0.817 \n
                F5 = SIDI * 0.751 \n
                
                     """)
            
    with st.expander("variables dataset test3"):
        st.write("""
                AREA_MN, ED, RES_PLUS, T_Viviendas, RES_UNI, SIDI, RNMDP_2020
                     """)
            
 
    # Revisar que estamos haciendo aqui
    
    # Damos la opcion de seleccionar por el nombre uno de los dataset
    dataset_clus = st.selectbox(label ="seleccionar dataset de entrada", options= key_dic_cluster)
    
    # Creamos el ag_clus con la seleccion del dataset y luego le incorporamos la 
    # columna ciudades. Este dataset lo utilizaremos para los hover
    ag_clus = copy.deepcopy(dic_cluster[dataset_clus]) 
    ag_clus['Ciudades'] = data['PTrans'].loc[:,['Ciudades']] #borrar revisar
    
    # creamos un ag_clus1 sin columna ciudades 
    ag_clus1 = dic_cluster[dataset_clus]
     
    # Aplicamos la funcion ex_variables para identificar cuales son las 
    # variables de cada dataset
    ag_variables1 = ex_variables(dic_cluster[dataset_clus])
    
    # Creamos una lista de variables y luego las pasamos como argumento para la selecion
    # dentro de un selectbox
    
    sel_cluster = np.array(ag_variables1)
    cluster = st.selectbox(label ="Elegir cluster", options = sel_cluster )
    
    # Identificamos el numero de cluster a partir de la posicion 2 del string del nombre
    # de la columan seleccionada (Esto falla para el cluster 10)
    
    n = int(cluster[2])
    
    # Transformamos la columna que queremos graficar a string para que la leyenda 
    # salga categorica
    
    ag_clus[cluster] = ag_clus[cluster].astype('string')

    ## aplicamos la funcion grafico que realiza la preparacion de los datos para realizar el grafico
    # con interactive_scatter.
    
    # utilizamos ag_clus1 necesitamos el dataset sin columnas con una variable string como ciudad.
    # 
    dt1 = grafico(ag_clus1,ag_variables1,cluster)
    
    # dentro de la funcion identificamos si el nombre del dataset tiene el numero 1,2,3 para identificar las variables 
    interactive_scatter(dt1,cluster)
    
    if "3" in dataset_clus:
        factores_var = np.array(['AREA_MN', "ED", "RES_PLU",'T_Viviendas','RES_UNI','SIDI','ED.1','RES_PLU',"RNMDP_2020"])
    else:
        factores_var = np.array(['F1','F2','F3','F4','F5'])
            
    col1, col2 = st.columns(2, gap = 'small')

    with col1:
         x_axis_val = st.selectbox('Select X-Axis Value', options = factores_var, index = 0)
    with col2:
         y_axis_val = st.selectbox('Select Y-Axis Value', options = factores_var, index = 1)
 
     #Custom color palette
    custom_palette = ["red", "green", "blue", "white", "yellow"]

    # Create the scatter plot using Plotly Express
    
    fig = px.scatter(data_frame=ag_clus, x=x_axis_val, y=y_axis_val, color=cluster, hover_data = [cluster,'Ciudades'],
                         color_discrete_map={value: color for value, color in zip(ag_clus[cluster].unique(), custom_palette)},
                         labels={'variable': ' ', 'value': 'Factor value'}, title='Fig 2. Distribución de las observaciones agrupadas por color para cada cluster')

     
    st.plotly_chart(fig, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
    
    # Customize the layout (optional)
    col1, col2, col3 = st.columns(3, gap = 'small')
    with col1:
         x_axis_val1 = st.selectbox('X-Axis Value', options = factores_var, index = 0)
    with col2:
         y_axis_val1 = st.selectbox('Y-Axis Value', options = factores_var, index = 1)
    with col3:
         z_axis_val1 = st.selectbox('Z-Axis Value', options = factores_var, index = 2)
    
    
    fig3d = px.scatter_3d(ag_clus, x = x_axis_val1, y = y_axis_val1, z = z_axis_val1, color = cluster, hover_data = [cluster,'Ciudades'],
                          color_discrete_map={value: color for value, color in zip(ag_clus[cluster].unique(), custom_palette)},
                         labels={'variable': ' ', 'value': 'Factor value'}, title='Municipios por cluster')
    fig3d.update_layout(title='3D Scatter Plot', scene=dict(xaxis_title=f'{x_axis_val1}', yaxis_title=f'{y_axis_val1}', zaxis_title=f'{z_axis_val1}'))

    # Show the plot
    st.plotly_chart(fig3d, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
    
    
    display_clusters(ag_clus, cluster, n) 
    
    with tab9:
        
        st.info('Aquí encontrarás las variables de los dintintos conjuntos de datos', icon="ℹ️")
        with st.expander("Variables dataset test1"):
            st.write("""
                F1 = LSI * 0.810 + TE * 0.924 + T_Viviendas * 0.979 + PobT * 0.979 + Vehiculos * 0.965 \n
                F2 = ED * 0.930 + AREA_MN * -0.878 \n
                F3 = SHEI * 0.991 + SIDI * 0.951 \n
                F4 = LPI * -0.821 + AREA_AM * -0.862 + MESH * -0.881 + SPLIT * 0.804 + DIVISION * 0.793 \n
                F5 = RES_UNI * -0.784
            """)
        
        with st.expander("variables dataset test2"):
            st.write("""
                F1 = AREA_MN * 0.815 + ED * 0.913 + RES_PLU  * 0.654 \n
                F2 = LPI * -0.848 + SPLIT * 0.834 + MESH * -0.764 \n
                F3 = LSI * 0.755 + T_Viviendas * 0.957 \n
                F4 = RES_UNI * 0.817 \n
                F5 = SIDI * 0.751 
                
                     """)
            
        with st.expander("variables dataset test3"):
            st.write("""
                AREA_MN, ED, RES_PLUS, T_Viviendas, RES_UNI, SIDI, RNMDP_2020
                     """)
            
        file = st.selectbox(label ="Selecciona un archivo", options= key_dic_cluster)
        clus_dataframe = dic_cluster[file]
        st.dataframe(clus_dataframe)
        st.write(clus_dataframe.describe())
        
    
    
 