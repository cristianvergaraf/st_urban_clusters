import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.formula.api as smf

st.set_page_config(layout = "wide")

## Leer datos

#datos = pd.read_csv('ciudades_paisaje_filter.csv')
data = {}
#data['Original'] = pd.read_csv('datos_metricas_socioeconomicos_porcentajes.csv', encoding = 'ISO-8859-1')
#data['Std'] = pd.read_csv('df_datos_std.csv', encoding = 'ISO-8859-1')
#data['MinMax'] = pd.read_csv('df_datos_MinMax.csv', encoding = 'ISO-8859-1')
#data['Rscaler'] = pd.read_csv('df_datos_Rscaler.csv', encoding = 'ISO-8859-1')
#data['PTrans'] = pd.read_csv('df_datos_PTrans.csv', encoding = 'ISO-8859-1')
#data['Normalizer'] = pd.read_csv('df_datos_Normalizer.csv', encoding = 'ISO-8859-1')


data['Original'] = pd.read_csvr(r'C:\Users\crist\Documents\GitHub\manifolds\st_urban_cluster\st_urban_clusters\datos_metricas_socioeconomicos_porcentajes.csv', encoding = 'ISO-8859-1')
data['Std'] = pd.read_csv(r'C:\Users\crist\Documents\GitHub\manifolds\st_urban_cluster\st_urban_clusters\df_datos_std.csv', encoding = 'ISO-8859-1')
data['MinMax'] = pd.read_csv(r'C:\Users\crist\Documents\GitHub\manifolds\st_urban_cluster\st_urban_clusters\df_datos_MinMax.csv', encoding = 'ISO-8859-1')
data['Rscaler'] = pd.read_csv(r'C:\Users\crist\Documents\GitHub\manifolds\st_urban_cluster\st_urban_clusters\df_datos_Rscaler.csv', encoding = 'ISO-8859-1')
data['PTrans'] = pd.read_csv(r'C:\Users\crist\Documents\GitHub\manifolds\st_urban_cluster\st_urban_clusters\df_datos_PTrans.csv', encoding = 'ISO-8859-1')
data['Normalizer'] = pd.read_csv(r'C:\Users\crist\Documents\GitHub\manifolds\st_urban_cluster\st_urban_clusters\df_datos_Normalizer.csv', encoding = 'ISO-8859-1')





selected_file = st.selectbox("Seleccion de datos 1", ['Original','Std','MinMax','Rscaler','PTrans','Normalizer'])
selected_file1 = st.selectbox("Seleccion de datos 2", ['Original','Std','MinMax','Rscaler','PTrans','Normalizer'])

datos = data[selected_file]

datos1 = data[selected_file1]

datos_tabla = datos.loc[:,['TA', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'GYRATE_MN',
       'GYRATE_AM', 'GYRATE_MD', 'PRD', 'SHDI', 'SIDI', 'MSIDI', 'SHEI',
       'SIEI', 'MSIEI', 'NP', 'DIVISION', 'SPLIT', 'MESH', 'PAFRAC',
       'SHAPE_MN', 'SHAPE_MD', 'PARA_MN', 'PARA_MD', 'FRAC_MD',
       'SQUARE_MN', 'SQUARE_MD', 'IJI', 'RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias',
       'T_viv_col', 'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
       'RES_UNI']]

datos_tabla1 = datos1.loc[:,['TA', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'GYRATE_MN',
       'GYRATE_AM', 'GYRATE_MD', 'PRD', 'SHDI', 'SIDI', 'MSIDI', 'SHEI',
       'SIEI', 'MSIEI', 'NP', 'DIVISION', 'SPLIT', 'MESH', 'PAFRAC',
       'SHAPE_MN', 'SHAPE_MD', 'PARA_MN', 'PARA_MD', 'FRAC_MD',
       'SQUARE_MN', 'SQUARE_MD', 'IJI', 'RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias',
       'T_viv_col', 'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
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
       'SQUARE_MN', 'SQUARE_MD', 'IJI', 'RNMDP_2020', 'PobT', 'PobH', 'PobM',
       'Vehiculos', 'T_Viviendas', 'T_Viv_Prin', 'T_Viv_Sec', 'Viv_vacias',
       'T_viv_col', 'COM', 'ED_SING', 'EQUIP', 'IND', 'OCIO', 'OFI', 'RES_PLU',
       'RES_UNI']) 


### Definir el sidebar

### Generamos las tabs como alternativa a una app multipage

tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["Histograma", "Correlation matrix", "Scatterplot", "Resumen Datos", "Boxplot", "Scatterplot matrix"])

with tab1:
   st.title("Análisis de la distribución de variables")
   col1, col2 = st.columns(2) 
   with col1:
       st.header("Selecciona una variable continua")
       opciones = st.selectbox(label ="variables_continuas", options = variables_continuas)
   with col2:
       st.header("Selecciona numero de bins")
       bins_num = int(st.slider("Selecciona numero de bins", format = r"%g", min_value = 1, max_value = 50, value = 25, step = 1))
       
   if st.button('Presiona el button para el grafico'):    
       hist_data = datos.loc[:,opciones]
       hist_data1 = datos1.loc[:,opciones]
       group_labels = [opciones]
   # Create distplot with custom bin_size
       fig = px.histogram(hist_data,x = group_labels, nbins = bins_num)
       fig1 = px.histogram(hist_data1,x = group_labels, nbins = bins_num)
  
   # Plot !!
       st.header(f"Ditribution {selected_file} de {opciones}")
       st.plotly_chart(fig, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')
       st.header(f"Ditribution {selected_file1} de {opciones}")
       st.plotly_chart(fig1, width = 1000, height = 500, use_container_width = True,
                   vertical_alignment ='center')   

   
with tab2:
    
    cont_multi_selected = st.multiselect('Correlation Matrix', variables_continuas,
                                     default=variables_continuas)
    if st.button('Selecciona las variables'):
        st.write("""           
              """)
       
        df_corr = datos[cont_multi_selected].corr()
        df_corr1 = datos1[cont_multi_selected].corr()
     
        fig_corr = go.Figure([go.Heatmap(
            z = df_corr.values, 
            x=df_corr.index.values, 
            y=df_corr.columns.values, 
            colorscale= 'gray',
                           text=[[format_value(value) for value in row] for row in df_corr.values],
                            texttemplate="%{text}",
                            textfont={"size":10})])
     
        fig_corr.update_layout(height=300, width=1000, margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
     
        st.plotly_chart(fig_corr, width = 1000, height = 1000, use_container_width = True,
                     vertical_alignment = 'center')
     
        fig_corr1 = go.Figure([go.Heatmap(
            z = df_corr1.values, 
            x=df_corr1.index.values, 
            y=df_corr1.columns.values, 
            colorscale= 'gray',
                           text=[[format_value(value) for value in row] for row in df_corr.values],
                            texttemplate="%{text}",
                            textfont={"size":10})])
     
        fig_corr1.update_layout(height=300, width=1000, margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
     
     
     
     
        st.plotly_chart(fig_corr1, width = 1000, height = 1000, use_container_width = True,
                     vertical_alignment = 'center')
     
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
    #correlacion = datos.corr().iloc[0,1]
    st.write(f'La ecuacion de la recta ajustada para {x_axis_val} y {y_axis_val} es {ecuacion}')
    st.write(f'El coeficiente de determinacion (r2) entre {x_axis_val} y {y_axis_val} es {r2:.2f}')
    
    plotscat = px.scatter(datos, x=x_axis_val, y = y_axis_val, trendline="ols",trendline_color_override='darkblue',
                          opacity=0.65,trendline_scope="overall", hover_data =['Ciudades'])
    plotscat.update_layout(legend=dict(yanchor="top", y=1.10, xanchor="center", x=0.5))
    
    st.plotly_chart(plotscat, width = 1000, heigth = 800,
                    use_container_width = True,
                    vertical_alignment ='center')
    
    
    formula = f"{y_axis_val} ~ {x_axis_val}"
    modelo1 = smf.ols(formula = formula, data = datos1).fit()
    pendiente1 = modelo1.params[f"{x_axis_val}"]
    intercepto1 = modelo1.params['Intercept']
    ecuacion1 = f'{y_axis_val} = {pendiente1:.2f}x + {intercepto1:.2f}'
    r2_1 = modelo1.rsquared
    #correlacion = datos.corr().iloc[0,1]
    st.write(f'La ecuacion de la recta ajustada para {x_axis_val} y {y_axis_val} es {ecuacion1}')
    st.write(f'El coeficiente de determinacion (r2) entre {x_axis_val} y {y_axis_val} es {r2_1:.2f}')
    
    plotscat1 = px.scatter(datos1, x=x_axis_val, y = y_axis_val, trendline="ols",trendline_color_override='darkblue',
                          opacity=0.65,trendline_scope="overall", hover_data =['Ciudades'])
    plotscat.update_layout(legend=dict(yanchor="top", y=1.10, xanchor="center", x=0.5))
    
    st.plotly_chart(plotscat1, width = 1000, heigth = 800,
                    use_container_width = True,
                    vertical_alignment ='center')
    
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
    # 3        st.write(descripcion_metricas.loc[y_axis_val,'Comments'])

                
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
         
    #st.dataframe(describe_datos.style.format("{:.2f}"), width = 1000, height=700)
  

with tab5:  
    col30, col40 = st.columns(2, gap = 'small')
         
    with col30:
        opciones_metricas1 = st.selectbox(label ="Boxplot", options = variables_continuas)
        hovertemp1 = "<b>Ciudad: </b> %{text} <br>"
        hovertemp1 += "<b>Value: </b> %{y}"
        fig_box_plot1 = go.Figure()
        fig_box_plot1.add_trace(go.Box(y=datos[opciones_metricas1].values, name=datos[opciones_metricas1].name,
                                hovertemplate = hovertemp1,
                                text = datos1['Ciudades']))
        
        st.plotly_chart(fig_box_plot1, width = 1000, height = 1000, use_container_width = True,
                    vertical_alignment = 'center')
    with col40:
        opciones_metricas2 = st.selectbox(label ="Boxplot2", options = variables_continuas)
        hovertemp = "<b>Ciudad: </b> %{text} <br>"
        hovertemp += "<b>Value: </b> %{y}"
        fig_box_plot2 = go.Figure()
        fig_box_plot2.add_trace(go.Box(y=datos1[opciones_metricas2].values, name=datos[opciones_metricas2].name,
                                   hovertemplate = hovertemp,
                                   text = datos1['Ciudades']
                                   #
                                   ))
        
        st.plotly_chart(fig_box_plot2, width = 1000, height = 1000, use_container_width = True,
                    vertical_alignment = 'center')
     
    
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
    
