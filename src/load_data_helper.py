import pandas as pd
import streamlit as st

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