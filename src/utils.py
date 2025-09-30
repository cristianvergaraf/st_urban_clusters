def ex_variables(df):
    """ Exclude variables"""
 
    my_list = df.columns
    if '3' in dataset_clus:
        exclude_element = ['AREA_MN', "ED", "RES_PLU",'T_Viviendas','RES_UNI','SIDI',"RNMDP_2020"]
        variables_keep = [item for item in my_list if item not in exclude_element]
    else:
        exclude_element = ['F1','F2','F3','F4','F5']
        variables_keep = [item for item in my_list if item not in exclude_element]
    return variables_keep


def ajustar_data(df, variables = None):
    if 'Ciudades' in df.columns:
        df = df.drop("Ciudades", axis = 1)
    df_melt = df.melt(id_vars = variables)
    return df_melt

def group_data(df,cluster):
    im3_groups_mean = df.groupby(['variable',cluster,
    ])['value'].mean().reset_index() 
    
    return im3_groups_mean

def grafico(df,variables,cluster):
    datos_melt_todos = ajustar_data(df, variables)
    datos_group = group_data(datos_melt_todos,cluster)
    return datos_group


def display_clusters(ag_clus, cluster, num_clusters): 
    for i in range(num_clusters):
        cluster_data = ag_clus[ag_clus[cluster] == str(i)]
        st.metric(label=f"Cluster {i}", value=len(cluster_data))
        st.write(f"Cluster {i}: " + ", ".join(cluster_data['Ciudades'].tolist()))
   
