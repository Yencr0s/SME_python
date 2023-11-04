import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sn
from functions.metricas import *
from functions.filtrado import *
from functions.graficos import *

'''
Funcion que calcula la correlacion de dos vectores.

@param x: vector
@param y: vector
@return: correlacion de los vectores
'''
def v_cor(x,y):
    #Si los vectores son numericos se calcula la correlacion sino se lanza una excepcion
    try:
        cv = covarianza(x,y)
        return cv/(desviacion(x)*desviacion(y))
    except:
        raise Exception('Los vectores deben ser numéricos')

'''
Funcion que calcula la matriz de correlacion de un dataframe entre todas las variables numéricas.

@param df: dataframe
@return: dataframe con los valores de la correlacion y matriz de correlacion
'''
def correlacion(df):
    #Obtenemos las variables numericas
    num = df.select_dtypes(include='number')

    #Generamos todas las combinaciones posibles de variables
    combinations = list(itertools.combinations(num.columns, 2))

    corr=[]
    names=[]
    aux_df = pd.DataFrame(0, index=num.columns, columns=num.columns)

    #Calculamos la correlacion para cada par de variables
    for i in combinations:
        aux_c = v_cor(num[i[0]],num[i[1]])
        corr.append(aux_c)
        names.append(i[0]+'-'+i[1])
        aux_df.loc[i[0], i[1]] = aux_c
        aux_df.loc[i[1], i[0]] = aux_c
    res_df = aux_df.copy()

    #Rellenamos la diagonal con 1
    np.fill_diagonal(res_df.values, 1)

    return pd.DataFrame({'Correlacion':corr}, index=names),res_df

'''
Funcion que calcula la informacion mutua de dos vectores.

@param x: vector
@param y: vector
@return: informacion mutua de los vectores
'''
def v_mutua(x,y):
    #Calculamos la probabilidad de cada valor de cada vector
    px = x.value_counts()/len(x)
    py = y.value_counts()/len(y)
    #Calculamos la probabilidad conjunta de cada par de valores de los vectores uniendolos mediante zip
    xy = list(zip(x,y))
    pxy = pd.Series(xy).value_counts()/len(xy)
    #Calculamos la informacion mutua
    mut = 0
    for i in range(len(pxy)):
        mut += pxy[i]*np.log2(pxy[i]/(px[pxy.index[i][0]]*py[pxy.index[i][1]]))
    return mut



'''
Funcion que calcula la matriz de informacion mutua de un dataframe entre las variables categóricas.

@param df: dataframe
@return: dataframe con los valores de la informacion mutua y matriz de informacion mutua
'''
def informacion_mutua(df):
    #Obtenemos las variables categoricas
    cat = df.select_dtypes(exclude='number')

    #Generamos todas las combinaciones posibles de variables
    combinations = list(itertools.combinations(cat.columns, 2))
    
    mut=[]
    names=[]
    aux_df = pd.DataFrame(0, index=cat.columns, columns=cat.columns)
    
    #Calculamos la informacion mutua para cada par de variables
    for i in combinations:
        aux_m = v_mutua(cat[i[0]],cat[i[1]])
        mut.append(aux_m)
        names.append(i[0]+'-'+i[1])
        aux_df.loc[i[0], i[1]] = aux_m
        aux_df.loc[i[1], i[0]] = aux_m

    #Rellenamos la diagonal con la entropia de cada variable
    for i in cat.columns:
        aux_df.loc[i, i] = entropia(cat[i])

    return pd.DataFrame({'Informacion mutua':mut}, index=names),aux_df

'''
Funcion que calcula la matriz de correlacion y la matriz de informacion mutua de un dataframe entre todas las variables.

@param df: dataframe
@return cor: dataframe con los valores de correlacion de las variables numericas
@return inf: dataframe con los valores de informacion mutua de las variables categoricas
@return a: df de correlacion e informacion mutua de todas las variables
'''
def cor_mut(df):
    #Calculamos la correlacion y la informacion mutua
    cor, m_cor = correlacion(df)
    inf, m_inf = informacion_mutua(df)

    #Unimos los dos dataframes
    merged_df = pd.concat([m_cor, m_inf], axis=0)
    full = merged_df.copy()

    return cor, inf, full

