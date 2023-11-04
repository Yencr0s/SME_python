import numpy as np
import pandas as pd
import copy
from functions.metricas import *

'''
Funcion para normalizar un vector

@param x: vector
@return: vector normalizado
'''
def v_normalizar(x):
    #si el vector es numerico se normaliza sino se lanza una excepcion
    try:
        if isinstance(x, (pd.Series)):
            x = x.values
        elif not isinstance(x, (np.ndarray)):
            x = np.array(x)
        return (x-min(x))/(max(x)-min(x))
    except:
        raise Exception('El vector debe ser numérico')

'''
Funcion para normalizar un dataframe

@param x: dataframe
@return: dataframe normalizado
'''
def t_normalizar(x):
    #Obtenemos las variables numericas
    nums = x.select_dtypes(include='number')
    #Aplicamos la funcion v_normalizar a cada columna
    aux = nums.apply(v_normalizar, axis=0)
    #Copiamos el dataframe y actualizamos las variables numericas
    x2 = copy.deepcopy(x)
    x2.update(aux)
    return x2

'''
Funcion que normaliza un dataframe o un vector.

@param x: vector o dataframe de pandas
@return: vector o dataframe normalizado
'''
def normalizar(x):
    #comprobamos si es un dataframe o un vector
    if isinstance(x, pd.DataFrame):
        return t_normalizar(x)
    else:
        return v_normalizar(x)
    


'''
Funcion para estandarizar un vector

@param x: vector
@return: vector estandarizado
'''
def v_estandarizar(x):
    #Si el vector es numerico se estandariza sino se lanza una excepcion
    try:
        if isinstance(x, (pd.Series)):
            x = x.values
        elif not isinstance(x, (np.ndarray)):
            x = np.array(x)
        return (x-np.mean(x))/desviacion(x)
    except:
        raise Exception('El vector debe ser numérico')

'''
Funcion para estandarizar un dataframe

@param x: dataframe
@return: dataframe estandarizado
'''
def t_estandarizar(x):
    #obtenemos las variables numericas
    nums = x.select_dtypes(include='number')
    #Aplicamos la funcion v_estandarizar a cada columna
    aux = nums.apply(v_estandarizar, axis=0)
    #Copiamos el dataframe y actualizamos las variables numericas
    x2 = copy.deepcopy(x)
    x2.update(aux)
    return x2

'''
Funcion que estandariza un dataframe o un vector.

@param x: vector o dataframe de pandas
@return: vector o dataframe estandarizado
'''
def estandarizar(x):
    #Comprobamos si es un dataframe o un vector
    if isinstance(x, pd.DataFrame):
        return t_estandarizar(x)
    else:
        return v_estandarizar(x)
    
