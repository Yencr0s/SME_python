import numpy as np
import pandas as pd
import copy

'''
Funcion para disctetizar un vector de datos

@param x: vector
@param bins: puntos de corte
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: vector discretizado
'''
def discretizar(x, bins, labels = False):
    #Añadimos los extremos del vector a los puntos de corte
    bins = [float('-inf')] + bins + [float('inf')]
    #si se quieren las etiquetas se genera un array con los intervalos
    if labels:
        labels = [f'[{bins[i]:.2f}, {bins[i+1]:.2f}]' if i > 0 else f'[{bins[i]:.2f}, {bins[i+1]:.2f}]' for i in range(len(bins)-1)]

    return pd.cut(x, bins=bins, labels=labels, duplicates='drop')


'''
Funcion para discretizar un vector con el metodo de equal width

@param x: vector
@param num_bins: número de intervalos que queremos obtener
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: vector discretizado
'''
def v_discretizarEW(x, num_bins, labels = False):
    #Si es una serie se convierte a array
    if isinstance(x, (pd.Series)):
        x = x.values
    #Se calcula el ancho de los intervalos
    min_val = min(x)
    max_val = max(x)
    bin_width = (max_val - min_val) / num_bins
    #Se calculan los puntos de corte
    cut_points = [min_val + i*bin_width for i in range(1, num_bins)]

    return discretizar(x, cut_points, labels)




'''
Funcion para discretizar un dataframe con el metodo de equal width

@param x: dataframe
@param num_bins: número de intervalos que queremos obtener
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: dataframe discretizado
'''
def t_discretizarEW(x, num_bins, labels = False):
    #Obtenemos las variables numericas
    nums = x.select_dtypes(include='number')
    #Se discretizan las variables numericas
    aux = nums.apply(v_discretizarEW, axis=0, args=(num_bins, labels))
    #Se copia el dataframe y se actualizan las variables numericas
    x2 = copy.deepcopy(x)
    x2.update(aux)
    #Si no se quieren las etiquetas se convierten las variables numericas a enteros
    if not labels:
        x2[aux.columns] = x2[aux.columns].astype('int')
    return x2


'''
Función para discretizar un dataframe o un vector con el algoritmo Equal Width.

@param x: dataframe o vector
@param num_bins: número de intervalos que queremos obtener
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: dataframe o vector discretizado
'''
def discretizarEW(x, num_bins, labels = False):
    #Comprobamos si es un dataframe o un vector
    if isinstance(x, pd.DataFrame):
        return t_discretizarEW(x, num_bins, labels)
    else:
        return v_discretizarEW(x, num_bins, labels)
    

'''
Funcion para discretizar un vector con el metodo de equal frequency

@param x: vector
@param num_bins: número de intervalos que queremos obtener
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: vector discretizado
'''
def v_disctetizarEF(x,num_bins, labels = False):
    #Si es una serie se convierte a array
    if isinstance(x, (pd.Series)):
        x = x.values
    #Se ordena el array
    x_sorted = sorted(x)
    #Se calculan los puntos de corte
    cut_points = [x_sorted[int(i*len(x_sorted)/num_bins)] for i in range(1, num_bins)]
    return discretizar(x, cut_points, labels)

'''
Funcion para discretizar un dataframe con el metodo de equal frequency

@param x: dataframe
@param num_bins: número de intervalos que queremos obtener
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: dataframe discretizado
'''
def t_discretizarEF(x, num_bins, labels = False):
    #Obtenemos las variables numericas
    nums = x.select_dtypes(include='number')
    #Se discretizan las variables numericas
    aux = nums.apply(v_disctetizarEF, axis=0, args=(num_bins, labels))
    #Se copia el dataframe y se actualizan las variables numericas
    x2 = copy.deepcopy(x)
    x2.update(aux)
    #Si no se quieren las etiquetas se convierten las variables numericas a enteros
    if not labels:
        x2[aux.columns] = x2[aux.columns].astype('int')
    return x2

'''
Función para discretizar un dataframe o un vector con el algoritmo Equal Frequency.

@param x: dataframe o vector
@param num_bins: número de intervalos que queremos obtener
@param labels: Booleano para indicar si se quiere que los intervalos se muestren en el array (True) o solo el indice del intervalo (False)
@return: dataframe o vector discretizado
'''
def discretizarEF(x, num_bins, labels = False):
    #Comprobamos si es un dataframe o un vector
    if isinstance(x, pd.DataFrame):
        return t_discretizarEF(x, num_bins, labels)
    else:
        return v_disctetizarEF(x, num_bins, labels)