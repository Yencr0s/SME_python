import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functionsP.graficos import plot_roc_curve

#para quitar los future warnings de numpy y pandas
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

'''
Funcion para calcular la entropia de un vector

@param x: vector de numpy
@return: entropía del vector
'''
def v_entropia(x):
    #Si es una serie se convierte a array
    if isinstance(x, (pd.Series)):
        x = x.values
    #Se calcula la frecuencia de cada valor
    frec = pd.value_counts(x)
    #Se calcula la probabilidad de cada valor
    p = frec/len(x)
    #Se calcula la entropia
    return -sum(p*np.log2(p))

'''
Funcion para calcular la entropia de un dataframe

@param x: dataframe de pandas
@return: entropía de cada columna
'''
def t_entropia(x):
    #Obtenemos las variables numericas
    nums = x.select_dtypes(exclude = 'number')
    #Aplicamos la funcion v_entropia a cada columna
    return nums.apply(v_entropia, axis=0)


'''
Función que calcula la entropía de un dataframe o un vector.

@param x: vector de numpy o dataframe de pandas
@return: entropía del vector o entropía de las columnas
'''
def entropia(x):
    #Comprobamos si es un dataframe o un vector
    if isinstance(x, pd.DataFrame):
        return t_entropia(x)
    else:
        return v_entropia(x)
    

'''
Tiene como entrada un vector y devuelve la varianza de ese vector.

@param x: vector
@return: varianza del vector
'''
def v_varianza(x):
    #si el vector es numerico se calcula la varianza sino se lanza una excepcion
    try:
        #si el vector es una serie se convierte a array
        if isinstance(x, (pd.Series)):
            x = x.values
        return np.sum((x-np.mean(x))**2)/(len(x)-1)
    except:
        raise Exception('El vector debe ser numérico')

'''
Tiene como entrada un dataframe y devuelve la varianza de cada columna.

@param x: dataframe
@return: varianza de cada columna
'''
def t_varianza(x):
    #Obtenemos las variables numericas
    nums = x.select_dtypes(include='number')
    #Aplicamos la funcion v_varianza a cada columna
    varianzas = nums.apply(v_varianza, axis=0)
    return varianzas


'''
Función que calcula la varianza de un dataframe o un vector.

@param x: vector o dataframe de pandas
@return: varianza del vector o varianza de las columnas
'''
def varianza(x):
    #Comprobamos si es un dataframe o un vector
    if isinstance(x, pd.DataFrame):
        return t_varianza(x)
    else:
        return v_varianza(x)
    

'''
Funcion para calcular la covarianza de dos vectores

@param x: vector
@param y: vector
@return: covarianza entre los dos vectores
'''
def covarianza(x,y):
    #Si los vectores son numericos se calcula la covarianza sino se lanza una excepcion
    try:
        if isinstance(x, (pd.Series)):
            x = x.values
        if isinstance(y, (pd.Series)):
            y = y.values
        return np.sum((x-np.mean(x))*(y-np.mean(y)))/(len(x)-1)
    except:
        raise Exception('Los vectores deben ser numéricos')


'''
Funcion para calcular la desviación típica de un vector.

@param x: vector
@return: desviación típica del vector
'''
def desviacion(x):
    return np.sqrt(v_varianza(x))


'''
Funcion que calcula el UAC, devuelve la tasa de verdaderos positivos, la tasa de falsos positivos y el área bajo la curva.

@param df: dataframe
@param target: variable objetivo
@param positive_class: clase binaria
@return: tprl, fprl, auc
'''
def aucf(df, target, bin_class):
    #Ordenamos el dataframe por la variable objetivo
    df = df.sort_values(by=target, ascending=False)

    #obtenemos los puntos de corte
    cut_points = df[target].unique()

    #Calculamos la tasa de verdaderos positivos y la tasa de falsos positivos para cada punto de corte
    tprl, fprl = [], []
    for cut in cut_points:
        tp = len(df[(df[target] >= cut) & (df[bin_class] == 1)])
        fp = len(df[(df[target] >= cut) & (df[bin_class] == 0)])
        tn = len(df[(df[target] < cut) & (df[bin_class] == 0)])
        fn = len(df[(df[target] < cut) & (df[bin_class] == 1)])
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tprl.append(tpr)
        fprl.append(fpr)
    
    #Calculamos el área bajo la curva
    auc = sum([(tprl[i+1]-tprl[i])*(fprl[i+1]+fprl[i])/2 for i in range(len(tprl)-1)])
    return tprl,fprl, auc



'''
Funcion que realiza el cálculo de métricas para los atributos de un dataset: varianza y AUC para las variables contínuas y entropía para las discretas.
Esta funcion reconoce automaticmente el tipo de variable de cada columna del dataframe y calcula la métrica correspondiente.

@param df: dataframe
@param bin_class: clase binaria con la que queremos calcular el AUC (si no se quiere calcular el AUC, se debe poner None)
@param Roc_curve: booleano que indica si se quiere representar la curva ROC o no
@return: res (dataframe con los resultados de las variables continuas), res2(dataframe con los resultados de las variables discretas)
'''
def metricas(df, bin_class=None, Roc_curve = False):
    res = pd.DataFrame()
    res2 = pd.DataFrame()

    #Seleccionamos las variables continuas
    contcols = df.select_dtypes(exclude='object').columns

    #calculamos las varianzas de las columnas de las variables continuas del dataframe
    var = varianza(df[contcols])
    res['Varianza'] = var

    #calculamos el AUC de las columnas de las variables continuas del dataframe y representar la curva ROC si se quiere
    if bin_class!=None:
        aucl = []
        for col in contcols:
            if col != bin_class:
                tprl, fprl, auc = aucf(df, col, bin_class)
                aucl.append(auc)
                if Roc_curve == True:
                    plot_roc_curve(tprl, fprl)
        res['AUC'] = aucl

    #Seleccionamos las variables categoricas
    catcols = df.select_dtypes(exclude='number').columns

    #calculamos la entropía de las columnas de las variables discretas del dataframe
    ent = entropia(df[catcols])
    res2["Entropia"] = ent

    return res, res2