import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
Función para representar la curva ROC.

@param tprl: lista de valores de verdaderos positivos
@param fprl: lista de valores de falsos positivos
@param zoom: booleano para indicar si se quiere hacer zoom en la gráfica
'''
def plot_roc_curve(tprl, fprl, zoom = False):
    fig = plt.figure()
    a1 = fig.add_axes([0,0,1,1])
    a1.plot(fprl, tprl, color='darkorange', lw=2)
    a1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    a1.set_title('exp')
    if zoom:
        a1.set_xlim(min(fprl)-0.01,max(fprl)+0.01)
        a1.set_ylim(min(tprl)-0.01,max(tprl)+0.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()

'''
Funcion para representar el heatmap de la matriz de correlaciones e informacion mutua

@param df: dataframe en el que se encuentran los valores de las correlaciones e informaciones mutuas
@param df2: (optional) dataframe en el que nos basaremos para ordenar las variables en el plot (en caso de no proporcionarlo se tomará df como orden)
esto hará que las variables continuas y las discretas estén separadas en dos bloques
'''
def plot_mat_heatmap(df,df2=pd.DataFrame()):
    if df2.empty:
        df2 = df
    d = df.reindex(index = df2.columns,columns=df2.columns)
    sns.heatmap(d, annot=True, cmap='coolwarm', fmt=".2f", vmax=1)
