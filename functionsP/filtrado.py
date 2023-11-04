import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functionsP.metricas import *


'''
Funcion para filtrar un dataframe por entropia o varianza. Se eliminan todas aquellas
columnas que no cumplan la condicion indicada. Si se quiere filtrar por entropia, se debe 
indicar el parametro Entropy como True, si se quiere filtrar por varianza, se debe indicar
como False.

@param df: dataframe
@param Entropy: booleano para filtrar por entropia o varianza (por defecto True)
@param condition: string con la condicion para filtrar (<, >, <=, >=, ==)
@param valueV: valor para filtrar
@return: dataframe filtrado
'''
def filtrar(df, Entropy=True, condition="", value = 0):
    #Obtenemos las metricas
    var, ent = metricas(df)
    #Comprobamos si se quiere filtrar por entropia o varianza
    if Entropy:
        if condition == "<":
            #se eliminan las columnas que no cumplen la condicion
            df = df.drop(columns = ent[ent >= value].dropna().index)
        elif condition == ">":
            df = df.drop(columns = ent[ent <= value].dropna().index)
        elif condition == "<=":
            df = df.drop(columns = ent[ent > value].dropna().index)
        elif condition == ">=":
            df = df.drop(columns = ent[ent < value].dropna().index)
        elif condition == "==":
            df = df.drop(columns = ent[ent != value].dropna().index)
    else:
        if condition == "<":
            df = df.drop(columns = var[var >= value].dropna().index)
        elif condition == ">":
            df = df.drop(columns = var[var <= value].dropna().index)
        elif condition == "<=":
            df = df.drop(columns = var[var > value].dropna().index)
        elif condition == ">=":
            df = df.drop(columns = var[var < value].dropna().index)
        elif condition == "==":
            df = df.drop(columns = var[var != value].dropna().index)
    return df




