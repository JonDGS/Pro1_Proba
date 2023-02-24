# -*- coding: utf-8 -*-
"""
Curso Proabilidad y Estadística

Tarea #1

Template con lectura de datos en archivo csv

"""

import numpy as np
import math as mt

#input_dir='C:/Users/PATH/' #PATH al archivo de datos, cambiar según cada computadora. Sirve para evitar 'File not found'
filename='energydata_complete.csv'

# Esta línea lee la matriz de datos (sin titulos) para números solamente. Otro tipo de variable (texto por ejemplo) se leerá como nan
#datos=np.genfromtxt(filename,delimiter=';',skip_header=1)

#alternativamente, se pueden leer columnas específicas entre el rango [X,Y] de esta forma:
datos=np.genfromtxt(filename,delimiter=';',skip_header=1, usecols = (0, 12), dtype=None, encoding=None)

def calculateAverage(dataArray):

    return np.average(dataArray)

def calculateMedian(dataArray):

    return np.median(dataArray)

def calculateQuartiles(dataArray, method):

    method = method.lower()

    result = None
    
    match method:

        case "numpy":
            return calculateQuartilesNumpy(dataArray)
        case "manual":
            return calculateQuartilesManually(dataArray)
        case _:
            return (None, None)

def calculateQuartilesNumpy(dataArray):

    Q1 = np.quantile(dataArray, q = 0.25)
    Q3 = np.quantile(dataArray, q = 0.75)

    return (Q1, Q3)

def calculateQuartilesManually(dataArray):

    n = len(dataArray)

    Q1 = dataArray[mt.ceil(0.25*n)]
    Q3 = dataArray[mt.ceil(0.75*n)]

    return (Q1, Q3)

def calculateVariance(dataArray):

    return np.var(dataArray)

def calculateStandardDeviation(variance):
    return np.sqrt(variance)

def calculateVarianceCoeficient(standardDeviation, average):
    return (standardDeviation/average)*100

def calculateSampleRange(dataArraySorted):
    return np.max(dataArraySorted) - np.min(dataArraySorted)

def calculateQuantileRange(Q1, Q3):
    return Q3 - Q1

def printValues(average, median, quantiles, variance, standardDeviation, varianceCoeficient, sampleRange, quantileRange):
    print("Datos has the following values\n"
          + "Average: " + str(average) + "\n"
          + "Median: " + str(median) + "\n"
          + "Q1: " + str(quantiles[0]) + "\n"
          + "Q3: " + str(quantiles[1]) + "\n"
          + "Variance: " + str(variance) + "\n"
          + "Standard Deviation: " + str(standardDeviation) + "\n"
          + "Variance Coeficient: " + str(average) + "\n"
          + "Sample Range: " + str(sampleRange) + "\n"
          + "Quantile Range: " + str(quantileRange) + "\n")

    

def calculateStatistics(datos, printResults):

    dateArray = []
    dataArray = []

    for line in datos:
        dateArray.insert(0, line[0])
        dataArray.insert(0, line[1])

    dataArraySorted = np.sort(dataArray)
    
    #Calcular promedio
    average = calculateAverage(dataArray)

    #Calcular mediana
    median = calculateMedian(dataArraySorted)

    #Calcular Quartiles
    quantiles = calculateQuartiles(dataArraySorted, "manual")

    #Calcular varianza muestral
    variance = calculateVariance(dataArraySorted)

    #Calcular desviación estándar
    standardDeviation = calculateStandardDeviation(variance)

    #Calcular coeficiente de variación
    coeficient = calculateVarianceCoeficient(standardDeviation, average)

    #Calcular rango muestral
    sampleRange = calculateSampleRange(dataArraySorted)

    #Calcular rango interquartil
    quantileRange = calculateQuantileRange(quantiles[0], quantiles[1])

    if printResults:
        printValues(average, median, quantiles, variance, standardDeviation, coeficient, sampleRange, quantileRange)


    

calculateStatistics(datos, True)