# -*- coding: utf-8 -*-
"""
Curso Proabilidad y Estadística

Tarea #1

Template con lectura de datos en archivo csv

"""

import numpy as np
import math as mt
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cbook as cbook

def loadData():
    #input_dir='C:/Users/PATH/' #PATH al archivo de datos, cambiar según cada computadora. Sirve para evitar 'File not found'
    filename='energydata_complete.csv'

    # Esta línea lee la matriz de datos (sin titulos) para números solamente. Otro tipo de variable (texto por ejemplo) se leerá como nan
    #datos=np.genfromtxt(filename,delimiter=';',skip_header=1)

    #alternativamente, se pueden leer columnas específicas entre el rango [X,Y] de esta forma:
    datos=np.genfromtxt(filename,delimiter=';',skip_header=1, usecols = (0, 12), dtype=None, encoding=None)

    return datos


#Calculates the average of the data along an specific axis
#using numpy.average with just sums the values and divides them
#by the amount of values
#Parameters:
# - dataArray: raw data from csv
#Return: Average of data
def calculateAverage(dataArray):

    return np.average(dataArray)

#Calculates the median of the data using numpy.median
#which computes the median finding the X((n-1)/2), if
#n happened to be odd, it uses the average of the 2 middle
#values
#Parameters:
# - dataArray: sorted data from csv in ascending order
#Return: Median of data
def calculateMedian(dataArray):

    return np.median(dataArray)

#Auxiliare function for calling functions that calculate quantiles
#Parameters:
# - dataArray: sorted values from csv in ascending order
# - method: which of the methods know to use
#Return: list of quantiles
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

#Calculates quantiles using numpy.quantile whcih uses a linear method
#meaning it calculates the quantile using the following equation
# i + g = q*(n - alpha - beta + 1) + alpha
#where i is the floor and g the fractional part of the index q
#the default is the linear method which uses alpha = beta = 1
#Parameters:
# - dataArray: Sorted data array
#Return: Q1 and Q3
def calculateQuartilesNumpy(dataArray):

    Q1 = np.quantile(dataArray, q = 0.25)
    Q3 = np.quantile(dataArray, q = 0.75)

    return (Q1, Q3)

#Calculates quantiles using the ceiling of the 0.25*n th index and
#0.75*n th
#Parameters:
# - dataArray: Sorted data array
#Return: Q1 and Q3
def calculateQuartilesManually(dataArray):

    n = len(dataArray)

    Q1 = dataArray[mt.ceil(0.25*n)]
    Q3 = dataArray[mt.ceil(0.75*n)]

    return (Q1, Q3)

#Calculates the variance of the data using numpy.var
#using x's mean and 
def calculateVariance(dataArray):

    return np.var(dataArray)

#Calculates the standard deviation using the square root
#of the variance
#Parameters:
# - variance: N/A
#Return: square root of variance
def calculateStandardDeviation(variance):
    return np.sqrt(variance)

#Calculates the variance coefficient using
#standard deviation and average by dividing them in said respective order
#and multiplying it by 100
#Parameters:
# - standard deviation: N/A
# - average: N/A
#Return: result of substraction
def calculateVarianceCoeficient(standardDeviation, average):
    return (standardDeviation/average)*100

#Calculates the sample range substracting the max value in the data
#from the min value
#Parameters:
# - dataArraySorted: Sorted values in ascending order
#Return: result of substraction
def calculateSampleRange(dataArraySorted):
    return np.max(dataArraySorted) - np.min(dataArraySorted)

#Calculates the quantile range by substracting Q1 from Q3
#Parameters:
# - Q1: quantile 1
# - Q3: quantile 3
#Return: result of substraction
def calculateQuantileRange(Q1, Q3):
    return Q3 - Q1

def calculateMode(dataArray):

    return stats.mode(dataArray)

def printValues(valuesObtained):
    print("Datos has the following values\n"
          + "Average: " + str(valuesObtained['average']) + "\n"
          + "Median: " + str(valuesObtained['median']) + "\n"
          + "Mode: " + str(valuesObtained['mode']) + "\n"
          + "Q1: " + str(valuesObtained['quantiles'][0]) + "\n"
          + "Q3: " + str(valuesObtained['quantiles'][1]) + "\n"
          + "Variance: " + str(valuesObtained['variance']) + "\n"
          + "Standard Deviation: " + str(valuesObtained['standardDeviation']) + "\n"
          + "Variance Coeficient: " + str(valuesObtained['varianceCoeficient']) + "\n"
          + "Sample Range: " + str(valuesObtained['sampleRange']) + "\n"
          + "Quantile Range: " + str(valuesObtained['quantileRange']) + "\n")
    
#Builds the histogram using humidity data found in dataArray
#The number of clases or bars are calculated by numOfClases
#By calculating the square root of the total number of data
def makeHistogram(dataArray, sampleRange):

    numOfClases = (mt.ceil(np.sqrt(len(dataArray))))

    plt.figure(figsize=(20,10))
    
    counts, bins, patches = plt.hist(dataArray,bins=numOfClases, edgecolor="black", rwidth=0.9)

    style.use('bmh')
    plt.xlabel("Porcentaje de Humedad")
    plt.ylabel("Frecuencia de mediciones")
    plt.title("Histograma")
    plt.xticks(ticks=bins)
    plt.show(block=False)

    
#Builds a box plot using humidity data
#We used this function to found outlier data
#Sort the values in ascending order
#Calculates Q1,Q2,Q3 and the limits of the boxplot the same way we calculate this values   in class
def makeBoxplot(dataArray):
    plt.figure(figsize=(20,10))
    
    boxplot = plt.boxplot(dataArray, vert=True, showmeans=True)

    caps = boxplot['caps']
    med = boxplot['medians'][0]

    capbottom = caps[0].get_ydata()[0]
    captop = caps[1].get_ydata()[0]

    median = med.get_ydata()[1]

    pc25 = boxplot['boxes'][0].get_ydata().min()
    pc75 = boxplot['boxes'][0].get_ydata().max()

    xpos = med.get_xdata()
    xoff = 0.10 * (xpos[1] - xpos[0])
    xlabel = xpos[1] + xoff

    plt.text(xlabel, median,
            'Median = {:6.3g}'.format(median), va='center')
    plt.text(xlabel, pc25,
            '25th percentile = {:6.3g}'.format(pc25), va='center')
    plt.text(xlabel, pc75,
            '75th percentile = {:6.3g}'.format(pc75), va='center')
    plt.text(xlabel, capbottom,
            'Bottom cap = {:6.3g}'.format(capbottom), va='center')
    plt.text(xlabel, captop,
            'Top cap = {:6.3g}'.format(captop), va='center')
    
    plt.ylabel("Porcentaje de Humedad")
    plt.title("Diagrama de Cajas")

    style.use('bmh')
    plt.show()

#This function generates the statistics for a sample of data
#Parameters:
# - printResults: bool indicating whether to print results to terminal
#Return: Returns a dictionary of the statistics for the sample data
#possible values for the dictionary are:
# average, median, quantiles (list of 2 values), variance, standardDeviation,
# varianceCoefficient, sampleRange, quantileRange 
def calculateStatistics(printResults):

    datos = loadData()
    
    dateArray = []
    dataArray = []

    valuesObtained = dict()

    for line in datos:
        dateArray.insert(0, line[0])
        dataArray.insert(0, line[1])

    dataArraySorted = np.sort(dataArray)
    
    #Calcular promedio
    valuesObtained['average'] = calculateAverage(dataArray)

    #Calcular mediana
    valuesObtained['median'] = calculateMedian(dataArraySorted)

    #Calcular moda
    valuesObtained['mode'] = calculateMode(dataArraySorted)

    #Calcular Quartiles
    valuesObtained['quantiles'] = calculateQuartiles(dataArraySorted, "manual")

    #Calcular varianza muestral
    valuesObtained['variance'] = calculateVariance(dataArraySorted)

    #Calcular desviación estándar
    valuesObtained['standardDeviation'] = calculateStandardDeviation(valuesObtained['variance'])

    #Calcular coeficiente de variación
    valuesObtained['varianceCoeficient'] = calculateVarianceCoeficient(valuesObtained['standardDeviation'],
                                                                        valuesObtained['average'])

    #Calcular rango muestral
    valuesObtained['sampleRange'] = calculateSampleRange(dataArraySorted)

    #Calcular rango interquartil
    valuesObtained['quantileRange'] = calculateQuantileRange(valuesObtained['quantiles'][0],
                                                              valuesObtained['quantiles'][1])

    if printResults:
        printValues(valuesObtained)

    #Realizar histograma
    makeHistogram(dataArraySorted, valuesObtained['sampleRange'])

    #Realizar diagrama de Cajas
    makeBoxplot(dataArraySorted)

    return valuesObtained


    

calculateStatistics(True)