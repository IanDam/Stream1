import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/labeconometria/MLxE/main/proyectos2do/datasets3.csv")

st.title("Proyecto 6")
st.write(" En la presente pagina, a partir de un dataset dentro del que se determina la potabilidad del agua a partir de caracteristicas especificas como la cantidad de carbono organico o su condutividad, se realizan lo siguiente: ")
st.subheader("1. Exploracion inicial: ")
st.write("Para la exploración inicial obtenemos información acerca del dataset cargado. En este encontramos 3276 entradas y un total de 10 variables dentro de ella (columnas). Así mismo vemos que de las 10 columnas de datos; 1 es de typo int, 9 de tipo float.Adicionalmente se observa que existen datos nulos en las variables ph, Sulfate y Trihalomethanes.")
st.dataframe(df)
st.caption("Dataframe interactiva: presionando una vez sobre el nombre de las variables se puede ver si existen datos nulos; presionando dos veces sobre los datos los muestra con mayor exactitud")
st.table(df.describe())
st.caption("Tabla con las caracteristicas desciptivas de las diferentes variables")
st.subheader("Limpieza de datos:")
st.write("En primer lugar, estas estadisticas descriptivas son tomadas antes de la eliminación de valores nulos y de los outliers, observandose así que en las variables ph, Solids, sulfate, Conductivity, Organic_carbon, Turbidity y Photability se presenta un sego negativo ya que la mediana es menor que la media, mientras que en Hardnes, Chloramines y Trihalomethanes la media es menor que la mediana, por lo que se presenta un sesgo positivo. ")
st.write("Como observamos en la información del data existen datos nulos que es necesario corregir. Comprobamos así por el metodo isna la cantidad de estos datos y dentro de que variable se encuentra.")

st.table(df.isna().sum())
st.write("A través del método LinearRegresion corregimos la existencia de datos nulos dentro de las variables identificadas ph, Sulfate y Trihalomethanes.")

st.write("Para determinar outliers multivariados se calcula la distancia de mahalanobis, evaluandose los resultados obtenidos en una prueba chi cuadrado, quedandonos con los datos que obtuvieron un p valor mayor a 0.05 , los indexamos dentro del dataframe y eliminamos los datos nulos (datos con un p valor menor a 0.05)")

st.subheader("Transformacion de datos:")
st.code("""data = data.astype({"Potability":"category"})""", language="python")
st.subheader("categorizacion: ")
st.write("Para la categorización, dividimos las variables en 4 intervalos.")
st.code(""" data["Conductivity_2"] = pd.cut(data["Conductivity"],4) 
data["Chloramines_2"] = pd.cut(data["Chloramines"],4) 
data["Hardness_2"] = pd.cut(data["Hardness"],4) 
data["Solids_2"] = pd.cut(data["Solids"],4) 
data["Organic_carbon_2"] = pd.cut(data["Organic_carbon"],4) 
data["Turbidity_2"] = pd.cut(data["Turbidity"],4) 
data["ph_2"] = pd.cut(data["ph"],4) 
data["Sulfate_2"] = pd.cut(data["Sulfate"],4) 
data["Trihalomethanes_2"] = pd.cut(data["Trihalomethanes"],4)""", language="pyhton")
st.write("conclusiones categorizacion")

st.subheader("visualizacion: ")
st.write(" Para la visulaización de la distribución de los datos realizamos graficos de barras divididos en potable y o potable para cada variable")

st.write("Para el caso de la variable ph, encontramos que independientemente del valor, existen más muestras no potables que potables, encontrandose la mayor cantidad de datos en el intervalo entre ph 6 y ph 8 tanto para potables como para no potables")
