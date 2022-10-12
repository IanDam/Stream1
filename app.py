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
st.image("./Images/visu1.png",caption="Poporciones de la variable ph, y de potability en estas")
st.write("Para el caso de la variable solids, la proporción de muestras no potables es mayor en casí todos los intervalos, sin embargo en los intervalos más altos las muestras potables son mayores que las no potables . Por otra parte la mayor cantidad de datos se agrupa entre los valores de 10000 y 30000")

st.image("./Images/visu2.png", caption="Poporciones de la variable solids, y de potability en estas")
st.write("Con la variable Hardness sucede algo similar a lo visto con la variable solids, las muestras no potables son mayores a las potables en casi todos los intervalos, sin embargo en los intervalos más bajos y en los más altos en algunos puntos hay más muestra potables quen no potables. Por otra parte la mayor cantidad de datos se agrupa entre 175 y 225")
st.image("./Images/visu3.png", caption="Poporciones de la variable hardness, y de potability en estas")
st.write("Al igual que con la variable anteriror, en Chloramines las muestras no potables son mayores a las potables en casi todos los intervalos, a excepción de algunos puntos en los intervalos más bajos y en los más altos donde hay más muestra potables que no potables. Adicionalmente entre 6 y 8 se agrupa la mayor cantidad de datos")
st.image("./Images/visu4.png",caption="Poporciones de la variable Chloramiines, y de potability en estas")
st.write("Al igual que con la variable anteriror, en sulfate las muestras no potables son mayores a las potables en casi todos los intervalos, a excepción de algunos puntos en los intervalos más bajos y en los más altos donde hay más muestra potables que no potables, sin embargo estos puntos son mucho más marcados en comparación con las variables anteriores.Por otra parte, aproximadamente entre 325 y 375 se sagrupa la mayor cantidad de datos.")
st.image("./Images/visu5.png",caption="Poporciones de la variable sulfure, y de potability en estas")
st.write("Conductivitysigue la misma tendencia de las variables anteriores de predominio de las muestras no potables en practicamente todos los intervalos a excepciond de pequeños puntos en los intervalos más bajos y más altos , sin embargo, para este caso, los puntos donde resaltan las muestras potables son más pequeños. Por otra parte, aporximadamente entre 325 y 500 se agrupa la mayor cantidad de datos")
st.image("./Images/visu6.png",caption="Poporciones de la variable conductivity, y de potability en estas")
st.write("Para el caso de Organic_Carbon , a excepción de un punto bastante marcado entre 5 y 7.5 donde las muestras potables son mayores a las potables, las muestras no potables son mayores que las potables.Adicoonalmente, entre el 12 y 17.5 aproximandemante se encuentra la mayor cantidad de datos")
st.image("./Images/visu.png7.png",caption="Poporciones de la variable organic_carbon, y de potability en estas")
st.write("Dentro de la variable Trihalomethanes, resalta un punto entre 60 y 80 en el que el conteo de datos es mayor en comparación con los demas puntos. Adiconalmente dentro de esta variable sigue predominando las muestras no potables.")
st.image("./Images/visu8.png",caption="Poporciones de la variable trihalomethanes, y de potability en estas")
st.write("Por ultimo en Turbidity , las muestras no potables resaltan más que las potables en todos los intervalos. Adiconalmente en 3.5 y 4 se agrupa la mayor cantidad de datos.")
st.image("./Images/visu9.png",caption="Poporciones de la variable turbidity, y de potability en estas")
st.write("Con respecto a las proporciones de la variable objetivo, encontramos que existen más muestras no potables que potables.")
st.image("./Images/propo1.png",caption="Proporcion de la variable potability")
st.subheader("Relacion entre las variables:")
st.write("Para la relación entre variables se realizo un heatmmap y un correlograma dentro de los que se observó es practicamente nula la correlación que existe entre las variables. (colores muy oscuros en el heatmap y puntos sin relación en el correlograma)")
st.image("./Images/rela1.png")
st.image("./Images/pair1.png")
st.write("Para mostrar la relación entre dos distribucciones realizamos boxplots, todos respecto a la variable objeto.")
st.image("./Images/box1.png",caption="Boxlpot de la variable ph" )
st.image("./Images/box2.png",caption="Boxlpot de la variable hardness" )
st.image("./Images/box3.png",caption="Boxlpot de la variable solids" )
st.image("./Images/box4.png",caption="Boxlpot de la variable chloramines" )
st.image("./Images/box5.png",caption="Boxlpot de la variable sulfate" )
st.image("./Images/box6.png",caption="Boxlpot de la variable conductivity" )
st.image("./Images/box7.png",caption="Boxlpot de la variable organic_carbon" )
st.image("./Images/box8.png",caption="Boxlpot de la variable trihalomethanes" )
st.image("./Images/box9.png",caption="Boxlpot de la variable turbidity" )
st.subheader("Pruebas de Hipotesis:")
st.write("Se realiza un datasat sin la variable objetivo para realizar la pruab de distribucción normal multivariada")
st.write("Prueba de Hipotesis de normalidad multivariada")
st.write("Ho: El dataset sigue una distribucción normal multivarada")
st.write("Ha: El dataset NO sigue una distribucción normal multivariada.")
st.write("α=0.05")
st.code("""multivariate_normality(data2, alpha = .05 )""",language="python")
st.code("""HZResults(hz=1.1369243176166905, pval=4.509462445228134e-66, normal=False)""",language="python")
st.write("Para la prueba de distribución normal multivariada del dataset sin la variable objetivo, se tiene como resultado un pvalor inferior a 0.05, por lo tanto se asume que el dataset sin la variable objetivo no sigue una distribución normal")
st.write("Para la prueba de distribucción normal, se importa el modulo stats de la libreria scipy y se realiza un normaltest para cada una de las variables, en la cual ya esta incorporada la prueba hipotesis y da como resultado")
st.code("""Columna ph no sigue una distribución normal
Columna Hardness no sigue una distribución normal
Columna Solids no sigue una distribución normal
Columna Chloramines si sigue una distribución normal
Columna Sulfate no sigue una distribución normal
Columna Conductivity no sigue una distribución normal
Columna Organic_carbon si sigue una distribución normal
Columna Trihalomethanes si sigue una distribución normal
Columna Turbidity si sigue una distribución normal""",language="python")
st.write("Se concluye que las variables ph, solids,sulfate,conductivity, turbidity NO siguen una distribucción normal. Mientras que hardness, chloramines,organic carbon, trihalomethanes SI siguen una distribución normal.")

st.write("Se efectua la división del data set para la realización de las diferencias de medias y medianas.")
st.write("Prueba de Hipotesis de diferencia de Medias")
st.write("Ho: Las diferencias de medias es igual")
st.write("Ha: La diferencia de medias es diferente")
st.write("α=0.05")

st.code("""" Ttest_indResult(statistic=-0.3046442883875845, pvalue=0.7606581670556812) """,language="python")
st.code("""" Ttest_indResult(statistic=-0.8647102836072975, pvalue=0.38726679870076297) """,language="python")
st.code("""" Ttest_indResult(statistic=0.595206325721168, pvalue=0.5517503508954078)""",language="python")
st.code("""" Ttest_indResult(statistic=1.343448411247364, pvalue=0.17922819297971626)""",language="python")
st.code("""" Ttest_indResult(statistic=0.09646877342917845, pvalue=0.9231547020388317)""",language="python")
st.code("""" Ttest_indResult(statistic=-0.1363821049180547, pvalue=0.8915283492717541)""",language="python")
st.code("""" Ttest_indResult(statistic=-1.007730107311136, pvalue=0.31366512921636786)""",language="python")
st.code("""" Ttest_indResult(statistic=0.9463387265442308, pvalue=0.344051884391254)""",language="python")
st.code("""" Ttest_indResult(statistic=0.35868790729516203, pvalue=0.7198537565149352)""",language="python")


st.write("A partir de todos los ttest realizados con cada variable, el p-valor de cada uno arroja ser mayor a 0,05 motivo por el cual afirmamos que no hay suficiente evidencia estadistica para rechazar la hipotesis nula por lo tanto se concluye que todas las diferencias de medias de las variables son iguales entre ellas.")

st.code(""" (0.19004781886397173, 0.6628768212772176, 7.057552645613959, 
        array([[568, 936], [581, 924]]))""",language="python")
st.code(""" (0.0036865865680039776, 0.9515843796069338, 196.9823785600006,
 array([[573, 931], [576, 929]]))""",language="python")
st.code(""" (0.007987662311864645, 0.9287849061198299, 20743.348404224074,
 array([[576, 928], [573, 932]]))""",language="python")
st.code(""" (3.0289037993276766, 0.0817932600039606, 7.116809283709961,
 array([[598, 906], [551, 954]]))""",language="python")
st.code(""" (0.01843152769512674, 0.8920088296205512, 334.0274201302968,
 array([[572, 932], [577, 928]]))""",language="python")
st.code(""" (0.01843152769512674, 0.8920088296205512, 420.5492185259853,
 array([[572, 932], [577, 928]]))""",language="python")
st.code(""" (0.6580003678108246, 0.41726677745854024, 14.243373528509892,
 array([[563, 941], [586, 919]]))""",language="python")
st.code(""" (1.4763614453103326, 0.2243446768114634, 66.50978010824268,
 array([[591, 913], [558, 947]]))""",language="python")
st.code(""" (0.09891667092292114, 0.7531335646890226, 3.955916726029925,
 array([[579, 925], [570, 935]]))""",language="python")

st.write("A partir de todos los mediantest realizados con cada variable, el p-valor de cada uno arroja ser mayor a 0,05 motivo por el cual afirmamos que no hay suficiente evidencia estadistica para rechazar la hipotesis nula por lo tanto se concluye que todas las diferencias de medianas de las variables son iguales entre ellas.")

st.subheader("Conclusiones generales:")
st.write("Para este dataset podemos concluir a traves de la visualización de los datos que existe una proporción mayor entre todas las variables y la caracteristica No Potable de nuestra variable objetivo. Así mismo se observo que no existe alguna relación entre las variables independientes. Por ultimo a partir de las pruebas de hipotesis pudimos concluir que las diferencias de medias y de medianas es igual entre cada variable despues de haber realizado una division del dataset a partir de las carcteristicas de la variable objetivo.")

st.write("1. Una categorización de variables y conclusión acerca de la relación de estas con la variable objetivo")
st.write("2. Visualización de proporciones de las variables, relacion entre las variables sin la variable objetivo, Boxplot entre las variables y la vraiable objetivo")
st.write("3. Pruebas de hipotesis sobre la normalidad de las variables, diferencia de medias y median si se divide el dataset en 2, distribución normal multivariada del dataset sin la variable objetivo")
