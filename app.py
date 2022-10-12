import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/labeconometria/MLxE/main/proyectos1er/dataset_2.csv")

st.title("Proyecto 8")
st.write("En la presente página se realizará el análisis de un dataset que busca explicar qué tipo de cliente se tiene, si uno existente o uno retirado, a partir de datos como la edad, nivel de educación, ingresos,entre otras. Para la cual se realiza lo siguiente:")
st.dataframe(df)
st.caption("Dataframe interactiva")
st.subheader("Descripción suncita de las variables")
st.write(""" *   Attrition Flag "Desgaste(Cliente)" , es nuestra variable objetivo, de tipo category donde se tienen Existing Customer "Cliente Existente" y Attrited Customer "Cliente Atraido". """)
st.write(""" *   Customer Age "Edad del Cliente", variable continua donde sus datos van desde 26 hasta 70 años, presenta una media de 46 años.""")
st.write(""" *   Gender "Genero", variable categoria que nos muestra si el cliente es hombre o mujer. """)
st.write(""" *   Dependent Count "Dependencia", variable continua que demuestra el nivel de ***, sus valores van desde 0 a 5, presenta una media de 2 aproximadamente. """)
st.write(""" *   Education Level "Nivel de Educacion", variable categorica que permite conocer el nivel de educacion del cliente, posibles respuestas son escuela secundaria, pregrado, sin educación, desconocido, universidad, posgrado, doctorado. """)
st.write(""" *   Marital Status "Estado Civil", variable de tipo category en donde se permite conocer en cuál de los siguientes estados se encuentra el cliente; Casado, soltero, divorciado , desconocido. """)
st.write(""" *   Income Category "Categoría de Ingresos" , variable de tipo categórico donde se da información sobre el nivel de ingreso del cliente, esta variable presenta los datos a través de 6 rangos de ingreso desde menos de 40K hasta más de 120 K y con desconocido. """)
st.write(""" *   Card Category "Categoría de la tarjeta", variable que nos da conocimiento sobre el tipo de tarjeta que posee el cliente puede ser Azul, Dorada, Plateada y Platino. """)
st.write(""" *   Months on book-Discreta "Meses en el libro", variable discreta donde se da informacion sobre el periodo de tiempo dentro del libro, esta variable tiene datos desde 13 a 56 meses y con una media de 36 meses. """)
st.write(""" *   Credit Limit "Credito Limite", variable de tipo continua donde se da informacion de los limites de credito para cada cliente, la media del credito es de 8606. """)
st.write(""" *   Las demas Variables que tenemos informacion son totales estadisticos de las anteriores mencionadas, poseen datos mas grandes y su informacion estadistica se encuentra en el describe. """)
st.table(df.describe())
st.write("A partir de esto realizamos los graficos de datos relevantes " )
st.write("Cada gráfico está separado a partir de la variable objetivo Attrition Flag")
