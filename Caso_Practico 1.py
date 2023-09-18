# Caso Practico 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_2015 = pd.read_csv("2015.csv")
df_2016 = pd.read_csv("2016.csv")

# Identificando columnas
print("Columnas en el informe de felicidad de 2015:")
print(df_2015.columns)
print("\nColumnas en el informe de felicidad de 2016:")
print(df_2016.columns)
# Respuesta: Si, hay diferencia

# Une ambos df
df_merged = pd.concat([df_2015, df_2016], ignore_index=True)

# Revisa el número de nulos que hay por cada columna, así como su porcentaje
## Número de nulos por columna
nulos_por_columna = df_merged.isnull().sum()

## Porcentaje de nulos por columna
porcentaje_nulos = (nulos_por_columna / len(df_merged)) * 100

## Mostrar los resultados
print("Número de nulos por columna:")
print(nulos_por_columna)
print("\nPorcentaje de nulos por columna:")
print(porcentaje_nulos)

# Cambia los valores nulos de las columnas "Lower Confidence Interval" y "Upper Confidence Interval" 
# por un número aleatorio entre el valor mínimo y máximo de la misma columna
## Reemplazar valores nulos en "Lower Confidence Interval"
min_lower_confidence = df_merged['Lower Confidence Interval'].min()
max_lower_confidence = df_merged['Lower Confidence Interval'].max()
df_merged['Lower Confidence Interval'].fillna(np.random.uniform(min_lower_confidence, max_lower_confidence), inplace=True)

## Reemplazar valores nulos en "Upper Confidence Interval"
min_upper_confidence = df_merged['Upper Confidence Interval'].min()
max_upper_confidence = df_merged['Upper Confidence Interval'].max()
df_merged['Upper Confidence Interval'].fillna(np.random.uniform(min_upper_confidence, max_upper_confidence), inplace=True)

# Cambia los valores nulos de la columna "Standard Error" por su media al cuadrado.
## Calcular la media de la columna "Standard Error"
mean_standard_error = df_merged['Standard Error'].mean()

## Reemplazar valores nulos por la media al cuadrado
df_merged['Standard Error'].fillna(mean_standard_error**2, inplace=True)

# Obtener un resumen estadístico del dataframe sin valores nulos
resumen_estadistico = df_merged.dropna().describe()
print("Resumen estadístico del dataframe sin valores nulos:")
print(resumen_estadistico)

# Mostrar de forma gráfica la relación entre la familia y la salud
plt.scatter(df_merged['Family'], df_merged['Health (Life Expectancy)'])
plt.xlabel('Family')
plt.ylabel('Health (Life Expectancy)')
plt.title('Relación entre Family y Health')
plt.show()

# Muestra de forma gráfica la relación entre la puntuación de felicidad y la confianza (corrupción del gobierno).
plt.scatter(df_merged['Happiness Score'], df_merged['Trust (Government Corruption)'])
plt.xlabel('Happiness Score')
plt.ylabel('Trust (Government Corruption)')
plt.title('Relación entre Happiness Score y Trust')
plt.show()

# Mostrar la matriz de correlación del dataframe
correlation_matrix = df_merged.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Agrupar el dataframe por país con el valor máximo de felicidad, sin importar el año
max_happiness_by_country = df_merged.groupby('Country')['Happiness Score'].max().reset_index()
max_happiness_by_country.columns = ['Country', 'Max Happiness Score']
print(max_happiness_by_country)

# Mostrar la relación entre felicidad, generosidad y puntuación de libertad
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Generosity', y='Happiness Score', hue='Freedom', data=df_merged)
plt.xlabel('Generosity')
plt.ylabel('Happiness Score')
plt.title('Relación entre Felicidad, Generosidad y Libertad')
plt.legend(title='Freedom')
plt.show()

# Mostrar la distribución del grado de distopía en función de la región
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Dystopia Residual', data=df_merged)
plt.xlabel('Region')
plt.ylabel('Dystopia Residual')
plt.title('Distribución del Grado de Distopía por Región')
plt.xticks(rotation=90)
plt.show()