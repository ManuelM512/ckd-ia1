# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler  # , OneHotEncoder

# %%
file_path = "chronic_kidney_disease_cleaned.csv"
df = pd.read_csv(file_path)

df.head()

# %% [markdown]
# En este caso de estudio, replicaremos los pasos que seguimos en el proyecto principal con Rapidminner
#
# Comenzaremos con el análisis de los datos.
# Tipo de datos en el dataset
# %%
df.dtypes
# Mostramos la distribución de las variables númericas
df.describe()

# %% [markdown]
# Seguimos con realizar la matriz de correlación de las variables numéricas
# %%
not_numericals = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    "class",
]
numericals = list(set(df.columns.to_list()) - set(not_numericals))
# %%
correlation_matrix = df[numericals].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# %%
# Aquí podremos ver mejor la relación entre las variables
sns.pairplot(df, diag_kind="kde", corner=True)
plt.show()

# %%
# Buscamos por los distintos outliers posibles
plt.figure(figsize=(15, len(numericals) * 4))
for i, column in enumerate(numericals, 1):
    plt.subplot(len(numericals), 1, i)
    sns.boxplot(data=df, x=column)
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()
plt.show()

# %%
# Identify numerical and categorical columns
numerical_columns = numericals
categorical_columns = df.select_dtypes(exclude=[np.number]).columns

# Impute missing values for numerical columns using the median
numerical_imputer = SimpleImputer(strategy="median")
df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

# Impute missing values for categorical columns using the mode
categorical_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# %%
# Replicamos las transformaciones para poder
df["ln"] = np.log(df["bu"])
df["ln"] = np.log(df["bgr"])
df["hemo"] = df["hemo"] ** 2
df["pcv"] = df["pcv"] ** 2
df["age"] = df["age"] ** 2

df_transformed = df

df_transformed.head()

# %% [markdown]
## Normalizamos la data

# %%
scaler = MinMaxScaler()

df_transformed[numericals] = scaler.fit_transform(df_transformed[numericals])

# Visualizamos las filas después de la transformación
df_transformed.head()

# %%
X = df_transformed[numericals]
y = df_transformed["class"]

selector = SelectKBest(score_func=f_classif, k="all")
X_selected = selector.fit_transform(X, y)

feature_scores = pd.DataFrame(
    {"Feature": X.columns, "Score": selector.scores_}
).sort_values(by="Score", ascending=False)
feature_scores

# %%
k = 5
top_k_features = feature_scores["Feature"].iloc[:k].values
X_top_k = df_transformed[top_k_features]
top_k_features
