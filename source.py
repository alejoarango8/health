import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

import sqlite3
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import  SelectKBest, chi2, mutual_info_classif, RFE, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVR
from collections import Counter

def heatmap(df=..., annot=False):
    """
    Funcion para el mapa de correlacion de las variables numericas de un Data Frame
    devuele una figura de matplotlib

    Parametros
    ------------
        df: DataFrame con la informacion a dibujar
        annot: Mostrar valores en el grafico

    """
    # Asegurarse de que df sea un DataFrame de pandas
    assert isinstance(df, pd.DataFrame), "df debe ser un DataFrame"

    # Calcular la matriz de correlación para las características numéricas, excluyendo la columna "NRODOC"
    corr = df.drop("NRODOC", axis=1)._get_numeric_data().corr().round(2)

    # Visualizar la matriz de correlación utilizando un mapa de calor de seaborn
    sns.heatmap(corr.abs(), annot=annot);


def scatterplots(feats, df=..., nrows=4, ncols=3):


    # Asegurarse de que df sea un DataFrame de pandas
    assert isinstance(df, pd.DataFrame), "df debe ser un DataFrame"

    # Crear una figura y un conjunto de ejes para los mapas de dispersión
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*((1+np.sqrt(5))/2), 7))
    i = 0
    j = 0

    # Iterar sobre todas las combinaciones de características para generar los mapas de dispersión
    for x, y in itertools.combinations(feats, r=2):

        # Verificar si hay una sola fila en los subplots
        if nrows == 1:
            df[[x, y]].plot.scatter(x=x, y=y, ax=axes[j], xlabel=f'{x}')
        else:
            df[[x, y]].plot.scatter(x=x, y=y, ax=axes[i,j], xlabel=f'{x}')
        
        # Actualizar índices para los subplots
        if j != 0 and (j + 1) % 3 == 0:
            j = 0
            i += 1
        else:
            j += 1

    # Agregar título a la figura
    fig.suptitle("Mapas de dispersión para variables de alta correlación")

    # Ajustar el diseño de la figura para una mejor visualización
    plt.tight_layout()


def fit_model(df, vars=None, clf="rf"):
    """
    Funcion para el entrenamiento de los modelos
    
    Parametros:
    --------------
        vars: Variables a utilizar para el modelo
        clf: Tipo de algoritmo a utilizar. ("rf", "gb", "svr")
        
    """   
    
    # Copiar el DataFrame para evitar modificar el original
    df = df.copy()  

    # Definir la variable objetivo
    target = "LOS"

    # Identificar las columnas con Fechas o informacion que no es util
    to_drop = df.columns[df.columns.str.contains("(^F[E-e]\w{3})")].values.tolist() + ["NRODOC", target, "Atencion", "Ingreso"]

    # Seleccionar las variables predictoras
    if vars:
        X = df[vars]
    else:
        X = df.drop(to_drop, axis=1)

    # Convertir la variable objetivo a horas
    y = df[target].dt.total_seconds() / 3600

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar el preprocesamiento de variables categóricas y numéricas
    if clf == "svr":
        cat_processor = OneHotEncoder()
    else:
        cat_processor = OrdinalEncoder()
   
   
    # Preprocesador de variables  numericas
    num_processor = StandardScaler()

    # DataFrame de variables numericas  
    num_vals = X._get_numeric_data().columns.tolist()
    # DataFrame de variables categoricas
    cat_vals = X.select_dtypes("object").columns.tolist()
      

    # Combinar preprocesadores de variables
    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

    # Seleccionar el modelo según el clasificador especificado
    if clf == "svr":
        model = make_pipeline(processor, SVR())
    elif clf == "rf":
        model = make_pipeline(processor, RandomForestRegressor(random_state=42))
    elif clf == "gb":
        model = make_pipeline(processor, GradientBoostingRegressor(random_state=42))

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Verificar si existe el directorio 'models', si no, crearlo
    if os.path.exists('models'):
        pass
    else:
        os.makedirs("models")

    # Retornar el modelo entrenado
    return model


def feature_sel(df, num_feat_kbest=20, num_rfe=15, sample=None, plot_metric="mutual_info", plot=False):
    """
    Funcion para la seleccion de las variables a utilizar

    Parametros:
    --------------
        num_feat_kbest: Variables a utilizar para de la seleccion por Kbest
        num_rfe: Variables a tener en cuenta con el metodo RFE
        plot_metric: Metrica por la cual se ordernaran los graficos
        plot: True si se quiere visualizar los resultados obtenidos
        
    """
    
    # DataFrame para las variables
    df_vars = df.copy()

    # Variable objetivo
    target = "LOS"

    # Columnas a no tener en cuenta para valores X
    to_drop = df_vars.columns[df.columns.str.contains("(^F[E-e]\w{3})")].values.tolist() + ["NRODOC", target, "Atencion", "Ingreso"]

    # DataFrame de variables
    X = df_vars.drop(to_drop, axis=1)

    # Vector de variable objetivo convertido a horas
    y = df_vars[target].dt.total_seconds()/3600

    # Division de datos de evaluacion y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Preprocesador de variables categoricas
    cat_processor = OrdinalEncoder()
    # Preprocesador de variables numericas
    num_processor = MinMaxScaler()

    # DataFrame de variables numericas
    num_vals = X._get_numeric_data().columns.tolist()
    # DataFrame de variables categoricas
    cat_vals = X.select_dtypes("object").columns.tolist()

    # Union de procesadores
    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

    # Dicionarion para guardar resultados del algoritmo Kbest (una columna por metrica)
    vars = {"f_regression":[], "mutual_info":[]}

    # Evaluacion de algoritmo Kbest con diferentes metricas y diferentes k
    for m in (f_regression, mutual_info_regression):
      for k in range(5, 25):

        # Pipeline con el selector y preprocesador 
        selector = make_pipeline(processor,
                              SelectKBest(m, k=k))

        # Entrenar selector
        selector.fit(X_train, y_train)

        # Asignar lista con variables escogidas con cada kbest al valor correspondiente en el diccionario
        if m == f_regression:
          # Asignar listas a la llave f_regression
          vars["f_regression"] += selector.get_feature_names_out().tolist()
        else:
          # Asignar listas a la llave mutual_info
          vars["mutual_info"] += selector.get_feature_names_out().tolist()

    # Crear data frame con los index como las variables y los valores de veces escogidas por cada metrica
    vars_kb = pd.DataFrame({i:pd.Series(j).value_counts() for i,j in vars.items()})
    vars_kb.index = vars_kb.index.str[5:]

    # Plot the DataFrame con variables y las veces escogidas
    if plot:
        # Ordenar DataFrame segun metrica escogida
        vars_kb.sort_values(by=plot_metric, ascending=True).plot(kind="barh")
        # Ubicar legenda
        plt.legend(loc=[0.7, 0.2])
        plt.title("Feature Importance (f_regression vs mutual information)")
        plt.show()

    # Listas con metricas
    metric = ["f_regression", "mutual_info"]
    criterion = ["squared_error", "absolute_error"]

    # DataFrame para guardar resultados de RFE
    vars_rfe = pd.DataFrame()

    # Recorrer cada metrica
    for m in metric:
      
      # Variable objetivo
      target = "LOS"
      # Tomar las variables seleccionadas para el Kbest
      X = df_vars[vars_kb[m].sort_values(ascending=False).index.values.tolist()[:num_feat_kbest]]

      # Vector de variable objetivo convertido a horas
      y = df_vars[target].dt.total_seconds()/3600
      # Division de datos de evaluacion y entrenamiento
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
      # Preprocesador de variables categoricas
      cat_processor = OrdinalEncoder()
      # Preprocesador de variables numericas
      num_processor = MinMaxScaler()

      # DataFrame de variables numericas  
      num_vals = X._get_numeric_data().columns.tolist()
      # DataFrame de variables categoricas
      cat_vals = X.select_dtypes("object").columns.tolist()
      
      # Union de procesadores
      processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

      for c in criterion:

        # Pipeline con el selector y preprocesador 
        selector = make_pipeline(processor,
                                RFE(DecisionTreeRegressor(criterion=c, random_state=42), n_features_to_select=num_rfe))

        # Entrenar selector
        selector.fit(X_train, y_train)

        # Guardar las variables seleccionadas con cada combianacion de criterios en las columnas    
        vars_rfe[c+f"_{m}"] = X.columns[selector.named_steps["rfe"].support_].values

    # Instanciar Counter
    counter = Counter()

    # Contar las variables mas repetidas en todos los criterios
    for i in vars_rfe.columns:
      counter.update(vars_rfe[i])

    # DataFrame con los nombres de las variables y las veces que aparecieron en cada criterio
    n_select = pd.DataFrame(counter.values(), index=counter.keys()).rename(columns={0:"count"}).sort_values(by="count", ascending=True)

    
    if plot:

        # Diagrama de barras de las variables mas recurrentes en cada criterio    
        n_select.plot(kind="barh", title="Numero de apariciones en los criterios", legend=False)
        plt.show() 

    # Diccionario con los nombres de combinaciones de criterios como columnas
    info = {i:[] for i in vars_rfe.columns}
    # Algoritmos a evaluar
    clfs = ["rf", "gb", "svr"]


    # DataFrame a usar en el entrenamiento de los modeloss
    df_train = df_vars.copy()

    # Iterar sobre cada combinación de variables seleccionadas y clasificadores
    for v in vars_rfe:
        for clf in clfs:
            # Extraer variables seleccionadas y variable objetivo
            vars = vars_rfe[v].values.tolist()
            target = "LOS"
            
            # Extraer variables en X y variable objetivo en horas
            X = df_train[vars]
            y = df_train[target].dt.total_seconds() / 3600
            
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar el modelo y hacer predicciones
            model = fit_model(df_train, vars=vars, clf=clf, sample=sample)
            pred = model.predict(X_test)
            
            # Calcular el error absoluto medio y agregarlo al diccionario de info
            info[v].append(mean_absolute_error(y_test, pred))

    # Diccionario para almacenar el error absoluto medio para la combinación de variables mas recurrentes
    _ = {"recurrente": []}

    # Iterar sobre cada clasificador para la combinación de variables  mas presentes en todos los criterios
    for clf in clfs:
        # Extraer variables seleccionadas
        vars = n_select.index.values.tolist()
        
        # Extraer variables en X y variable objetivo en horas   
        X = df_train[vars]
        y = df_train[target].dt.total_seconds() / 3600
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar el modelo y hacer predicciones
        model = fit_model(df_train, vars=vars, clf=clf, sample=sample)
        pred = model.predict(X_test)
        
        # Calcular el error absoluto medio y agregarlo al diccionario de info
        _["recurrente"].append(mean_absolute_error(y_test, pred))

    # Actualizar el diccionario de info con el error absoluto medio para la combinación 'recurrente'
    info.update(_)

    # Diccionario para almacenar el error absoluto medio usando todas las varaibles
    every = {"all_feat": []}

    # Iterar sobre cada clasificador usando todas las varaibles elegidas antes de algoritmos de seleccion
    for clf in clfs:

        # Extraer variables seleccionadas
        vars = n_select.index.values.tolist()
        
        # Extraer variables en X y variable objetivo en horas
        X = df_train.drop([target, "Atencion", "Ingreso"], axis=1)
        y = df_train[target].dt.total_seconds() / 3600
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar el modelo y hacer predicciones
        model = fit_model(df_train, vars=vars, clf=clf, sample=sample)
        pred = model.predict(X_test)
        
        # Calcular el error absoluto medio y agregarlo al diccionario 'every'
        every["all_feat"].append(mean_absolute_error(y_test, pred))

    # Actualizar el diccionario de info con el error absoluto medio para la combinación 'all_feat'
    info.update(every)

    # Crear DataFrame con la información del error absoluto medio
    eval_df = pd.DataFrame(info, index=[clfs])

    # Verificar si el directorio 'features' existe, si no, crearlo
    if os.path.exists('features'):
        pass
    else:
        os.mkdir("features")

    # Lista de DataFrames para almacenar
    to_store = [vars_kb, vars_rfe, n_select, eval_df]

    # Iterar sobre los DataFrame para almacenar y guardarlos como archivos pickle
    for i in to_store:
        for name, value in locals().items():
            if i is value:
                if name == "i":
                    pass
                else:
                    with open(f"features/{name}.pkl", "wb") as f:
                        pickle.dump(i, f)


def hyper_tunning(df):
    
    """
    Funcion para el afinamiento de los hyperparametros del modelo seleccionado
    
    Parametros:
    --------------
        df: DataFrame a entrenar
    """  

    # Cargar el archivo que contiene las variables seleccionadas por RFE
    with open("features/vars_rfe.pkl", "rb") as f:
        vars_rfe = pickle.load(f)

    # Definir el grid de parámetros para la búsqueda de hiperparámetros
    param_grid = {
        "randomforestregressor__n_estimators": range(25, 100, 25),
        "randomforestregressor__criterion": ["squared_error", "absolute_error", "friedman_mse"],
        "randomforestregressor__max_features": ["sqrt", "log2", None]
    }

    # Seleccionar las variables a utilizar 
    features = vars_rfe["absolute_error_mutual_info"].values.tolist()

    # Copiar el DataFrame para evitar modificar el original
    df = df.copy()

    # Definir el número de divisiones para la validación cruzada
    cv = 7

    # Definir la variable objetivo
    target = "LOS"

    # Extraer variables en X y variable objetivo en horas
    X = df.drop([target, "Atencion", "Ingreso"], axis=1)
    y = df[target].dt.total_seconds() / 3600

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar preprocesamiento de variables categóricas y numéricas
    cat_processor = OrdinalEncoder()
    num_processor = StandardScaler()

    # DataFrame de variables numericas  
    num_vals = X._get_numeric_data().columns.tolist()
    # DataFrame de variables categoricas
    cat_vals = X.select_dtypes("object").columns.tolist()

    # Combinar preprocesadores de variables
    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

    # Configurar el modelo como un pipeline con el preprocesador y el RandomForestRegressor
    clf = make_pipeline(processor, RandomForestRegressor(random_state=42))

    # Configurar la búsqueda de hiperparámetros utilizando GridSearchCV
    model_tofit = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)

    # Entrenar el modelo utilizando la búsqueda de hiperparámetros
    model_tofit.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor modelo
    best_params = {i[len("randomforestregressor__"):]: j for i, j in model_tofit.best_params_.items()}
    best_model = model_tofit.best_estimator_

    # Verificar si existe el directorio 'models/best_model', si no, crearlo
    if os.path.exists('models/best_model'):
        pass
    else:
        os.makedirs("models/best_model")

    # Lista de objetos para almacenar
    to_store = [features, best_params, df, best_model]

    # Iterar sobre los objetos para almacenar y guardarlos como archivos pickle
    for i in to_store:
        for name, value in locals().items():
            if i is value:
                if name == "i":
                    pass
                else:
                    with open(f"models/best_model/{name}.pkl", "wb") as f:
                        pickle.dump(i, f)