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


    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"

    corr = df.drop("NRODOC", axis=1)._get_numeric_data().corr().round(2)


    sns.heatmap(corr.abs(), annot=annot);

def scatterplots(feats, df=..., nrows=4, ncols=3):


    assert isinstance(df, pd.DataFrame), "df debe ser un Dataframe"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*((1+np.sqrt(5))/2), 7))
    i = 0
    j = 0

    for x, y in itertools.combinations(feats, r=2):

        if nrows == 1:
            df[[x, y]].plot.scatter(x=x, y=y, ax=axes[j], xlabel=f'{x}')
        else:
            df[[x, y]].plot.scatter(x=x, y=y, ax=axes[i,j], xlabel=f'{x}')
        if j!=0 and (j+1)%3 == 0:
            j = 0
            i += 1
        else:
            j += 1

    fig.suptitle("Mapas de dispersion para variables de alta correlacion")
    plt.tight_layout()

def feature_sel(df, num_feat_kbest=20, num_rfe=15, plot_metric="f_regression", plot=False, sample=None):
    """
    Funcion para la seleccion de las variables a utilizar

    Parametros:
    --------------
        num_feat_kbest: Variables a utilizar para de la seleccion por Kbest
        num_rfe: Variables a tener en cuenta con el metodo RFE
        plot_metric: Metrica por la cual se ordernaran los graficos
        plot: True si se quiere visualizar los resultados obtenidos
        year_vars: Año en que se tomaron los datos para entrenamiento de variables
        sample: Usar estrategia de re-sampleo para el entrenamiento
        year_train: Año en que se tomaron los datos para entrenamiento del modelo

    """
    

    df_vars = df.copy()

    target = "LOS"

    to_drop = df_vars.columns[df.columns.str.contains("(^F[E-e]\w{3})")].values.tolist() + ["NRODOC", target, "Atencion", "Ingreso"]

    X = df_vars.drop(to_drop, axis=1)
    y = df_vars[target].dt.total_seconds()/3600

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_processor = OrdinalEncoder()
    num_processor = MinMaxScaler()

    num_vals = X._get_numeric_data().columns.tolist()
    cat_vals = X.select_dtypes("object").columns.tolist()

    processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

    vars = {"f_regression":[], "mutual_info":[]}

    for m in (f_regression, mutual_info_regression):
      for k in range(5, 25):

        selector = make_pipeline(processor,
                              SelectKBest(m, k=k))

        selector.fit(X_train, y_train)
        if m == f_regression:
          vars["f_regression"] += selector.get_feature_names_out().tolist()
        else:
          vars["mutual_info"] += selector.get_feature_names_out().tolist()


    vars_kb = pd.DataFrame({i:pd.Series(j).value_counts() for i,j in vars.items()})
    vars_kb.index = vars_kb.index.str[5:]

    if plot:

        vars_kb.sort_values(by=plot_metric, ascending=True).plot(kind="barh",)
        plt.legend(loc=[0.7, 0.2])
        plt.title("Feature Importance (f_regression vs mutual information)")
        plt.show()


    metric = ["f_regression", "mutual_info"]
    criterion = ["squared_error", "absolute_error"]

    vars_rfe = pd.DataFrame()

    for m in metric:

      target = "LOS"

      X = df_vars[vars_kb[m].sort_values(ascending=False).index.values.tolist()[:num_feat_kbest]]
      y = df_vars[target].dt.total_seconds()/3600

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      cat_processor = OrdinalEncoder()
      num_processor = MinMaxScaler()

      num_vals = X._get_numeric_data().columns.tolist()
      cat_vals = X.select_dtypes("object").columns.tolist()

      processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

      for c in criterion:

        selector = make_pipeline(processor,
                                RFE(DecisionTreeRegressor(criterion=c, random_state=42), n_features_to_select=num_rfe))

        selector.fit(X_train, y_train)

        X.columns[selector.named_steps["rfe"].support_].values.tolist()

        vars_rfe[c+f"_{m}"] = X.columns[selector.named_steps["rfe"].support_].values


    counter = Counter()

    for i in vars_rfe.columns:
      counter.update(vars_rfe[i])


    n_select = pd.DataFrame(counter.values(), index=counter.keys()).rename(columns={0:"count"}).sort_values(by="count", ascending=True)

    if plot:
        
        n_select.plot(kind="barh", title="Numero de apariciones en los criterios", legend=False)
        plt.show()

    info = {i:[] for i in vars_rfe.columns}
    clfs = ["rf", "gb", "svr"]

    df_train = df_vars.copy()

    for v in vars_rfe:
        for clf in clfs:
            vars = vars_rfe[v].values.tolist()
            target = "LOS"
            X = df_train[vars]
            y = df_train[target].dt.total_seconds()/3600
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = fit_model(df_train, vars=vars, clf=clf, sample=sample)
            pred = model.predict(X_test)
            info[v].append(mean_absolute_error(y_test, pred))

    _ = {"recurrente": []}
    for clf in clfs:
        vars = n_select.index.values.tolist()
        X = df_train[vars]
        y = df_train[target].dt.total_seconds()/3600
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = fit_model(df_train, vars=vars, clf=clf, sample=sample)
        pred = model.predict(X_test)
        _["recurrente"].append(mean_absolute_error(y_test, pred))

    info.update(_)

    every = {"all_feat": []}
    for clf in clfs:
        vars = n_select.index.values.tolist()
        X = df_train.drop([target, "Atencion", "Ingreso"], axis=1)
        y = df_train[target].dt.total_seconds()/3600
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = fit_model(df_train, vars=vars, clf=clf, sample=sample)
        pred = model.predict(X_test)
        every["all_feat"].append(mean_absolute_error(y_test, pred))

    info.update(every)

    eval_df = pd.DataFrame(info, index=[clfs])

    if os.path.exists('features'):
        pass
    else:
        os.mkdir("features")

    to_store = [vars_kb, vars_rfe, n_select, eval_df]

    for i in to_store:
      for name,     value in locals().items():
        if i is value:
            if name == "i":
                pass
            else:
                with open(f"features/{name}.pkl", "wb") as f:
                     pickle.dump(i, f)


def fit_model(df, vars=None, year=None, clf="rf", sample=None, save=False, prefix=""):
  """
  Funcion para el entrenamiento de los modelos
  
  Parametros:
  --------------
      vars: Variables a utilizar para el modelo
      year: Año en que se tomaron los datos para entrenamiento
      clf: Tipo de algoritmo a utilizar. ("rf", "gb", "svc")
      sample: Usar estrategia de re-sampleo para el entrenamiento
      save: True si se quiere guardar el modelo como archivo. Default False
      prefix: prefijo a utilizar en el modelo, solo funciona si True es verdadero
      
  """  
  df = df.copy()  

  target = "LOS"

  to_drop = df.columns[df.columns.str.contains("(^F[E-e]\w{3})")].values.tolist() + ["NRODOC", target, "Atencion", "Ingreso"]

  if vars:
    X = df[vars]
  else:
    X = df.drop(to_drop, axis=1)

  y = df[target].dt.total_seconds()/3600


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  if clf == "svr":
    cat_processor = OneHotEncoder()

  else:
    cat_processor = OrdinalEncoder()

  num_processor = StandardScaler()

  num_vals = X._get_numeric_data().columns.tolist()
  cat_vals = X.select_dtypes("object").columns.tolist()


  processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

  if clf == "svr":

    model = make_pipeline(processor,
                      SVR())
  elif clf == "rf":

    model = make_pipeline(processor,
                      RandomForestRegressor(random_state=42))
  elif clf == "gb":

    model = make_pipeline(processor,
                      GradientBoostingRegressor(random_state=42))

  model.fit(X_train, y_train)
  
  if os.path.exists('models'):
        pass
  else:
      os.makedirs(f"models/{prefix}/model_{prefix}")

  to_store = [model]

  for i in to_store:
      for name, value in locals().items():
          if i is value:
              if name == "i":
                  pass
  
              with open(f"models/{prefix}/{name}.pkl", "wb") as f:
                  pickle.dump(i, f)
      

  return model


def hyper_tunning(df):
    
  """
  Funcion para el afinamiento de los hyperparametros del modelo seleccionado
  
  Parametros:
  --------------
      year_train: Año en que se tomaron los datos para entrenamiento del modelo
  """  

  with open("features/vars_rfe.pkl", "rb") as f:
      vars_rfe = pickle.load(f)

  param_grid = {"randomforestregressor__n_estimators":range(25,100,25),
                "randomforestregressor__criterion":["squared_error", "absolute_error", "friedman_mse"],
                "randomforestregressor__max_features":["sqrt", "log2", None]}

  features = vars_rfe["absolute_error_mutual_info"].values.tolist()
  

  df = df.copy()
  cv = 7

  target = "LOS"

  X = df[features]
  y = df[target].dt.total_seconds()/3600

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  cat_processor = OrdinalEncoder()
  num_processor = StandardScaler()

  num_vals = X._get_numeric_data().columns.tolist()
  cat_vals = X.select_dtypes("object").columns.tolist()

  processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

  clf = make_pipeline(processor,
                      RandomForestRegressor(random_state=42))

  model_tofit = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)


  model_tofit.fit(X_train, y_train)


  best_params = {i[len("randomforestregressor__"):]:j for i,j in model_tofit.best_params_.items()}
  best_model = model_tofit.best_estimator_

  if os.path.exists('models/best_model'):
      pass
  else:
      os.makedirs("models/best_model")

  to_store = [features, best_params, df, best_model]

  for i in to_store:
    for name, value in locals().items():
      if i is value:
          if name == "i":
              pass
          else:  
              with open(f"models/best_model/{name}.pkl", "wb") as f:
                  pickle.dump(i, f)


def make_prediction(df=None):
    """
    Funcion para realizar la prediccion con nuevos datos
    
    Parametros:
    ---------------
        df: Data frame con la informacion a predecir
    """

    assert isinstance(df, pd.DataFrame), "Se necesita una base de datos para la prediccion"

    with open("models/best_model/features.pkl", "rb") as f:
        feat = pickle.load(f)

    with open("models/best_model/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    X = df[feat]

    prediction = model.predict(X)

    pred_df = pd.DataFrame(prediction, index=X.index, columns=["Prediccion"])

    pred_df.to_csv("models/best_model/predictions.csv", index=False)

    return prediction

def hyper_tunning(df):
    
  """
  Funcion para el afinamiento de los hyperparametros del modelo seleccionado
  
  Parametros:
  --------------
      year_train: Año en que se tomaron los datos para entrenamiento del modelo
  """  

  with open("features/vars_rfe.pkl", "rb") as f:
      vars_rfe = pickle.load(f)

  param_grid = {"randomforestregressor__n_estimators":range(25,100,25),
                "randomforestregressor__criterion":["squared_error", "absolute_error", "friedman_mse"],
                "randomforestregressor__max_features":["sqrt", "log2", None]}

  features = vars_rfe["absolute_error_mutual_info"].values.tolist()
  

  df = df.copy()
  cv = 7

  target = "LOS"

  X = df[features]
  y = df[target].dt.total_seconds()/3600

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  cat_processor = OrdinalEncoder()
  num_processor = StandardScaler()

  num_vals = X._get_numeric_data().columns.tolist()
  cat_vals = X.select_dtypes("object").columns.tolist()

  processor = ColumnTransformer(transformers=[("cat", cat_processor, cat_vals), ("num", num_processor, num_vals)])

  clf = make_pipeline(processor,
                      RandomForestRegressor(random_state=42))

  model_tofit = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)


  model_tofit.fit(X_train, y_train)


  best_params = {i[len("randomforestregressor__"):]:j for i,j in model_tofit.best_params_.items()}
  best_model = model_tofit.best_estimator_

  if os.path.exists('models/best_model'):
      pass
  else:
      os.makedirs("models/best_model")

  to_store = [features, best_params, df, best_model]

  for i in to_store:
    for name, value in locals().items():
      if i is value:
          if name == "i":
              pass
          else:  
              with open(f"models/best_model/{name}.pkl", "wb") as f:
                  pickle.dump(i, f)
