import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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