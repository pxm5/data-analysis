import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import datetime as dt
import click
from rich import print
from rich.console import Console

from analysis import shared


def _lineplot(x, y):
    df = shared.df
    sns.lineplot(data=df, x=x, y=y)
    plt.show()

@click.command()
@click.argument("x")
@click.argument("y")
def lineplot(x, y):
    _lineplot(x, y)

def _scatterplot(x, y):
    df = shared.df
    sns.scatterplot(data=df, x=x, y=y)
    plt.show()

@click.command()
@click.argument("x")
@click.argument("y")
def scatterplot(x, y):
    _scatterplot(x, y)


def _boxplot(x, y):
    df = shared.df
    if y:
        sns.boxplot(df, x=x, y=y)
    else:
        sns.boxplot(df, x=x)
    plt.show()

@click.command()
@click.argument("x")
@click.option("--y", default=None)
def boxplot(x, y):
    _boxplot(x, y)

def _histogram(x, y):
    df = shared.df
    if y:
        sns.histplot(df, x=x, y=y)
    else:
        sns.histplot(df, x=x)
    plt.show()

@click.command()
@click.argument("x")
@click.option("--y", default=None)
def histogram(x, y):
    _histogram(x, y)

def _normalize_values(col, kind):
    df = shared.df
    if kind == "minmax":
        min_val = df[col].min()
        max_val = df[col].max()
        func = lambda x: (x - min_val) / (max_val - min_val)
    elif kind == "zscore":
        mean = df[col].mean()
        std = df[col].std()
        func = lambda x: (x - mean) / std
    elif kind == "robust":
        median = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        func = lambda x: (x - median) / iqr
    df[col] = df[col].apply(func)

@click.command()
@click.argument("col")
@click.option("--kind", type=click.Choice(["minmax", "zscore", "robust"]), default="minmax")
def normalize_values(col, kind):
    _normalize_values(col, kind)

def _fill_null_numeric(col, val, discrete):
    df = shared.df
    if val == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif val == "median":
        df[col] = df[col].fillna(df[col].median())
    elif val == "mode":
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df[col] = df[col].fillna(discrete)

@click.command()
@click.argument("col")
@click.option("--val", type=click.Choice(["mean", "median", "mode"]), default="mean")
@click.option("--discrete", default=0)
def fill_null_numeric(col, val, discrete):
    _fill_null_numeric(col, val, discrete)

def _rename(column, new_name):
    df = shared.df
    df.rename(columns={column: new_name}, inplace=True)

@click.command()
@click.argument("column")
@click.argument("new_name")
def rename(column, new_name):
    _rename(column, new_name)

def _binary_encode(column):
    df = shared.df
    if len(df[column].unique()) != 2:
        click.echo("Column does not have exactly 2 unique values.")
        return
    mapping = list(df[column].unique())
    df[column] = df[column].apply(lambda x: mapping.index(x))
    click.echo(f"Encoded: {mapping}")

@click.command()
@click.argument("column")
def binary_encode(column):
    _binary_encode(column)

def _onehot_encode(column, multilinearity):
    df = shared.df
    index = df.columns.get_loc(column)
    encoded = pd.get_dummies(df[column], drop_first=multilinearity, dtype=int, prefix=column)
    df.drop(column, inplace=True, axis=1)
    for i, col in enumerate(encoded.columns):
        df.insert(index + i, col, encoded[col])

@click.command()
@click.argument("column")
@click.option("--multilinearity", is_flag=True)
def onehot_encode(column, multilinearity):
    _onehot_encode(column, multilinearity)

def _to_numeric(column, dtype):
    df = shared.df
    if not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].apply(lambda x: re.sub(r"[^0-9.]+", "", str(x)))
    df[column] = df[column].astype(dtype)

@click.command()
@click.argument("column")
@click.option("--dtype", default="Int64")
def to_numeric(column, dtype):
    _to_numeric(column, dtype)

def _to_datetime(column, format):
    df = shared.df
    if format:
        df[column] = pd.to_datetime(df[column], format=format)
    else:
        df[column] = pd.to_datetime(df[column], errors="coerce")

@click.command()
@click.argument("column")
@click.option("--format", default=None)
def to_datetime(column, format):
    _to_datetime(column, format)

def _show(col):
    df = shared.df
    console = Console()
    if col is not None:
        console.print(df[col])
        return
    console.print(df)

@click.command()
@click.option("--col", "-c", help="Optional column to show")
def show(col):
    _show(col)

def _save(loc):
    shared.df.to_csv(loc, index=False)

@click.command()
@click.argument("loc")
def save(loc):
    _save(loc)

def _summary(showna):
    df = shared.df
    print(df.describe())


@click.command()
@click.option("--showna", "-n", help="Optional toggle to show null values", default=False, is_flag=True)
def summary():
    _summary()

def _nulls(col):
    df = shared.df
    if col is not None:
        print(df[col].isna().sum())
        return
    print("[bold blue]Null values with respect to columns:[/bold blue]")
    for col in df:
        print(f"{col}: {df[col].isna().sum()}")

@click.command()
@click.option("--col", "-c", default=None)
def nulls(col):
    _nulls(col)



def _handle_outliers(col, method, kind):
    df = shared.df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    match method:
        case "drop":
            df = df[df[col] >= lower_bound and df[col] <= upper_bound]
        case "winsorize":
            df[col] = np.clip(lower_bound, upper_bound)
        case "impute":
            match kind:
                case "mean":
                    val = df[col].mean()
                case "median":
                    val = df[col].median()
                case "mode":
                    val = df[col].mode()
            df[col] = df[col].where(df[col].between(lower_bound, upper_bound), val)




@click.command()
@click.argument('col')
@click.option('-m',"--method", type=click.Choice(["drop", "winsorize", "impute"]))
@click.option("--kind", type=click.Choice(["mean", "median", "mode"]), default=None)
def handle_outliers(col, method, kind):
   _handle_outliers(col, method, kind)
