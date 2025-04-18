import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import datetime as dt
from typing import Literal
import click

from analysis import shared

# Methods for data exploration

@click.command()
@click.argument("x")
@click.argument("y")
def make_lineplot(x, y):
    df = shared.df
    sns.lineplot(data=df, x=x, y=y)
    plt.show()

@click.command()
@click.argument("x")
@click.argument("y")
def make_scatterplot(x, y):
    df = shared.df
    sns.scatterplot(data=df, x=x, y=y)
    plt.show()

@click.command()
@click.argument("x")
@click.option("--y", default=None, help="Optional y-axis for boxplot")
def make_boxplot(x, y):
    df = shared.df
    if y:
        sns.boxplot(df, x=x, y=y)
    else:
        sns.boxplot(df, x=x)
    plt.show()

@click.command()
@click.argument("x")
@click.option("--y", default=None, help="Optional y-axis for histogram")
def make_histogram(x, y):
    df = shared.df
    if y:
        sns.histplot(df, x=x, y=y)
    else:
        sns.histplot(df, x=x)
    plt.show()

@click.command()
@click.argument("col")
@click.option("--kind", type=click.Choice(["minmax", "zscore", "robust"]), default="minmax")
def normalize_values(col, kind):
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
@click.option("--val", type=click.Choice(["mean", "median", "mode"]), default="mean")
@click.option("--discrete", default=0)
def fill_null_numeric(col, val, discrete):
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
@click.argument("column")
@click.argument("new_name")
def rename_column(column, new_name):
    df = shared.df
    df.rename(columns={column: new_name}, inplace=True)

@click.command()
@click.argument("column")
def binary_encoding(column):
    df = shared.df
    if len(df[column].unique()) != 2:
        click.echo("Column does not have exactly 2 unique values.")
        return
    mapping = list(df[column].unique())
    df[column] = df[column].apply(lambda x: mapping.index(x))
    click.echo(f"Encoded: {mapping}")

@click.command()
@click.argument("column")
@click.option("--multilinearity", is_flag=True, help="Drop one dummy to reduce multicollinearity")
def one_hot_encoding(column, multilinearity):
    df = shared.df
    index = df.columns.get_loc(column)
    encoded = pd.get_dummies(df[column], drop_first=multilinearity, dtype=int, prefix=column)
    df.drop(column, inplace=True, axis=1)
    for i, col in enumerate(encoded.columns):
        df.insert(index + i, col, encoded[col])

@click.command()
@click.argument("column")
@click.option("--dtype", default="Int64")
def convert_to_numeric(column, dtype):
    df = shared.df
    if not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].apply(lambda x: re.sub(r"[^0-9.]+", "", str(x)))
    df[column] = df[column].astype(dtype)

@click.command()
@click.argument("column")
@click.option("--format", default=None, help="Optional datetime format string")
def convert_to_datetime(column, format):
    df = shared.df
    if format:
        df[column] = pd.to_datetime(df[column], format=format)
    else:
        df[column] = pd.to_datetime(df[column], errors="coerce")

@click.command()
def show():
    click.echo(shared.df)


@click.command()
@click.argument("loc")
def save(loc):
    shared.df.to_csv(loc, index=False)