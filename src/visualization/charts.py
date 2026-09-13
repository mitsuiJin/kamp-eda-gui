"""Plotly 기반 기본 차트(히스토그램/박스플롯/막대그래프) 생성 함수 모음."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def histogram(series: pd.Series, title: str | None = None) -> go.Figure:
    return px.histogram(x=series.dropna(), title=title, labels={"x": series.name})


def boxplot(series: pd.Series, title: str | None = None) -> go.Figure:
    return px.box(y=series.dropna(), title=title, labels={"y": series.name})


def bar_chart(series: pd.Series, title: str | None = None) -> go.Figure:
    counts = series.value_counts(dropna=False)
    return px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title=title,
        labels={"x": series.name, "y": "count"},
    )


def correlation_heatmap(corr: pd.DataFrame, title: str | None = None) -> go.Figure:
    return px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
    )


def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str | None = None) -> go.Figure:
    return px.scatter(df, x=x, y=y, title=title)
