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


def grouped_boxplot(df: pd.DataFrame, group_column: str, value_column: str, title: str | None = None) -> go.Figure:
    return px.box(df, x=group_column, y=value_column, title=title)


def grouped_histogram(df: pd.DataFrame, group_column: str, value_column: str, title: str | None = None) -> go.Figure:
    return px.histogram(
        df,
        x=value_column,
        color=group_column,
        barmode="overlay",
        opacity=0.6,
        title=title,
    )


def outlier_scatter_over_time(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    mask: pd.Series,
    title: str | None = None,
) -> go.Figure:
    labels = mask.map({True: "이상치", False: "정상"})
    return px.scatter(df, x=time_column, y=value_column, color=labels, title=title)


def time_series_line(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    rolling_window: int | None = None,
    title: str | None = None,
) -> go.Figure:
    fig = px.line(df, x=time_column, y=value_column, title=title)
    if rolling_window and rolling_window > 1:
        rolling = df[value_column].rolling(rolling_window, min_periods=1).mean()
        fig.add_scatter(x=df[time_column], y=rolling, mode="lines", name=f"rolling mean ({rolling_window})")
    return fig
