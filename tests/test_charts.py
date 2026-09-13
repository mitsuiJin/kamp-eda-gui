"""charts의 기본 차트 생성 함수가 예외 없이 Figure를 반환하는지 확인하는 테스트."""

import pandas as pd
import plotly.graph_objects as go

from src.visualization.charts import bar_chart, boxplot, correlation_heatmap, histogram, scatter_plot


def test_histogram_returns_figure():
    series = pd.Series([1, 2, 2, 3], name="x")
    assert isinstance(histogram(series), go.Figure)


def test_boxplot_returns_figure():
    series = pd.Series([1, 2, 2, 3], name="x")
    assert isinstance(boxplot(series), go.Figure)


def test_bar_chart_returns_figure():
    series = pd.Series(["a", "b", "b"], name="cat")
    assert isinstance(bar_chart(series), go.Figure)


def test_correlation_heatmap_returns_figure():
    corr = pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])
    assert isinstance(correlation_heatmap(corr), go.Figure)


def test_scatter_plot_returns_figure():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    assert isinstance(scatter_plot(df, "a", "b"), go.Figure)
