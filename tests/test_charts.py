"""charts의 기본 차트 생성 함수가 예외 없이 Figure를 반환하는지 확인하는 테스트."""

import pandas as pd
import plotly.graph_objects as go

from src.visualization.charts import bar_chart, boxplot, histogram


def test_histogram_returns_figure():
    series = pd.Series([1, 2, 2, 3], name="x")
    assert isinstance(histogram(series), go.Figure)


def test_boxplot_returns_figure():
    series = pd.Series([1, 2, 2, 3], name="x")
    assert isinstance(boxplot(series), go.Figure)


def test_bar_chart_returns_figure():
    series = pd.Series(["a", "b", "b"], name="cat")
    assert isinstance(bar_chart(series), go.Figure)
