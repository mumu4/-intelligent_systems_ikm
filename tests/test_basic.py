import sys
from pathlib import Path

# Чтобы Python видел app.py
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app import (
    predict_country_cluster,
    demo
)


def test_prediction_runs():
    """
    Тест 1:
    Проверяем, что функция
    предсказания работает
    и не падает.
    """

    result = predict_country_cluster(
        9.5,    # GDP
        0.8,    # Social support
        68,     # Life expectancy
        0.75,   # Freedom
        0.10,   # Generosity
        0.15    # Corruption
    )

    assert result is not None


def test_prediction_format():
    """
    Тест 2:
    Проверяем формат ответа.
    Должна возвращаться строка.
    """

    result = predict_country_cluster(
        10.2,
        0.9,
        72,
        0.85,
        0.15,
        0.10
    )

    assert isinstance(result, str)


def test_gradio_interface_exists():
    """
    Тест 3:
    Проверяем,
    что интерфейс создан.
    """

    assert demo is not None