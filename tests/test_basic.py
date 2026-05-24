import sys
from pathlib import Path

# Чтобы Python видел app.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import predict_diabetes, demo


def test_prediction_runs():
    """
    Тест 1:
    Проверяем, что предсказание работает
    и не падает на корректных данных.
    """

    result = predict_diabetes(
        1,      # pregnancies
        100,    # glucose
        70,     # blood pressure
        20,     # skin thickness
        80,     # insulin
        24.5,   # bmi
        0.3,    # pedigree
        25      # age
    )

    assert result is not None


def test_prediction_format():
    """
    Тест 2:
    Проверяем правильный формат ответа.
    Функция должна вернуть строку.
    """

    result = predict_diabetes(
        2,
        120,
        75,
        25,
        90,
        28,
        0.4,
        30
    )

    assert isinstance(result, str)


def test_gradio_interface_exists():
    """
    Тест 3:
    Проверяем, что интерфейс создан.
    """

    assert demo is not None