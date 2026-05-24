# ==============================
# app.py
# Веб-интерфейс кластеризации стран
# ==============================

import gradio as gr
import pandas as pd
import joblib


# ==========================================
# Загрузка обученной модели
# ==========================================

model = joblib.load("model/saved_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# ==========================================
# Основная функция предсказания
# ==========================================

def predict_country_cluster(
    gdp,
    social_support,
    life_expectancy,
    freedom,
    generosity,
    corruption
):
    """
    Определение кластера страны
    по социально-экономическим
    показателям.
    """

    # DataFrame с теми же признаками,
    # что использовались в train.py
    data = pd.DataFrame([{
        "Logged GDP per capita": gdp,
        "Social support": social_support,
        "Healthy life expectancy": life_expectancy,
        "Freedom to make life choices": freedom,
        "Generosity": generosity,
        "Perceptions of corruption": corruption
    }])

    # Масштабирование данных
    scaled_data = scaler.transform(data)

    # Предсказание кластера
    cluster = model.predict(scaled_data)[0]

    # Интерпретация результата
    cluster_descriptions = {
        0: (
            "🔴 Кластер 0\n\n"
            "Страны с более низкими "
            "социально-экономическими "
            "показателями."
        ),

        1: (
            "🟡 Кластер 1\n\n"
            "Страны со средним уровнем "
            "развития и качества жизни."
        ),

        2: (
            "🟢 Кластер 2\n\n"
            "Страны с высокими "
            "социально-экономическими "
            "показателями и высоким "
            "уровнем жизни."
        )
    }

    return cluster_descriptions.get(
        cluster,
        f"Кластер: {cluster}"
    )


# ==========================================
# Создание интерфейса Gradio
# ==========================================

demo = gr.Interface(
    fn=predict_country_cluster,

    inputs=[

        gr.Number(
            label="GDP per capita",
            info="Логарифм ВВП на душу населения",
            value=8.0
        ),

        gr.Number(
            label="Social support",
            info="Уровень социальной поддержки",
            value=0.8
        ),

        gr.Number(
            label="Healthy life expectancy",
            info="Ожидаемая продолжительность жизни",
            value=65
        ),

        gr.Number(
            label="Freedom to make life choices",
            info="Свобода жизненного выбора",
            value=0.7
        ),

        gr.Number(
            label="Generosity",
            info="Уровень щедрости общества",
            value=0.1
        ),

        gr.Number(
            label="Perceptions of corruption",
            info="Восприятие коррупции",
            value=0.1
        )
    ],

    outputs=gr.Textbox(
        label="Результат кластеризации"
    ),

    title="🌍 Кластеризация стран",

    description=(
        "Введите социально-экономические "
        "показатели страны, после чего "
        "модель определит, к какому "
        "кластеру она относится."
    ),

    submit_btn="Определить кластер",
    clear_btn="Очистить",

    allow_flagging="never"
)


# ==========================================
# Точка входа
# ==========================================

def main():
    """
    Запуск веб-приложения.
    """
    demo.launch()


if __name__ == "__main__":
    main()