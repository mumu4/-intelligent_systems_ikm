# ==============================
# app.py
# Веб-интерфейс диагностики диабета
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

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
):
    """
    Предсказание вероятности диабета
    на основе параметров пациента.
    """

    # Создаём DataFrame с правильными
    # именами признаков как в train.py
    data = pd.DataFrame([{
        "preg": pregnancies,
        "plas": glucose,
        "pres": blood_pressure,
        "skin": skin_thickness,
        "insu": insulin,
        "mass": bmi,
        "pedi": diabetes_pedigree,
        "age": age
    }])

    # Масштабирование данных
    scaled_data = scaler.transform(data)

    # Предсказание
    prediction = model.predict(scaled_data)[0]

    # Возврат результата
    if prediction == 1:
        return (
            "⚠️ Высокая вероятность диабета\n\n"
            "Рекомендуется обратиться к врачу "
            "для дополнительной диагностики."
        )

    return (
        "✅ Низкая вероятность диабета\n\n"
        "Явных признаков повышенного риска "
        "не обнаружено."
    )


# ==========================================
# Создание интерфейса Gradio
# ==========================================

demo = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(
            label="Количество беременностей",
            value=0
        ),

        gr.Number(
            label="Уровень глюкозы",
            info="Концентрация глюкозы в крови",
            value=0
        ),

        gr.Number(
            label="Артериальное давление",
            info="мм рт. ст.",
            value=0
        ),

        gr.Number(
            label="Толщина кожной складки",
            info="мм",
            value=0
        ),

        gr.Number(
            label="Уровень инсулина",
            info="мкЕд/мл",
            value=0
        ),

        gr.Number(
            label="BMI (индекс массы тела)",
            info="Показатель соотношения роста и веса",
            value=0
        ),

        gr.Number(
            label="Наследственная предрасположенность к диабету",
            info="Чем выше значение, тем сильнее семейная история диабета",
            value=0
        ),

        gr.Number(
            label="Возраст",
            info="Полных лет",
            value=0
        )
    ],

    outputs=gr.Textbox(
        label="Результат диагностики"
    ),

    title="🩺 Система диагностики диабета",

    description=(
        "Введите параметры пациента, "
        "после чего модель оценит "
        "вероятность наличия диабета."
    ),

    submit_btn="Проверить",
    clear_btn="Очистить",

    # Убираем кнопку Flag
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