# ==============================
# train.py
# Диагностика диабета (Pima Indians Diabetes Database)
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================================================
# 1. ПОДГОТОВКА ДАННЫХ
# =====================================================

print("Загрузка датасета Pima Indians Diabetes...")

# Загружаем датасет из OpenML
dataset = fetch_openml(name="diabetes", version=1, as_frame=True)

df = dataset.frame

# -----------------------------------------------------
# Признаки (X):
# preg, plas, pres, skin, insu, mass, pedi, age
#
# Целевая переменная (y):
# class / Outcome (наличие диабета)
# 1 = диабет
# 0 = нет диабета
# -----------------------------------------------------

X = df.drop(columns=["class"])
y = df["class"]

# Преобразуем target в числа
# Категориальный признак -> число
mapping = {
    "tested_positive": 1,
    "tested_negative": 0
}

y = y.map(mapping)

# Убедимся, что признаки числовые
X = X.astype(float)

# -----------------------------------------------------
# Масштабирование признаков
# Вычитаем среднее и делим на стандартное отклонение
# Это помогает моделям работать стабильнее
# -----------------------------------------------------

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================
# 2. СОБСТВЕННАЯ ПРОСТАЯ МОДЕЛЬ
# =====================================================

class GlucoseThresholdClassifier:
    """
    Простая модель:
    если уровень глюкозы высокий -> диабет
    иначе -> нет диабета
    """

    def __init__(self):
        self.threshold = None

    def fit(self, X, y):
        glucose = X[:, 1]

        best_acc = 0
        best_threshold = 0

        for t in np.linspace(glucose.min(), glucose.max(), 100):
            pred = (glucose > t).astype(int)
            acc = np.mean(pred == y)

            if acc > best_acc:
                best_acc = acc
                best_threshold = t

        self.threshold = best_threshold

    def predict(self, X):
        glucose = X[:, 1]
        return (glucose > self.threshold).astype(int)


# =====================================================
# 3. ВТОРАЯ МОДЕЛЬ (СЛОЖНАЯ)
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

simple_model = GlucoseThresholdClassifier()


# =====================================================
# 4. КРОСС-ВАЛИДАЦИЯ
# =====================================================

print("\nКросс-валидация моделей...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = ["accuracy", "precision", "recall", "f1"]


# ---------- Random Forest ----------
rf_scores = cross_validate(
    rf_model,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring=scoring
)

print("\nRandom Forest")
print("Accuracy:", rf_scores["test_accuracy"].mean())
print("Precision:", rf_scores["test_precision"].mean())
print("Recall:", rf_scores["test_recall"].mean())
print("F1:", rf_scores["test_f1"].mean())


# ---------- Простая модель ----------
simple_acc = []
simple_precision = []
simple_recall = []
simple_f1 = []

for train_idx, val_idx in cv.split(X_train_scaled, y_train):

    X_tr = X_train_scaled[train_idx]
    X_val = X_train_scaled[val_idx]

    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    simple_model.fit(X_tr, y_tr)
    pred = simple_model.predict(X_val)

    simple_acc.append(accuracy_score(y_val, pred))
    simple_precision.append(
        precision_score(y_val, pred, zero_division=0)
    )
    simple_recall.append(
        recall_score(y_val, pred, zero_division=0)
    )
    simple_f1.append(
        f1_score(y_val, pred, zero_division=0)
    )

print("\nSimple Threshold Model")
print("Accuracy:", np.mean(simple_acc))
print("Precision:", np.mean(simple_precision))
print("Recall:", np.mean(simple_recall))
print("F1:", np.mean(simple_f1))


# =====================================================
# 5. ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# =====================================================

rf_f1 = rf_scores["test_f1"].mean()
simple_f1_avg = np.mean(simple_f1)

if rf_f1 > simple_f1_avg:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = simple_model
    best_model_name = "Glucose Threshold Model"

print(f"\nЛучшая модель: {best_model_name}")


# =====================================================
# 6. ФИНАЛЬНОЕ ОБУЧЕНИЕ
# =====================================================

best_model.fit(X_train_scaled, y_train)

pred_test = best_model.predict(X_test_scaled)

acc = accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)
recall = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)

print("\n=== Итоговый тест ===")
print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nClassification Report:")
print(classification_report(y_test, pred_test))


# =====================================================
# 7. АНАЛИЗ ОШИБОК
# =====================================================

cm = confusion_matrix(y_test, pred_test)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
plt.yticks([0, 1], ["No Diabetes", "Diabetes"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.xlabel("Predicted")
plt.ylabel("Real")

plt.savefig("results/confusion_matrix.png")
plt.close()


# Какие ошибки чаще?
fp = cm[0, 1]
fn = cm[1, 0]

if fn > fp:
    confusion_text = "больных людей со здоровыми"
else:
    confusion_text = "здоровых людей с больными"


# =====================================================
# 8. СОХРАНЕНИЕ МОДЕЛИ
# =====================================================

joblib.dump(best_model, "model/saved_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nМодель сохранена.")


# =====================================================
# 9. ИТОГОВЫЙ ОТЧЁТ
# =====================================================

print("\n===================================")
print(f"Лучшая модель — {best_model_name}")
print(f"Ключевая метрика F1 на новых данных — {f1:.3f}")
print(f"Чаще всего модель путает {confusion_text}")
print("===================================")