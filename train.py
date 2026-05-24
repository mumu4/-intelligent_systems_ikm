# ==============================
# train.py
# Кластеризация стран
# World Happiness Report 2023
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. ПОДГОТОВКА ДАННЫХ

df = pd.read_csv("data/WHR2023.csv")

# Признаки (X):
#
# Logged GDP per capita
# Social support
# Healthy life expectancy
# Freedom to make life choices
# Generosity
# Perceptions of corruption
#
# Целевой переменной (y) нет,
# так как это задача кластеризации

features = [
    "Logged GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption"
]

X = df[features].copy()

# Удаляем пропуски
X = X.dropna()

# Масштабирование признаков

scaler = StandardScaler()

# Делим данные:
# 80% — обучение/проверка
# 20% — итоговый тест

X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. СОБСТВЕННАЯ ПРОСТАЯ МОДЕЛЬ


class SimpleCountryCluster:
    """
    Простая модель:
    делит страны по GDP и
    продолжительности жизни
    """

    def fit(self, X):

        gdp = X[:, 0]
        health = X[:, 2]

        self.gdp_low = np.percentile(gdp, 33)
        self.gdp_high = np.percentile(gdp, 66)

        self.health_low = np.percentile(health, 33)
        self.health_high = np.percentile(health, 66)

    def predict(self, X):

        clusters = []

        for row in X:

            gdp = row[0]
            health = row[2]

            # --- GDP уровень ---
            if gdp < self.gdp_low:
                gdp_level = 0
            elif gdp < self.gdp_high:
                gdp_level = 1
            else:
                gdp_level = 2

            # --- Health уровень ---
            if health < self.health_low:
                health_level = 0
            elif health < self.health_high:
                health_level = 1
            else:
                health_level = 2

            # --- объединение ---
            cluster = gdp_level + health_level

            clusters.append(cluster)

        return np.array(clusters)


# 3. ВТОРАЯ МОДЕЛЬ (СЛОЖНАЯ)

kmeans_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

simple_model = SimpleCountryCluster()

# 4. "КРОСС-ВАЛИДАЦИЯ"

print("\nПроверка моделей...")

n_splits = 5

simple_silhouette = []
simple_db = []

kmeans_silhouette = []
kmeans_db = []

for i in range(n_splits):

    X_tr, X_val = train_test_split(
        X_train_scaled,
        test_size=0.2,
        random_state=i
    )

    # ---------- Простая модель ----------
    simple_model.fit(X_tr)

    pred_simple = simple_model.predict(X_val)

    simple_silhouette.append(
        silhouette_score(X_val, pred_simple)
    )

    simple_db.append(
        davies_bouldin_score(X_val, pred_simple)
    )

    # ---------- KMeans ----------
    kmeans_model.fit(X_tr)

    pred_kmeans = kmeans_model.predict(X_val)

    kmeans_silhouette.append(
        silhouette_score(X_val, pred_kmeans)
    )

    kmeans_db.append(
        davies_bouldin_score(X_val, pred_kmeans)
    )

print("\nSimple Rule Model")
print("Silhouette Score:",
      np.mean(simple_silhouette))
print("Davies-Bouldin Score:",
      np.mean(simple_db))

print("\nKMeans")
print("Silhouette Score:",
      np.mean(kmeans_silhouette))
print("Davies-Bouldin Score:",
      np.mean(kmeans_db))

# 5. ВЫБОР ЛУЧШЕЙ МОДЕЛИ

simple_score = np.mean(simple_silhouette)
kmeans_score = np.mean(kmeans_silhouette)

if kmeans_score > simple_score:
    best_model = kmeans_model
    best_model_name = "KMeans"
else:
    best_model = simple_model
    best_model_name = "Simple Rule Model"

print(f"\nЛучшая модель: {best_model_name}")

# 6. ФИНАЛЬНОЕ ОБУЧЕНИЕ

best_model.fit(X_train_scaled)

pred_test = best_model.predict(X_test_scaled)

# ПРИМЕРЫ СТРАН ИЗ КАЖДОГО КЛАСТЕРА

print("\n" + "=" * 60)
print("ПРИМЕРЫ СТРАН ПО КЛАСТЕРАМ")
print("=" * 60)

test_countries = X_test.copy()
test_countries["cluster"] = pred_test

for cluster_num in [0, 1, 2]:

    print("\n" + "-" * 60)
    print(f"КЛАСТЕР {cluster_num}")
    print("-" * 60)

    cluster_examples = test_countries[
        test_countries["cluster"] == cluster_num
    ].head(5)

    if len(cluster_examples) == 0:
        print("Нет примеров")
        continue

    print(
        cluster_examples.to_string(
            index=True
        )
    )

silhouette = silhouette_score(
    X_test_scaled,
    pred_test
)

db_score = davies_bouldin_score(
    X_test_scaled,
    pred_test
)

print("\n=== Итоговый тест ===")
print("Silhouette Score:",
      silhouette)

print("Davies-Bouldin Score:",
      db_score)

# 7. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pred_test
)

plt.title("Country Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.savefig(
    "results/cluster_visualization.png"
)

plt.close()

# Какие страны чаще попадают в один кластер

cluster_counts = pd.Series(
    pred_test
).value_counts()

most_common_cluster = cluster_counts.idxmax()

# 8. СОХРАНЕНИЕ МОДЕЛИ

joblib.dump(
    best_model,
    "model/saved_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nМодель сохранена.")

# 9. ИТОГОВЫЙ ОТЧЁТ

print("\n===================================")
print(f"Лучшая модель — {best_model_name}")
print(
    f"Silhouette Score "
    f"на новых данных — "
    f"{silhouette:.3f}"
)
print(
    f"Чаще всего страны "
    f"попадают в кластер "
    f"{most_common_cluster}"
)
print("===================================")