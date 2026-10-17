import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import time

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
from skopt import BayesSearchCV
from skopt.space import Integer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = 'BPM'
X_train = train.drop(target, axis=1)
y_train = train[target]
X_test = test.drop(target, axis=1)
y_test = test[target]

rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
param_dist = {
    'n_estimators': randint(100, 1001),
    'max_depth': randint(1, 21),
    'min_samples_split': randint(2, 31),
    'min_samples_leaf': randint(1, 10)
}
search_spaces = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(1, 20),
    'min_samples_split': Integer(2, 30),
    'min_samples_leaf': Integer(1, 10)
}

results = []

print("=== Grid Search ===")
start_time = time.time()
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)
duration = time.time() - start_time

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
results.append({
    "method": "Grid Search",
    "time": duration,
    "params": grid.best_params_,
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
})
print(f"Время: {duration:.2f} с | Лучшие параметры: {grid.best_params_}")

print("\n=== Randomized Search ===")
start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0
)
random_search.fit(X_train, y_train)
duration = time.time() - start_time

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
results.append({
    "method": "Randomized Search",
    "time": duration,
    "params": random_search.best_params_,
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
})
print(f"Время: {duration:.2f} с | Лучшие параметры: {random_search.best_params_}")

print("\n=== Bayes Search ===")
start_time = time.time()
bayes_search = BayesSearchCV(
    estimator=rf,
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0
)
bayes_search.fit(X_train, y_train)
duration = time.time() - start_time

best_model = bayes_search.best_estimator_
y_pred = best_model.predict(X_test)
results.append({
    "method": "Bayes Search",
    "time": duration,
    "params": bayes_search.best_params_,
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
})
print(f"Время: {duration:.2f} с | Лучшие параметры: {bayes_search.best_params_}")

methods = [r["method"] for r in results]
times = [r["time"] for r in results]
maes = [r["MAE"] for r in results]
mses = [r["MSE"] for r in results]
r2s = [r["R2"] for r in results]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].bar(methods, maes, color='skyblue')
axs[0].set_title("MAE")
axs[1].bar(methods, mses, color='salmon')
axs[1].set_title("MSE")
axs[2].bar(methods, r2s, color='lightgreen')
axs[2].set_title("R2")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].barh(methods, times, color='orange')
axs[0].set_xlabel("Время (с)")
axs[0].set_title("Время выполнения")
axs[1].barh(methods, maes, color='skyblue')
axs[1].set_xlabel("MAE")
axs[1].set_title("Качество моделей (MAE)")
plt.tight_layout()
plt.show()

df_results = pd.DataFrame(results)
print(df_results[["method", "time", "MAE", "MSE", "R2"]])
