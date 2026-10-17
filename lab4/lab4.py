import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv('music_mental_health_clean.csv')
target = 'BPM'
X = df.drop(columns=[target])
y = df[target]

bpm_derived_features = ['avg_bpm_favorite', 'bpm_category']
for f in bpm_derived_features:
    if f in X.columns:
        X = X.drop(columns=[f])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_log_flag = False
if y_train.skew() > 1:
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)
    y_log_flag = True

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(drop='first', sparse_output=False)
if len(categorical_cols) > 0:
    encoder.fit(X_train[categorical_cols])
    encoded_train = encoder.transform(X_train[categorical_cols])
    encoded_test = encoder.transform(X_test[categorical_cols])
    
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
    
    X_train_num = X_train.drop(columns=categorical_cols)
    X_test_num = X_test.drop(columns=categorical_cols)
    
    X_train = pd.concat([X_train_num, encoded_train_df], axis=1)
    X_test = pd.concat([X_test_num, encoded_test_df], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = X_train_scaled 
X_test = X_test_scaled

# X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
# X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


def metrics(y_true, y_pred):
    if y_log_flag:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    mae = np.sum(np.abs(y_true - y_pred)) / n
    
    mean_y = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {'MSE': mse, 'MAE': mae, 'R2': r2}


results_list = []

rf_params_list = [
    {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt'},
    {'n_estimators': 300, 'max_depth': 10, 'max_features': 'sqrt'},
    {'n_estimators': 700, 'max_depth': 30, 'max_features': 'log2'}
]

xgb_params_list = [
    {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05},
    {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03}
]

for i, params in enumerate(rf_params_list, 1):
    print(f"Random Forest Run {i} с параметрами {params}...")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    rf.fit(X_train, y_train)
    
    metrics_dict = metrics(y_test, rf.predict(X_test))
    metrics_dict.update({'model': 'RandomForest', 'run': i})
    results_list.append(metrics_dict)

for i, params in enumerate(xgb_params_list, 1):
    print(f"XGBoost Run {i} с параметрами {params}...")
    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **params)
    xgb.fit(X_train, y_train)
    
    metrics_dict = metrics(y_test, xgb.predict(X_test))
    metrics_dict.update({'model': 'XGBoost', 'run': i})
    results_list.append(metrics_dict)

results_df = pd.DataFrame(results_list)
print(results_df.to_string())
results_df.to_csv('results.csv', index=False)

plt.figure(figsize=(6, 4))
results_for_plot = results_df.groupby('model').mean()[['MSE', 'MAE', 'R2']]
results_for_plot.plot(kind='bar')
plt.title('Сравнение средних метрик моделей')
plt.ylabel('Значение метрики')
plt.xlabel('Модель')
# plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
plt.savefig('metrics.png', dpi=200)
plt.close()
