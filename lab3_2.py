import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('music_mental_health_clean.csv')

target = 'BPM'
X = df.drop(columns=[target])
y = df[target]

print("ЭКСПЕРИМЕНТ 2")
print(f"Целевая переменная: {target}")
print(f"Размер данных: {df.shape}")

print("\nЭКСПЕРТНЫЙ АНАЛИЗ")
bpm_derived_features = ['avg_bpm_favorite', 'bpm_category']
existing_bpm_features = [f for f in bpm_derived_features if f in X.columns]
if existing_bpm_features:
    X = X.drop(columns=existing_bpm_features)
    print(f"Удалено: {existing_bpm_features}")

# 1. РАЗДЕЛЕНИЕ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# 2. ЛОГАРИФМИРОВАНИЕ
print(f"\nАсимметрия BPM на train: {y_train.skew():.3f}")
if y_train.skew() > 1:
    print("→ Применяем логарифмирование")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    y_log_flag = True
else:
    y_train_log = y_train
    y_test_log = y_test
    y_log_flag = False

# 3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ
categorical_cols = X_train.select_dtypes(include=['object']).columns
print(f"\nКатегориальные признаки: {list(categorical_cols)}")

encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encoded_train = encoder.fit_transform(X_train[categorical_cols])
encoded_test = encoder.transform(X_test[categorical_cols])

encoded_cols = encoder.get_feature_names_out(categorical_cols)
encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)

X_train = X_train.drop(columns=categorical_cols)
X_test = X_test.drop(columns=categorical_cols)
X_train = pd.concat([X_train, encoded_train_df], axis=1)
X_test = pd.concat([X_test, encoded_test_df], axis=1)
print(f"После кодирования: {X_train.shape[1]} признаков")

# 4. СТАНДАРТИЗАЦИЯ
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# 5. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
plt.figure(figsize=(14, 12))
corr_matrix = X_train_scaled.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Корреляционная матрица 2')
plt.tight_layout()
plt.savefig('exp2_correlation_matrix_train.png', dpi=150)
plt.close()

corr_with_target = X_train_scaled.corrwith(y_train_log).sort_values(ascending=False)
print("\nКОРРЕЛЯЦИЯ С BPM")
print(corr_with_target.head(15))

weak_features = corr_with_target[abs(corr_with_target) < 0.05].index.tolist()
print(f"\nСлабо связанные признаки (|r| < 0.05): {len(weak_features)} шт.")

print("\nПОИСК МУЛЬТИКОЛЛИНЕАРНОСТИ")

def remove_high_corr_features(X, y, threshold=0.95):
    X = X.copy()
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    
    for column in upper.columns:
        high_corr = upper[column][upper[column] > threshold].index.tolist()
        if high_corr:
            correlated_group = high_corr + [column]
            corr_with_y = X[correlated_group].corrwith(y).abs()
            keep = corr_with_y.idxmax()
            to_drop.update([c for c in correlated_group if c != keep])
    
    print(f"Удалено признаков из-за мультиколлинеарности: {len(to_drop)}")
    return X.drop(columns=to_drop)

# Применяем функцию
X_train_clean = remove_high_corr_features(X_train_scaled, y_train_log, threshold=0.95)
X_test_clean = X_test_scaled[X_train_clean.columns]

print(f"После удаления: {X_train_clean.shape[1]} признаков")

# 7. ОБУЧЕНИЕ МОДЕЛИ
lr = LinearRegression()
lr.fit(X_train_clean, y_train_log)
y_pred_log = lr.predict(X_test_clean)

if y_log_flag:
    y_test_orig = np.expm1(y_test_log)
    y_pred_orig = np.expm1(y_pred_log)
else:
    y_test_orig = y_test_log
    y_pred_orig = y_pred_log

print("\nЛИНЕЙНАЯ РЕГРЕССИЯ")
print(f"MSE: {mean_squared_error(y_test_orig, y_pred_orig):.3f}")
print(f"MAE: {mean_absolute_error(y_test_orig, y_pred_orig):.3f}")
print(f"R²: {r2_score(y_test_orig, y_pred_orig):.3f}")

# 8. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
coef_df = pd.DataFrame({
    'Признак': X_train_clean.columns,
    'Коэффициент': lr.coef_,
    'Важность': abs(lr.coef_)
}).sort_values('Важность', ascending=False).head(13)

print("\n13 ВАЖНЫХ ПРИЗНАКОВ")
print(coef_df.to_string(index=False))

print("\nПРЕДСКАЗАНИЕ")
new_person = pd.DataFrame({
    'Age': [23],
    'Primary streaming service': ['Youtube'],
    'Hours per day': [7],
    'While working': ['Yes'],
    'Instrumentalist': ['No'],
    'Composer': ['No'],
    'Fav genre': ['Country'],
    'Exploratory': ['No'],
    'Foreign languages': ['Yes'],
    'Frequency [Classical]': [3],
    'Frequency [Country]': [2],
    'Frequency [EDM]': [2],
    'Frequency [Folk]': [1],
    'Frequency [Gospel]': [1],
    'Frequency [Hip hop]': [2],
    'Frequency [Jazz]': [1],
    'Frequency [K pop]': [1],
    'Frequency [Latin]': [1],
    'Frequency [Lofi]': [3],
    'Frequency [Metal]': [2],
    'Frequency [Pop]': [2],
    'Frequency [R&B]': [1],
    'Frequency [Rap]': [3],
    'Frequency [Rock]': [2],
    'Frequency [Video game music]': [2],
    'Anxiety': [2],
    'Depression': [2],
    'Insomnia': [1],
    'OCD': [2],
    'Music effects': ['Improve'],
    'age_category': ['20-24'],
    'mental_sum': [7],
    'prefers_calm': [1],
    'prefers_energetic': [2],
    'avg_bpm_favorite': [120.0],
    'bpm_category': ['moderate']
})

cat_cols_in_new = [col for col in categorical_cols if col in new_person.columns]
print(f"\nКатегориальные колонки для кодирования: {cat_cols_in_new}")

# Кодирование
if cat_cols_in_new:
    new_encoded = encoder.transform(new_person[cat_cols_in_new])
    new_encoded_df = pd.DataFrame(new_encoded, columns=encoded_cols, index=new_person.index)
    new_person = new_person.drop(columns=cat_cols_in_new)
    new_person = pd.concat([new_person, new_encoded_df], axis=1)

# Добавляем недостающие колонки 
for col in X_train.columns:
    if col not in new_person.columns:
        new_person[col] = 0
new_person = new_person[X_train.columns]

# Стандартизация
new_scaled = scaler.transform(new_person)
new_clean = pd.DataFrame(new_scaled, columns=X_train.columns)

# Предсказание
pred_log = lr.predict(new_clean)
if y_log_flag:
    pred = np.expm1(pred_log)
else:
    pred = pred_log

print(f"\nПредсказанный BPM: {pred[0]:.1f}")

# 10. ВИЗУАЛИЗАЦИЯ
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
plt.plot([y_test_orig.min(), y_test_orig.max()], 
            [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel('Фактический BPM')
plt.ylabel('Предсказанный BPM')
plt.title('Линейная регрессия: предсказания vs факт')
plt.tight_layout()
plt.savefig('exp2_predictions_vs_actual.png', dpi=150)
plt.close()