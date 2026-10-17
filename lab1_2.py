import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')
print(f"Всего выстрелов: {len(df)}")

x_min = np.random.randint(df['x'].min(), df['x'].max() - 100)
x_max = np.random.randint(x_min + 100, df['x'].max())
y_min = np.random.randint(df['y'].min(), df['y'].max() - 100)
y_max = np.random.randint(y_min + 100, df['y'].max())

# x_min = -100; x_max = 100
# y_min = -100; y_max = 100

print(f"\nПрямоугольник:")
print(f"Верхний левый угол: ({x_min}, {y_max})")
print(f"Правый нижний угол: ({x_max}, {y_min})")

filtered = df[(df['x'] >= x_min) & (df['x'] <= x_max) & 
            (df['y'] >= y_min) & (df['y'] <= y_max)]

print(f"Выстрелов в области: {len(filtered)}")

plt.scatter(filtered['x'], filtered['y'], alpha=0.5, c='green', s=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Прямоугольник [{x_min},{y_max}] - [{x_max},{y_min}]')
plt.grid(True)
plt.show()

col = filtered['x']
print(col)
print(df.describe())