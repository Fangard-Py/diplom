import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
Задача регрессии
Задача прогнозирования цен на жилье на основе синтетического набора данных.
"""

# Генерация синтетических данных
start_time = time.time()
np.random.seed(42)
n_samples = 1000
X = np.random.uniform(low=0, high=200000, size=(n_samples, 1))  # пробег автомобилей
noise = np.random.normal(scale=4000, size=n_samples)
y = 35000 - 0.08 * X.flatten() + noise  # стоимость автомобиля
data_generation_time = time.time() - start_time
print(f"Время генерации данных: {data_generation_time:.4f} секунд")

# Проверка размеров X и y
assert len(X) == len(y), "Размеры X и y должны совпадать!"

# Построение графика исходных данных
plt.figure(figsize=(12, 8))
plt.scatter(X, y, marker='.', label='Данные')
plt.xlabel('Пробег (км)')
plt.ylabel('Стоимость ($)')
plt.legend()
plt.title("Исходные данные")
plt.ion()
plt.show()
plt.pause(10)
plt.close()

# Определение модели
start_time = time.time()
model = Sequential([
    Dense(1, input_dim=1)
])
model_definition_time = time.time() - start_time
print(f"Время определения модели: {model_definition_time:.4f} секунд")

# Компиляция модели
start_time = time.time()
model.compile(optimizer='adam', loss='mse')
compilation_time = time.time() - start_time
print(f"Время компиляции модели: {compilation_time:.4f} секунд")

# Обучение модели
start_time = time.time()
history = model.fit(X, y, epochs=500, verbose=0)
training_time = time.time() - start_time
print(f"Время обучения модели: {training_time:.4f} секунд")

# Прогнозирование значений
start_time = time.time()
y_pred = model.predict(X)
prediction_time = time.time() - start_time
print(f"Время предсказания: {prediction_time:.4f} секунд")

# Построение графика результатов
plt.figure(figsize=(12, 8))
plt.scatter(X, y, marker='.', label='Исходные данные')
plt.plot(X, y_pred, color='r', label='Прогноз')
plt.xlabel('Пробег (км)')
plt.ylabel('Стоимость ($)')
plt.legend()
plt.title("Результаты прогнозирования")
plt.show()
plt.pause(10)
plt.close()

# Оценка модели
start_time = time.time()
loss = model.evaluate(X, y, verbose=0)
evaluation_time = time.time() - start_time
print(f"Время оценки модели: {evaluation_time:.4f} секунд")

# Вывести результаты
print(f'MSE: {loss:.4f}')

# Общее время выполнения программы
total_time = data_generation_time + model_definition_time + compilation_time + \
             training_time + prediction_time + evaluation_time
print(f"Общее время выполнения: {total_time:.4f} секунд")


"""
Давайте представим себе следующую картину: у нас есть задача предсказывать стоимость автомобиля на основании его пробега. Чтобы решить эту задачу, мы будем использовать метод машинного обучения под названием линейная регрессия.

1. Подготовка данных: 
Сначала нам нужно создать набор данных, состоящий из двух переменных: пробег автомобиля и его стоимость. Мы можем сгенерировать эти данные случайно, добавив немного шума, чтобы они выглядели более реалистично.

2. Визуализация данных: 
Чтобы лучше понять наши данные, мы построим график, на котором отметим каждую точку, соответствующую пробегу и цене автомобиля. Это поможет увидеть общую тенденцию зависимости цены от пробега.

3. Создание модели: 
Теперь создадим простую модель линейной регрессии. Эта модель будет иметь всего один слой, который принимает на вход пробег автомобиля и выдает предсказанную стоимость.

4. Обучение модели: 
На следующем этапе мы обучаем нашу модель на подготовленных данных. Процесс обучения заключается в том, чтобы найти такие параметры модели, при которых разница между реальной стоимостью и предсказанной будет минимальной.

5. Предсказание: 
Когда модель обучена, мы используем её для предсказания стоимости новых автомобилей на основании их пробега.

6. Оценка качества модели: 
Чтобы оценить, насколько хорошо наша модель справляется со своей задачей, мы рассчитываем среднюю квадратичную ошибку (MSE) между реальными и предсказанными значениями.

7. Анализ результатов: 
Наконец, мы строим еще один график, на котором сравниваем реальные данные с предсказаниями нашей модели. Это позволяет визуально оценить точность наших прогнозов.

Вот таким образом решается задача предсказания стоимости автомобиля на основании его пробега с использованием метода линейной регрессии.
"""