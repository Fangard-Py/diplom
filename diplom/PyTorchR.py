import time
import torch
from torch import nn
import matplotlib.pyplot as plt

"""
Задача регрессии
"""

# Генерация данных
x = torch.linspace(-10, 10, 100).unsqueeze(1)
y = 2 * x + 3 + torch.randn_like(x)

# Модель
model = nn.Linear(1, 1)

# Определение потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Измерение времени начала обучения
start_time = time.time()

# Обучение модели
epochs = 500
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Эпоха {epoch + 1}/{epochs}, потеря: {loss.item():.4f}')

# Измерение времени окончания обучения
end_time = time.time()

# Время выполнения обучения
training_time = end_time - start_time
print(f"Время выполнения обучения: {training_time:.2f} секунд")

# График динамики изменения потерь
plt.plot(range(epochs), losses)
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.title('Динамика изменения потерь')
plt.show()

# Проверка результатов
predicted = model(x).detach().numpy()

plt.figure(figsize=(12, 8))
plt.scatter(x.numpy(), y.numpy(), label='Истинные значения', color='blue')
plt.plot(x.numpy(), predicted, label='Предсказанные значения', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение истинных и предсказанных значений')
plt.show()

"""
Что происходит в процессе решения задачи линейной регрессии с использованием библиотеки PyTorch.

1. Подготовка данных:
Сначала генерируются входные данные (x) и соответствующие им целевые значения (y), которые представляют собой линейную зависимость с добавлением случайного шума. Это делается для того, чтобы смоделировать реальные данные, где всегда присутствует некоторая погрешность.

2. Создание модели:
Модель представляет собой простую нейронную сеть, состоящую всего из одного слоя — линейного преобразования. Этот слой принимает на вход одно значение и возвращает одно значение, так как наша задача сводится к прогнозированию одного числа на основе другого.

3. Выбор функции потерь и оптимизатора: 
Функцией потерь выбрана средняя квадратичная ошибка (MSE), поскольку она широко применяется в задачах регрессии. Оптимизация параметров модели осуществляется с помощью алгоритма стохастического градиентного спуска (SGD).

4. Процесс обучения: 
Процесс обучения состоит из нескольких этапов:

1. Инициализация градиентов: Перед каждым шагом обучения необходимо обнулить накопленные ранее градиенты, чтобы избежать их суммирования.
   
2. Прямое распространение: Данные пропускаются через модель, и вычисляются выходные значения.

3. Вычисление потерь: Рассчитывается разница между предсказанными значениями и реальными данными с использованием выбранной функции потерь.

4. Обратное распространение ошибок: Производится вычисление градиента функции потерь относительно параметров модели.

5. Обновление весов: Параметры модели корректируются на основе полученных градиентов с использованием выбранного оптимизатора.

Эти этапы повторяются заданное количество раз (эпох), пока модель не достигнет приемлемого уровня точности.

5. Оценка результата: 
После завершения обучения проверяется качество работы модели путём сравнения предсказаний с реальными значениями. Для этого строятся графики, показывающие исходные данные и результаты предсказаний.

Таким образом, данный алгоритм позволяет построить простую модель регрессии, способную аппроксимировать линейную зависимость между двумя переменными.
"""