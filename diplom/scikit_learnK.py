import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
Задача классификации
Базовый подход к решению задачи классификации с помощью библиотеки Scikit-learn.
"""

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__}: Время выполнения: {(end_time - start_time):.4f} секунд')
        return result

    return wrapper


@measure_time
def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y


@measure_time
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@measure_time
def scale_data(train_data, test_data):
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    return scaled_train_data, scaled_test_data


@measure_time
def train_model(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


@measure_time
def predict(model, X_test):
    return model.predict(X_test)


if __name__ == "__main__":
    # Загрузить данные
    X, y = load_data()

    # Разделение данных на обучающие и тестовые наборы
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Масштабирование данных
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Обучить модель
    trained_model = train_model(X_train_scaled, y_train)

    # Сделать прогноз
    y_pred = predict(trained_model, X_test_scaled)

    # Оценить точность модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')


"""
Этот пример демонстрирует базовый подход к решению задачи классификации с помощью библиотеки Scikit-learn. 

1. Загрузка данных: 
Мы загружаем встроенный набор данных Iris, который содержит информацию о трёх видах ирисов. Для простоты используем только две характеристики (признаки), такие как длина и ширина лепестков.

2. Разделение данных: 
Данные делятся на обучающий и тестовый наборы. Это необходимо для того, чтобы модель могла обучиться на одной части данных, а затем проверить свою точность на другой, ранее невидимой ей части.

3. Масштабирование данных: 
Признаки могут иметь разные диапазоны значений, поэтому перед обучением модели важно привести их к одному масштабу. Это делается с помощью стандартизации, когда все значения преобразуются так, чтобы среднее значение было равно нулю, а стандартное отклонение — единице.

4. Обучение модели: 
Модель логистической регрессии обучается на обучающих данных. Логистическая регрессия — это метод классификации, который хорошо подходит для бинарных задач (когда нужно разделить объекты на два класса).

5. Прогнозирование: 
После обучения модель делает прогнозы на тестовом наборе данных. Эти прогнозы сравниваются с реальными значениями классов, чтобы оценить точность модели.

6. Оценка точности: 
Сравнивая реальные классы объектов с теми, которые предсказала модель, вычисляется метрика точности. Она показывает долю правильно классифицированных примеров среди всех тестовых данных.

Таким образом, этот процесс представляет собой стандартный пайплайн для решения задачи классификации с использованием библиотеки Scikit-learn.
"""