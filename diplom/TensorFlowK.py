import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

"""
Задача классификации
Используется набор данных MNIST для распознавания рукописных цифр.
"""

# Загрузить данные MNIST
start_time = time.time()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
data_load_time = time.time() - start_time
print(f"Время загрузки данных: {data_load_time:.4f} секунд")

# Привести данные к диапазону [0, 1]
start_time = time.time()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
normalization_time = time.time() - start_time
print(f"Время нормализации данных: {normalization_time:.4f} секунд")

# Преобразовать метки в категориальные переменные
start_time = time.time()
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
one_hot_encoding_time = time.time() - start_time
print(f"Время преобразования меток: {one_hot_encoding_time:.4f} секунд")

# Определить модель
start_time = time.time()
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model_definition_time = time.time() - start_time
print(f"Время определения модели: {model_definition_time:.4f} секунд")

# Компилировать модель
start_time = time.time()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
compilation_time = time.time() - start_time
print(f"Время компиляции модели: {compilation_time:.4f} секунд")

# Обучить модель
start_time = time.time()
model.fit(x_train, y_train,
          batch_size=64,
          epochs=5,
          validation_data=(x_test, y_test))
training_time = time.time() - start_time
print(f"Время обучения модели: {training_time:.4f} секунд")

# Оценить модель на тестовых данных
start_time = time.time()
score = model.evaluate(x_test, y_test, verbose=0)
evaluation_time = time.time() - start_time
print(f"Время оценки модели: {evaluation_time:.4f} секунд")

# Вывести результаты
test_loss, test_accuracy = score
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

# Общее время выполнения программы
total_time = data_load_time + normalization_time + one_hot_encoding_time + \
             model_definition_time + compilation_time + training_time + evaluation_time
print(f"Общее время выполнения: {total_time:.4f} секунд")


"""
Эта программа решает задачу классификации рукописных цифр с использованием набора данных MNIST. Она проходит через несколько этапов обработки данных и создания модели машинного обучения.

1. Загрузка данных:
Программа начинает с загрузки набора данных MNIST, который содержит изображения рукописных цифр и их соответствующие метки. Данные разделены на два набора: тренировочный и тестовый.

2. Приведение данных к нужному формату:
Затем программа нормализует данные, чтобы привести значения пикселей изображений к диапазону от 0 до 1. Это делается путем деления каждого пиксельного значения на 255.

3. Преобразование меток:
После этого метки классов преобразуются в категориальный формат, где каждая цифра представлена в виде вектора длины 10, содержащего единицу на позиции соответствующей цифры и нули на остальных позициях.

4. Создание модели:
Далее создается простая нейронная сеть с двумя полносвязными слоями. Первый слой имеет 128 нейронов и использует функцию активации ReLU, второй слой имеет 10 нейронов и использует функцию активации softmax для вывода вероятностей принадлежности к каждому классу.

5. Компилирование модели:
Модель компилируется с использованием оптимизатора Adam и функции потерь categorical cross-entropy. Также указывается метрику точности для отслеживания прогресса обучения.

6. Обучение модели:
На этом этапе модель обучается на тренировочных данных. Используется пакетный размер 64 и 5 эпох обучения. В процессе обучения проверяется точность модели на тестовых данных после каждой эпохи.

7. Оценка модели:
По окончании обучения производится финальная оценка модели на тестовых данных, чтобы определить её точность и потерю.

8. Вывод результатов:
Результаты оценивания выводятся вместе со временем выполнения каждой части процесса, включая загрузку данных, нормализацию, обучение и оценку.

Таким образом, программа последовательно выполняет все этапы, необходимые для построения и тестирования простой модели машинного обучения для задачи классификации изображений.
"""



