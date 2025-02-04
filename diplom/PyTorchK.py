import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
Задача классификации
"""

# Определение устройства для вычислений (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка и преобразование данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Определение модели
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Создание экземпляра модели и перемещение её на устройство
model = Net().to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Цикл обучения
num_epochs = 10

start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100}')
            running_loss = 0.0

end_time = time.time()
training_time = end_time - start_time
print(f"Тренировка заняла {training_time:.2f} секунд.")

# Оценка модели на тестовом наборе данных
start_time = time.time()

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (100 * correct / total)
print(f'Test Accuracy: {accuracy}%')

end_time = time.time()
testing_time = end_time - start_time
print(f"Проверка заняла {testing_time:.2f} секунд.")

# Сохранение параметров модели
torch.save(model.state_dict(), 'mnist_model.pt')

"""
Для реализации задачи классификации с использованием PyTorch, мы выполняем следующие шаги:

1. Подготовка данных: 
Сначала загружаются данные из стандартного набора данных, такого как MNIST. Данные предварительно обрабатываются с помощью преобразования ToTensor, которое переводит изображения в тензоры, и нормализации, чтобы привести значения пикселей к диапазону от 0 до 1. Затем данные разбиваются на тренировочный и тестовый наборы.

2. Создание модели: 
Создается нейронная сеть, состоящая из нескольких слоев. В данном случае используется простая сверточная нейронная сеть (CNN), которая включает два сверточных слоя (Conv2d), два слоя пулинга (MaxPool2d) и два полносвязанных слоя (Linear).

3. Обучение модели: 
Модель обучается на тренировочном наборе данных. Для этого создается объект оптимизатора, который будет использоваться для обновления весов модели, и функция потерь, которая будет оценивать ошибки предсказаний. Далее запускается цикл обучения, где на каждой итерации происходит пересчет градиентов и обновление весов модели.

4. Оценка модели: 
После завершения обучения проверяется точность модели на тестовом наборе данных. Для этого модель делает предсказания для каждого изображения из тестового набора, и результаты сравниваются с истинными метками. На основе этих сравнений рассчитывается общая точность модели.

5. Сохранение и загрузка модели: 
Параметры обученной модели сохраняются в файл, чтобы их можно было использовать позже. При необходимости эти параметры можно загрузить обратно в модель для продолжения обучения или использования в других приложениях.

Таким образом, весь процесс включает в себя подготовку данных, создание и обучение модели, оценку ее производительности и сохранение результатов для последующего использования.
"""