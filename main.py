import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import CarDataset
from model import get_alexnet_model
from optimizer import AdaSmoothDelta

# === Путь к данным ===
data_path = "dataset"
train_img_dir = os.path.join(data_path, "cars_train")
test_img_dir = os.path.join(data_path, "cars_test")

# === Загрузка аннотаций ===
train_annos = scipy.io.loadmat(os.path.join(data_path, "cars_train_annos.mat"))["annotations"][0]
test_annos = scipy.io.loadmat(os.path.join(data_path, "cars_test_annos_withlabels_eval.mat"))["annotations"][0]
class_names = scipy.io.loadmat(os.path.join(data_path, "cars_meta.mat"))["class_names"][0]

print("✅ Аннотации загружены.")
print("🧾 Классов:", len(class_names))
print("📷 Пример изображения:", train_annos[0]["fname"][0])

# === Трансформации ===
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

transform = get_transforms()

# === Создание dataset и dataloader ===
train_dataset = CarDataset(train_annos, train_img_dir, transform)
test_dataset = CarDataset(test_annos, test_img_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✅ Датасеты готовы. Обучающих изображений:", len(train_dataset))

# === Используем предобученную AlexNet ===
num_classes = len(class_names)
alexnet = get_alexnet_model(num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)

print("✅ AlexNet готова.")

optimizer_adam = optim.Adam(alexnet.parameters(), lr=0.0001)

# === Обучение модели ===
def train_model(model, optimizer, train_loader, test_loader, device, epochs=40):
    criterion = nn.CrossEntropyLoss()
    train_acc_history = []
    test_acc_history = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_acc_history.append(train_accuracy)

        # Оценка на тесте
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_acc_history.append(test_accuracy)

        print(f"📚 Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    return train_acc_history, test_acc_history

# === Обучение с Adam ===
print("\n🔁 Обучение с Adam:")
alexnet_adam = get_alexnet_model(num_classes)
alexnet_adam = alexnet_adam.to(device)
optimizer_adam = optim.Adam(alexnet_adam.parameters(), lr=0.0001)

adam_train_acc, adam_test_acc = train_model(alexnet_adam, optimizer_adam, train_loader, test_loader, device, epochs=40)

# === Обучение с AdaSmoothDelta ===
print("\n🔁 Обучение с AdaSmoothDelta:")
alexnet_custom = get_alexnet_model(num_classes)
alexnet_custom = alexnet_custom.to(device)
optimizer_custom = AdaSmoothDelta(alexnet_custom.parameters(), lr=1.0)

custom_train_acc, custom_test_acc = train_model(alexnet_custom, optimizer_custom, train_loader, test_loader, device, epochs=40)

# === Графики ===
plt.figure(figsize=(10, 5))
plt.plot(adam_test_acc, label="Adam")
plt.plot(custom_test_acc, label="AdaSmoothDelta")
plt.xlabel("Эпоха")
plt.ylabel("Точность на тесте (%)")
plt.title("📊 Сравнение оптимизаторов")
plt.legend()
plt.grid(True)
plt.show()

# === Сохраняем модели ===
torch.save(alexnet_adam.state_dict(), "alexnet_adam.pt")
torch.save(alexnet_custom.state_dict(), "alexnet_adasmoothdelta.pt")
print("✅ Модели сохранены!")
