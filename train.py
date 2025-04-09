import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, optimizer, train_loader, test_loader, device, epochs=10):
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
