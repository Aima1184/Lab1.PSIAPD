import matplotlib.pyplot as plt

def plot_results(adam_test_acc, custom_test_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(adam_test_acc, label="Adam")
    plt.plot(custom_test_acc, label="AdaSmoothDelta")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность на тесте (%)")
    plt.title("📊 Сравнение оптимизаторов")
    plt.legend()
    plt.grid(True)
    plt.show()
