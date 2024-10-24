import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Girdi verilerini normalleştir
X_train = X_train / 255.0
X_test = X_test / 255.0

#y_train etiketleri 0-9 arasında zaten
y_test = y_test.reshape(-1,)

resim_siniflari = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(resim_siniflari[y[index]])
    plt.show()

plot_sample(X_test, y_test, 1)

deep_learning_model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

deep_learning_model.compile(optimizer="adam",
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])

# Modeli eğit (daha fazla epoch ile denenebilir)
deep_learning_model.fit(X_train, y_train, epochs=10)


deep_learning_model.evaluate(X_test, y_test)

##resimler de çok kalitesiz olduğundan başarı olasılığı çok da iyi değil.
