import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


if __name__ == "__main__":
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()

    print(X_train[0])

    # Plot some figures.
    plt.figure(figsize=(5,5))
    for k in range(12):
        plt.subplot(3,4, k+1)
        plt.imshow(X_train[k], cmap="Greys")
        plt.axis("off")
    plt.tight_layout()
    plt.show()