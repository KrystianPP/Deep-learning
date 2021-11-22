import numpy as np
from numpy.random import randint, randn
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Reshape, Conv2DTranspose
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10


def def_discriminator(input_shape=(32, 32, 3)):
    model = Sequential(
        [
            Conv2D(64, 3, padding='same', input_shape=input_shape),
            LeakyReLU(alpha=0.2),
            Conv2D(128, 3, strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(128, 3, strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Conv2D(256, 3, strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


def def_generator(gen_dim):
    n_nodes = 256 * 4 * 4
    model = Sequential(
          [
                Dense(n_nodes, input_dim=gen_dim),
                LeakyReLU(alpha=0.2),
                Reshape((4, 4, 256)),
                Conv2DTranspose(128, 4, strides=2, padding='same'),
                LeakyReLU(alpha=0.2),
                Conv2DTranspose(128, 4, strides=2, padding='same'),
                LeakyReLU(alpha=0.2),
                Conv2DTranspose(128, 4, strides=2, padding='same'),
                LeakyReLU(alpha=0.2),
                Conv2D(3, 3, activation='tanh', padding='same')
          ]
    )
    return model


def load_real_imgs():
    (x_train, _), (_, _) = cifar10.load_data()
    x = x_train.astype('float32')
    x = (x - 127.5) / 127.5
    return x


def generate_real_samples(dataset, n_samples):
    i = randint(0, dataset.shape[0], n_samples)
    x = dataset[i]
    y = np.ones((n_samples, 1))
    return x, y


def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


def def_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model


def save_plot(examples, epoch, n=7):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    save_plot(x_fake, epoch)
    filename = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=150, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


latent_dim = 100
d_model = def_discriminator()
g_model = def_generator(latent_dim)
gan_model = def_gan(g_model, d_model)
dataset = load_real_imgs()
train(g_model, d_model, gan_model, dataset, latent_dim)

