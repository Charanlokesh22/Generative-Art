import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.datasets import cifar10

# Load and preprocess the images
def load_and_preprocess_images(image_size=(64, 64)):
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0  # Normalize images to [0, 1]
    x_train = np.array([img_to_array(img) for img in x_train])
    x_train = np.array([np.resize(img, image_size + (3,)) for img in x_train])  # Ensure correct shape
    return x_train

# Build the generator
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(256 * 16 * 16, input_dim=latent_dim))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Reshape((16, 16, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2DTranspose(3, kernel_size=7, activation='tanh', padding='same'))
    return model

# Build the discriminator model
def build_discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=image_shape))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

# Save generated images
def save_generated_images(generator, epoch, latent_dim):
    noise = np.random.normal(0, 1, size=[16, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]
    
    # Code to save images (e.g., using matplotlib or PIL) would go here
    # For example:
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    # cnt = 0
    # for i in range(4):
    
    #         axs[i,j].imshow(generated_images[cnt])
    #         axs[i,j].axis('off')
    #         cnt += 1
    # plt.savefig(f'gan_generated_image_epoch_{epoch}.png')

# Train the GAN model
def train_gan(gan, generator, discriminator, images, epochs=100, batch_size=64):
    batch_count = images.shape[0] // batch_size
    for epoch in range(epochs):
        d_loss = None
        g_loss = None
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            generated_images = generator.predict(noise)
            real_images = images[np.random.randint(0, images.shape[0], size=batch_size)]

            # Check shapes before concatenation
            if real_images.shape[1:] != generated_images.shape[1:]:
                raise ValueError(f"Shape mismatch: real_images {real_images.shape} != generated_images {generated_images.shape}")

            X = np.concatenate([real_images, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # Smooth labels for real images

            try:
                d_loss = discriminator.train_on_batch(X, y_dis)
                print(f"Discriminator loss: {d_loss}")
            except Exception as e:
                print(f"Discriminator training error: {e}")
                return

            noise = np.random.normal(0, 1, size=[batch_size, 100])
            y_gen = np.ones(batch_size)
            try:
                g_loss = gan.train_on_batch(noise, y_gen)
                print(f"Generator loss: {g_loss}")
            except Exception as e:
                print(f"Generator training error: {e}")
                return

        print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss}")
        save_generated_images(generator, epoch, latent_dim)

# Main 
def main():
    image_size = (64, 64)
    images = load_and_preprocess_images(image_size=image_size)
    
    latent_dim = 100
    image_shape = (64, 64, 3)

    generator = build_generator(latent_dim)
    discriminator = build_discriminator(image_shape)
    gan = build_gan(generator, discriminator)

    train_gan(gan, generator, discriminator, images, epochs=100)

if __name__ == "__main__":
    main()
