import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
import os

from models import unet_generator, patchgan_discriminator, baseline_unet
from data_generator import ImageMaskGenerator

# Path to the dataset
dataset = 'data/BraTS2020_training_data/content/data'

# Define a function to compute the Dice coefficient
def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

if __name__ == "__main__":
    # Instantiate the models
    generator = unet_generator()
    discriminator = patchgan_discriminator()
    baseline_unet_model = baseline_unet()


    discriminator.compile(loss='mse', optimizer=Adam(2e-4, 0.5))

    # Combined model setup
    input_img = Input(shape=(240, 240, 1))
    fake_mask = generator(input_img)
    discriminator.trainable = False
    valid = discriminator(concatenate([input_img, fake_mask], axis=-1))
    combined = Model(inputs=input_img, outputs=[fake_mask, valid])
    combined.compile(optimizer=Adam(2e-4, 0.5), loss=['binary_crossentropy', 'mse'], loss_weights=[1, 100])

    # Training parameters
    epochs = 25
    batch_size = 12

    all_files = os.listdir(dataset)[:10000]

    data_gen = ImageMaskGenerator(all_files, batch_size, dataset)

    # Calculate steps per epoch
    steps_per_epoch = len(all_files) // batch_size

    # Training loop
    for epoch in range(epochs):
        for batch_i, (imgs, masks) in enumerate(data_gen):
            current_batch_size = imgs.shape[0]
            if current_batch_size == 0:
                continue  # Skip empty batches
            if current_batch_size < batch_size:
                # Handle the last batch if it's smaller than the batch size
                valid = np.ones((current_batch_size, 30, 30, 1))
                fake = np.zeros((current_batch_size, 30, 30, 1))
            else:
                valid = np.ones((batch_size, 30, 30, 1))
                fake = np.zeros((batch_size, 30, 30, 1))

            fake_masks = generator.predict(imgs)
            d_loss_real = discriminator.train_on_batch(concatenate([imgs, masks], axis=-1), valid)
            d_loss_fake = discriminator.train_on_batch(concatenate([imgs, fake_masks], axis=-1), fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = combined.train_on_batch(imgs, [masks, valid])
            if batch_i % 20 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_i}, D loss: {d_loss}, G loss: {g_loss}')

    # Generate predictions 
    for imgs, masks in data_gen:
        if imgs.size == 0 or masks.size == 0:
            continue  
        predicted_masks = generator.predict(imgs)
        break  

    # Evaluate the performance
    dice_scores = []
    accuracy_scores = []
    for i in range(len(imgs)):
        dice = dice_coefficient(masks[i], predicted_masks[i])
        dice_scores.append(dice)

        # Binarize the masks
        y_true_bin = (masks[i] > 0.5).astype(np.float32)
        y_pred_bin = (predicted_masks[i] > 0.5).astype(np.float32)
 
    mean_dice = np.mean(dice_scores)

    print(f'Mean Dice Coefficient: {mean_dice}')

    baseline_unet_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coefficient])

    # Training parameters
    epochs = 25
    batch_size = 12

    all_files = os.listdir(dataset)[:10000]
    data_gen = ImageMaskGenerator(all_files, batch_size, dataset)

    # Train the baseline U-Net model
    print("Training baseline U-Net model")
    for epoch in range(epochs):
        for batch_i, (imgs, masks) in enumerate(data_gen):
            current_batch_size = imgs.shape[0]
            if current_batch_size == 0:
                continue  
            loss = baseline_unet_model.train_on_batch(imgs, masks)
            if batch_i % 20 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_i}, Loss: {loss}')

    # Evaluate the baseline U-Net model
    print("Evaluating baseline U-Net model")
    dice_scores = []
    accuracy_scores = []
    for imgs, masks in data_gen:
        if imgs.size == 0 or masks.size == 0:
            continue  
        predicted_masks = baseline_unet_model.predict(imgs)
        for i in range(len(imgs)):
            dice = dice_coefficient(masks[i], predicted_masks[i])
            dice_scores.append(dice)

            # Binarize the masks
            y_true_bin = (masks[i] > 0.5).astype(np.float32)
            y_pred_bin = (predicted_masks[i] > 0.5).astype(np.float32)


    mean_dice = np.mean(dice_scores)

    print(f'Baseline U-Net Mean Dice Coefficient: {mean_dice}')


    # Plot the original images, and the predicted masks
    n_samples = min(5, len(imgs))  # Number of samples to visualize

    plt.figure(figsize=(15, 10))
    for i in range(n_samples):
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(n_samples, 3, i*3 + 3)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.imshow(predicted_masks[i].squeeze(), cmap='Reds', alpha=0.5)  
        plt.title('Predicted Mask (Red Overlay)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()