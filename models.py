import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.models import Model

def unet_generator(input_shape=(240, 240, 1)):
    inputs = Input(input_shape)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    b = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    b = BatchNormalization()(b)

    u1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(b)
    u1 = concatenate([u1, c2])
    u1 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    u1 = BatchNormalization()(u1)

    u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(u1)
    u2 = concatenate([u2, c1])
    u2 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    u2 = BatchNormalization()(u2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u2)
    model = Model(inputs, outputs)
    return model

def patchgan_discriminator(input_shape=(240, 240, 2)):
    inputs = Input(input_shape)
    d1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
    d1 = LeakyReLU(0.2)(d1)

    d2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(0.2)(d2)

    d3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(0.2)(d3)

    d4 = Conv2D(512, (4, 4), padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU(0.2)(d4)

    outputs = Conv2D(1, (4, 4), padding='same')(d4)
    model = Model(inputs, outputs)
    return model

def baseline_unet(input_shape=(240, 240, 1)):
    inputs = Input(input_shape)
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    b = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    b = BatchNormalization()(b)

    # Decoder
    u1 = UpSampling2D(size=(2, 2))(b)
    u1 = concatenate([u1, c4])
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(u1)
    c5 = BatchNormalization()(c5)

    u2 = UpSampling2D(size=(2, 2))(c5)
    u2 = concatenate([u2, c3])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u2)
    c6 = BatchNormalization()(c6)

    u3 = UpSampling2D(size=(2, 2))(c6)
    u3 = concatenate([u3, c2])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    c7 = BatchNormalization()(c7)

    u4 = UpSampling2D(size=(2, 2))(c7)
    u4 = concatenate([u4, c1])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c8 = BatchNormalization()(c8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    model = Model(inputs, outputs)
    return model