from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import SeparableConvolution2D, DepthwiseConv2D

#keras functional network
def create_base_net(input_shape):

    input = Input(shape = input_shape)
    x = DepthwiseConv2D((16,16), activation='relu', strides=(3,3), padding='same')(input)
    x = Conv2D(32, (1,1), activation='relu', strides=(1,1))(x)
    x = MaxPooling2D(pool_size = (2,2), strides=(2,2))(x)

    x = SeparableConvolution2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)

    x = SeparableConvolution2D(256, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)

    x = Flatten()(x)
    x = Dense(512, activation = 'sigmoid')(x)

    model = Model(input, x)
    model.summary()
    return model