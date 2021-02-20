from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D


def create_model():
    model = Sequential()

    model.add(Conv2D(12, kernel_size=3, activation='relu',padding='same')) #13*1023*12
    model.add(MaxPooling2D(pool_size=(3, 5), strides=None, padding='same', data_format=None))#4*204*12
    model.add(Conv2D(20, kernel_size=3, activation='relu',padding='same'))#2*202*20
    model.add(MaxPooling2D(pool_size=(1, 5), strides=None, padding='same', data_format=None))#2*40*20
    model.add(Conv2D(30, kernel_size=3, activation='relu',padding='same'))#2*40*20 --------------------------
    model.add(Conv2D(40, kernel_size=3, activation='relu',padding='same'))
    model.add(Conv2D(30, kernel_size=3, activation='relu',padding='same'))
    model.add(Conv2D(20, kernel_size=3, activation='relu',padding='same'))
    model.add(UpSampling2D(size=(1, 5), data_format=None, interpolation='nearest'))
    model.add(Conv2D(12, kernel_size=3, activation='relu',padding='same'))
    model.add(UpSampling2D(size=(3, 5), data_format=None, interpolation='nearest'))
    model.add(Conv2D(1, kernel_size=3, activation='relu',padding='same'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
