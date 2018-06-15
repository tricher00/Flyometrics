# Simple CNN model for CIFAR-10
import numpy
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as bk
bk.set_image_dim_ordering('th')


def create_model(num_classes, shape):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    return model

# fix random seed for reproducibility
seed = 7
seed = numpy.random.seed(seed)

# normalize inputs from 0-255 to 0.0-1.0
t1 = pd.read_csv("data/Trial_1.csv")
files = ["data/images/Trial_1/{}.png".format(x) for x in t1['File name'].tolist()]
images = numpy.asarray([cv2.imread(x) for x in files])
sex = t1['Sex'].tolist()
encoder = LabelEncoder()
encoder.fit(sex)
sex = encoder.transform(sex)

images = images.astype('float32')
images = images / 255.0

sex = np_utils.to_categorical(sex)

model = create_model(sex.shape[1], images.shape[1:])

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(images[:8], sex[:8], validation_data=(images[8:], sex[8:]), epochs=epochs, batch_size=32)

scores = model.evaluate(images[8:], sex[8:], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

