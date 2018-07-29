from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


def create_datasets(class_type):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        'data/fly/train/{}'.format(class_type),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    test_set = test_datagen.flow_from_directory(
        'data/fly/valid/{}'.format(class_type),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    return training_set, test_set


def create_classifier(training_set, test_set, epochs):
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation='relu'))

    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    classifier.fit_generator(
        training_set,
        steps_per_epoch=25,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=25
    )

    classes = {v: k for k, v in training_set.class_indices.items()}

    return classifier
