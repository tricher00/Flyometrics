import os
import time
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from _datetime import datetime

def main():
    image_dir = "testing"
    df = pd.DataFrame()

    eye_color_model = load_model("models/ec30")
    sex_model = load_model("models/sex30")

    os.chdir(image_dir)

    image_dir_files = os.listdir()

    last_class_time = datetime.now()
    classifying = True
    while classifying:
        diff = list(set(os.listdir()) - set(image_dir_files))
        if len(diff) is 0:
            if (datetime.now() - last_class_time).total_seconds() > 60:
                cont = input("No new images have been detected in over a minute. Would you like to continue?(Y/N)")
                cont = cont.lower()
                if cont == 'y':
                    last_class_time = datetime.now()
                elif cont == 'n':
                    classifying = False
            continue
        elif len(diff) > 1:
            print("Multiple new images found since last classification. Ending classifier.")
            classifying = False
        image_path = diff[0]
        print(image_path)
        time.sleep(1)
        test_image = image.load_img(image_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        eye_color = eye_color_model.predict_classes(test_image)[0][0]
        sex = sex_model.predict_classes(test_image)[0][0]
        print("Eye Color: {} Sex: {}".format(eye_color, sex))
        os.remove(image_path)
        image_dir_files = os.listdir()
        df = df.append({
            "File": image_path,
            "EyeColor": eye_color,
            "Sex": sex
        }, ignore_index=True)
        df.to_csv("../classifications.csv", index=False)
        last_class_time = datetime.now()


if __name__ == "__main__":
    main()
