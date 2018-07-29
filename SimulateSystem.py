import os
import shutil
import time
import pandas as pd


def main():
    df = pd.DataFrame()
    testing_dir = "C:/Users/richert/git/Flyometrics/testing"
    os.chdir("D:/Flyometrics")

    eye_colors = ['Red', 'White']
    sexes = ['Male', 'Female']

    for ec in eye_colors:
        for sex in sexes:
            folder = "{}Eye-{}".format(ec, sex)
            print(folder)
            files = os.listdir(folder)
            for file in files:
                shutil.copy(os.path.join(folder, file), testing_dir)
                df = df.append({
                    "EyeColor": ec,
                    "Sex": sex
                }, ignore_index=True)
                time.sleep(5)
    df.to_csv("key.csv", index=False)


if __name__ == "__main__":
    main()
