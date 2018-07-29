import os
import pandas as pd
from Create_Classifier import *
from datetime import datetime, timedelta


def main():
    ec_training_set, ec_test_set = create_datasets("EyeColor")
    sex_training_set, sex_test_set = create_datasets("Sex")

    ec_df = pd.read_csv("ec_data.csv")
    sex_df = pd.read_csv("sex_data.csv")

    i = 15
    while i < 50:
        ec_start_time = datetime.now()
        ec_model = create_classifier(ec_training_set, ec_test_set, i)
        ec_end_time = datetime.now()
        ec_scores = ec_model.evaluate_generator(ec_test_set)
        ec_df = ec_df.append({
            "Epochs": i,
            "Accuracy": ec_scores[1],
            "Loss": ec_scores[0],
            "Time": (ec_end_time - ec_start_time).total_seconds()
        }, ignore_index=True)
        ec_model.save("models/ec{}".format(i))
        ec_df = ec_df[["Epochs", "Accuracy", "Loss", "Time"]]
        ec_df.to_csv("ec_data.csv", index=False)

        sex_start_time = datetime.now()
        sex_model = create_classifier(sex_training_set, sex_test_set, i)
        sex_end_time = datetime.now()
        sex_scores = sex_model.evaluate_generator(sex_test_set)
        sex_df = sex_df.append({
            "Epochs": i,
            "Accuracy": sex_scores[1],
            "Loss": sex_scores[0],
            "Time": (sex_end_time - sex_start_time).total_seconds()
        }, ignore_index=True)
        sex_model.save("models/sex{}".format(i))
        sex_df = sex_df[["Epochs", "Accuracy", "Loss", "Time"]]
        sex_df.to_csv("sex_data.csv", index=False)

        i += 5


if __name__ == "__main__":
    main()