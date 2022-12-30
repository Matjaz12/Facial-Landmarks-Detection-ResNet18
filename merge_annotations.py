import os
import pandas as pd


if __name__ == "__main__":
    ANNOTATIONS_DIR = "./data/annotation"
    train_names, test_names = [], []
    annotations_train, annotations_test = [], []

    with open("./data/names/trainnames.txt", mode="r", encoding="utf-8") as f_names:
        for line in f_names:
            line = line.strip()
            train_names.append(line)


    with open("./data/names/testnames.txt", mode="r", encoding="utf-8") as f_names:
        for line in f_names:
            line = line.strip()
            test_names.append(line)

    for subdir, dirs, files in os.walk(ANNOTATIONS_DIR):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".txt"):
                # print(filepath)

                # Read image_name and landmarks
                image_name = []
                landmarks = []
                with open(filepath, mode="r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if idx == 0:
                            image_name.append(line + ".jpg")
                            continue

                        line_lst = line.split(",")
                        x, y = float(line_lst[0]), float(line_lst[1])
                        landmarks.append(x)
                        landmarks.append(y)

                if image_name[0].split(".")[0] in train_names:
                    annotations_train.append(image_name + landmarks)
                elif image_name[0].split(".")[0] in test_names:
                    annotations_test.append(image_name + landmarks)
                else:
                    raise Exception(f"{image_name} not in test not train list !")


    # Create a annotations dataframe
    annotations_cols = ["image_name"]
    for i in range(len(annotations_train[0][1: ]) // 2):
        annotations_cols.append(f"x_{i}")
        annotations_cols.append(f"y_{i}")

    annotations_df = pd.DataFrame(annotations_train, columns=annotations_cols)
    annotations_df.to_csv(ANNOTATIONS_DIR + "/annotations_train.csv")
    annotations_df = pd.DataFrame(annotations_test, columns=annotations_cols)
    annotations_df.to_csv(ANNOTATIONS_DIR + "/annotations_test.csv")