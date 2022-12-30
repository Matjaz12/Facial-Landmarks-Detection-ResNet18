import os
from skimage import io, transform
from skimage.util import img_as_ubyte
import pandas as pd
import numpy as np

def rescale(image, landmarks, output_size):
    h, w = image.shape[:2]

    new_h, new_w = output_size
    new_h, new_w = int(new_h), int(new_w)

    image = transform.resize(image,(new_h, new_w))
    landmarks = landmarks * [new_w / w, new_h / h]

    return image, landmarks

if __name__ == "__main__":
    IMAGE_DIRS = ["./cropped_faces/train", "./cropped_faces/test"] 
    ANNOTATIONS_DIRS = ["./cropped_faces/annotations/annotations_train.csv", "./cropped_faces/annotations/annotations_test.csv"]
    OUTPUT_SIZE = (300, 300)

    for image_dir, annotations_dir in zip(IMAGE_DIRS, ANNOTATIONS_DIRS):
        landmarks_frame = pd.read_csv(annotations_dir, index_col=0)
        resized_annotations = []

        print(f"Rescaling data in {image_dir}")

        for subdir, dirs, files in os.walk(image_dir):
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith(".jpg"):
                    image = io.imread(filepath)

                    landmarks = np.array(landmarks_frame.loc[landmarks_frame["image_name"] == file])[0][1:]
                    n_landmarks = landmarks.shape[0]
                    landmarks = landmarks.astype("float").reshape(-1, 2)

                    image, landmarks = rescale(image, landmarks, OUTPUT_SIZE)
                    image = img_as_ubyte(image)

                    io.imsave(f"./resized_data/{image_dir.split('/')[-1]}/{file}", image)
                    landmarks_r = landmarks.astype("float").reshape(n_landmarks)
                    resized_annotation = np.array([file])
                    resized_annotation = np.concatenate((resized_annotation, landmarks_r), axis=0)
                    resized_annotations.append(resized_annotation)

        landmarks_resized_frame = pd.DataFrame(resized_annotations, columns=landmarks_frame.columns)
        landmarks_resized_frame.to_csv(f"./resized_data/annotations/annotations_{image_dir.split('/')[-1]}.csv")
