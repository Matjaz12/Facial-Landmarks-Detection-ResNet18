import os
from skimage import io, transform
from skimage.util import img_as_ubyte
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def crop_face(image, landmarks):
    # Find bottom most landmark, left most landmark, right most landmark
    bottom_most = landmarks[np.argmax(landmarks[:, 1], keepdims=True)[0]].copy()
    left_most = landmarks[np.argmin(landmarks[:, 0], keepdims=True)[0]].copy()
    right_most = landmarks[np.argmax(landmarks[:, 0], keepdims=True)[0]].copy()

    # Compute upper most landmark
    padding = 5
    upper_most = landmarks[np.argmin(landmarks[:, 1], keepdims=True)[0]].copy()
    upper_bottom_dist = abs(upper_most[1] - bottom_most[1])
    temp =  upper_most[1] - 1 / 3 * upper_bottom_dist
    upper_most[1] = temp if temp > 0 else padding
    
    # Compute the masked image
    h, w = image.shape[:2]
    # masked_image = np.zeros_like(image)
    row_min = int(upper_most[1]) - padding if (int(upper_most[1]) - padding) >= 0 else int(upper_most[1])
    row_max = int(bottom_most[1]) + padding if (int(bottom_most[1]) + padding) < h else int(bottom_most[1])
    col_min = int(left_most[0]) - padding if (int(left_most[0]) - padding) > 0 else int(left_most[0])
    col_max = int(right_most[0]) + padding if (int(right_most[0]) + padding) <= w else int(right_most[0])

    # Special case: sometimes the annotations are over slightly over the image dimension
    # we only have to correct for the following two cases

    if col_min < 0:
        pass

    row_min = 0 if row_min < 0 else row_min
    col_min = 0 if col_min < 0 else col_min

    #masked_image[row_min:row_max, col_min:col_max] = image[row_min:row_max, col_min:col_max]
    cropped_image = image[row_min:row_max, col_min:col_max].copy()
    cropped_landmarks = np.zeros_like(landmarks)
    cropped_landmarks[:, 0] = landmarks[:, 0] - col_min
    cropped_landmarks[:, 1] = landmarks[:, 1] - row_min

    # _, ax = plt.subplots(3)
    # ax[0].imshow(image)
    # ax[0].scatter(x=landmarks[:, 0], y=landmarks[:, 1], s=20, marker=".")
    #  #ax[1].imshow(masked_image)
    #  #ax[1].scatter(x=landmarks[:, 0], y=landmarks[:, 1], s=20, marker=".")
    # ax[2].imshow(cropped_image)
    # ax[2].scatter(x=cropped_landmarks[:, 0], y=cropped_landmarks[:, 1], s=20, marker=".")
    # plt.show()

    return cropped_image, cropped_landmarks


if __name__ == "__main__":
    IMAGE_DIRS = ["./data/train", "./data/test"] 
    ANNOTATIONS_DIRS = ["./data/annotation/annotations_train.csv", "./data/annotation/annotations_test.csv"]

    for image_dir, annotations_dir in zip(IMAGE_DIRS, ANNOTATIONS_DIRS):

        landmarks_frame = pd.read_csv(annotations_dir, index_col=0)
        resized_annotations = []

        print(f"Cropping faces in {image_dir}...")

        for subdir, dirs, files in os.walk(image_dir):
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith(".jpg"):
                    image = io.imread(filepath)

                    landmarks = np.array(landmarks_frame.loc[landmarks_frame["image_name"] == file])[0][1:]
                    n_landmarks = landmarks.shape[0]
                    landmarks = landmarks.astype("float").reshape(-1, 2)

                    cropped_image, cropped_landmarks = crop_face(image, landmarks)
                    image = img_as_ubyte(image)

                    #print(f"./cropped_faces/{image_dir.split('/')[-1]}/{file}")
                    io.imsave(f"./cropped_faces/{image_dir.split('/')[-1]}/{file}", cropped_image)
                    landmarks_r = cropped_landmarks.astype("float").reshape(n_landmarks)
                    resized_annotation = np.array([file])
                    resized_annotation = np.concatenate((resized_annotation, landmarks_r), axis=0)
                    resized_annotations.append(resized_annotation)

        landmarks_resized_frame = pd.DataFrame(resized_annotations, columns=landmarks_frame.columns)
        landmarks_resized_frame.to_csv(f"./cropped_faces/annotations/annotations_{image_dir.split('/')[-1]}.csv")
