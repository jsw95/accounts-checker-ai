import os
import pandas as pd
from src.data_processing import resize_img, crop_text_in_box
from skimage import io, img_as_ubyte
import numpy as np
from src.utils import plot_image
from src.data_processing import transform_imgs_for_training

# base_data_path = "/home/jack/Workspace/data/accounts/images/"
base_data_path = "/home/jwells/data/accounts/"


def create_training_set():
    labels_df = pd.read_csv(f"{base_data_path}trainset/labels/words.csv", index_col=0)

    feats = []
    for filename in os.listdir(f"{base_data_path}trainset/words/"):
        file = io.imread(f"{base_data_path}trainset/words/{filename}")
        img = transform_imgs_for_training(file)
        feats.append((filename[:-4], img[0].astype(int)))

    feats_df = pd.DataFrame(feats, columns=['word_ref', 'img'])

    joined = pd.merge(feats_df, labels_df, on='word_ref')

    filtered = joined.groupby('label').label.filter(lambda x: len(x) > 20)
    joined = joined[joined.label.isin(filtered)]

    return joined



def generate_training_folder(folder):
    files = get_img_files(folder)

    for file in files:
        filename = file.split("/")[-1][:-4]
        print(filename)
        try:

            img = io.imread(file)

            resized = resize_img(img, (30, 150))

            io.imsave(f"{base_data_path}trainset/words/{filename}.jpg", resized)

        except ValueError:
            pass


def generate_testing_folder(folder):
    files = get_img_files(folder)

    for file in files:
        filename = file.split("/")[-1][:-4]
        print(filename)
        try:
            img = io.imread(file)

            cropped_words = crop_text_in_box(img)
            for idx, word in enumerate(cropped_words):
                resized = resize_img(word, (30, 150))

                io.imsave(f"{base_data_path}images/testset/words/{filename}-{idx}.jpg", resized)

        except ValueError:
            pass


def get_img_files(folder):
    files = []

    for r, d, f in os.walk(folder):
        for file in f:
            if '.png' in file or '.jpg' in file:
                files.append(os.path.join(r, file))

    return files


def generate_labels_csv():
    with open(f"{base_data_path}words/words.txt") as f:
        word_labels = [(line.rstrip().split(" ")[0], line.rstrip().split(" ")[-1]) for line in f.readlines()[18:]]
        labels_df = pd.DataFrame(word_labels, columns=['word_ref', 'label'])

        labels_df.to_csv(f"{base_data_path}trainset/labels/words.csv")


def save_box_images(box_locations, path):
    sorted_box_locations = sorted(box_locations, key=lambda x: [x[0][0], x[0][1]])

    for idx, box in enumerate(sorted_box_locations):
        box_img = box[1]
        box_img = img_as_ubyte(box_img)

        io.imsave(f"{base_data_path}/{path}box{idx}.jpg", box_img)
