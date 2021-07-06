import os
import re
import shutil
import string

import tensorflow as tf

""" Basic text classification
In this tutorial, we demonstrates how to use tensorflow to train a binary classifier to perform sentiment
analysis on an IMDB dataset(i.e. movie reviews).

The dataset which we use is from here. https://ai.stanford.edu/~amaas/data/sentiment/

"""


def show_text(file_path: str):
    with open(file_path) as f:
        print(f.read())


def step1() -> str:
    _URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    home_path = tf.keras.utils.get_file("aclImdb_v1", _URL,
                                        untar=True)
    return os.path.join(os.path.dirname(home_path), 'aclImdb')


def custom_standardization(input_data):
    lower_case = tf.strings.lower(input_data)
    remove_html_tag = tf.strings.regex_replace(lower_case, '<br />', ' ')
    return tf.strings.regex_replace(remove_html_tag,'[%s]' % re.escape(string.punctuation))



def main():
    # Step1: get the dataset
    imdb_dir = step1()
    # Step2: explore the data set.
    print("imdb data set content: {}".format(os.listdir(imdb_dir)))
    # We can find two dir which is important for our tutorial: train and test
    train_dir = os.path.join(imdb_dir, "train")
    test_dir = os.path.join(imdb_dir, "test")
    # You can find the pos and neg dir in both train and test dir. They contain many text
    # files, which are movie reviews.

    pos_review_sample = os.path.join(train_dir, 'pos/1818_8.txt')
    neg_review_sample = os.path.join(train_dir, 'neg/1187_3.txt')
    print("positive review sample: ")
    show_text(pos_review_sample)
    print("negative review sample: ")
    show_text(neg_review_sample)

    # Step3: preprocess the data set
    # To prepare a dataset for binary classification, you will need two folders on disk,
    # corresponding to class_a and class_b. In our case, it's pos and neg. But there is
    # another folder called unsup in the train_dir. So we need to remove it.
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    # When running a machine learning experiment, it is a best practice to divide your dataset
    # into three splits: train, validation and test. We already have train, test. So let's
    # creat a validation data set using an 80:20 split of the training data by using
    # validation_split argument.
    # And the subset argument defines this function returns the training (0.8) split
    # To avoid overlap between the split. We need to use a fix random seed, or set
    # shuffle=False. In our case, we choose to set seed=42.
    raw_imdb_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir, batch_size=32, validation_split=0.2, subset='training', seed=42)

    # Prepare the validation data, note it has the same seed value
    raw_imdb_validation_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir, batch_size=32, validation_split=0.2, subset='validation', seed=42)
    # Prepare the test data
    raw_imdb_test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir, batch_size=32)

    # show the first 3 comments and labels for 2 batch. You can notice the batch size is
    # 32 as we defined before
    for text_batch, label_batch in raw_imdb_train_ds.take(2):
        print("label_batch size: {}".format(len(label_batch)))
        for i in range(3):
            print("Review: {}".format(text_batch.numpy()[i]))
            print("Label: {}".format(label_batch.numpy()[i]))

    # Get classification string name
    classification_str_names = raw_imdb_train_ds.class_names
    print("class name in str: {}".format(classification_str_names))

    # Step4: Data cleaning
    # We have noticed in the reviews, there are various HTML tags in it (e.g. <br />). So
    # we need to remove them


if __name__ == "__main__":
    main()
