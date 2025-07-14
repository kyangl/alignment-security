import json
import glob
import os
import argparse
import sys


def load_imagenet_json():
    with open("datasets/imagenet_class_index.json", "r") as f:
        classes = json.load(f)
    return classes


def imagenet_classes_list():
    classes = load_imagenet_json()
    return [classes[str(i)][1] for i in range(1000)]


def pretty_imagenet_classes():
    classes = imagenet_classes_list()
    # have a dictionary with the class names as keys and a prettified version of the class names as values that removes the underscores
    return {c: c.replace("_", " ") for c in classes}


def imagenet_wordnet_to_name():
    classes = load_imagenet_json()
    return {classes[str(i)][0]: classes[str(i)][1] for i in range(1000)}


def imagenet_wordnet_list():
    classes = load_imagenet_json()
    return [classes[str(i)][0] for i in range(1000)]


def imagenet_num_to_wordnet():
    classes = load_imagenet_json()
    return {i: classes[str(i)][0] for i in range(1000)}


def imagenet_num_to_name():
    classes = load_imagenet_json()
    return {i: classes[str(i)][1] for i in range(1000)}


def imagenet_num_to_name_from_list(wordnet_list):
    # in case the dataset numbers are not the same as the numbers in the imagenet class index, we can use an alternate wordnet list that has the wordnet ids in the same order as the dataset, otherwise go by the order in the imagenet class index
    wn_to_name = imagenet_wordnet_to_name()
    # create a dict that
    return {i: wn_to_name[wn] for i, wn in enumerate(wordnet_list)}


def imagenet_wordnet_list_to_name_list(wordnet_list):
    wn_to_name = imagenet_wordnet_to_name()
    return [wn_to_name[wn] for wn in wordnet_list]


def progressBar(count_value, total, suffix=""):
    bar_length = 100
    filled_up_Length = int(round(bar_length * count_value / float(total)))
    percentage = round(100.0 * count_value / float(total), 1)
    bar = "=" * filled_up_Length + "-" * (bar_length - filled_up_Length)
    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percentage, "%", suffix))
    sys.stdout.flush()
