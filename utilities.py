import json
import glob
import os
import argparse
import sys
import pandas as pd


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


def setup_scores(package_path="/opt/homebrew/lib/python3.11/site-packages/"):

    mlist = glob.glob(f"{package_path}/brainscore_vision/models/*/__init__.py")

    og_model_names, reg_model_names = [], []
    for m in mlist:
        mname = m.split("/")[-2]
        with open(m, "r") as f:
            lines = f.readlines()
            for l in lines:
                if "model_registry[" in l:
                    print(l)
                    og_model_names.append(mname)
                    reg_model_names.append(
                        l.split("model_registry[")[1]
                        .split("]")[0]
                        .replace('"', "")
                        .replace("'", "")
                    )

    df_new = pd.DataFrame(
        {"og_model_name": og_model_names, "model_registry_name": reg_model_names}
    )
    df_new.to_csv("results/model_names_translated.csv", index=False)

    df_scores = pd.read_csv("results/benchmark_scores/benchmark_scores.csv")
    df_merged = pd.merge(
        df_new,
        df_scores,
        left_on="model_registry_name",
        right_on="model_name",
        how="inner",
    )
    df_merged.drop_duplicates(subset="model_registry_name", keep="first", inplace=True)
    df_merged.to_csv(
        "results/benchmark_scores/benchmark_scores_registry_merged.csv", index=False
    )
