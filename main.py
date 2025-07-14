import torch
import os
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
import utilities
import dataloading
import torchvision as tv
import evaluate_models
from models import load_model_by_name
import glob


def chtc_setup(datasets, models):
    for d in datasets:
        if "ptd" in d:
            if d.endswith(".png"):
                class_name = d.split("/")[-2]
            else:
                class_name = d.split("/")[-1]
            os.system(
                f"unzip -q {STAGING_DIR}/ptd/ptd_final/images/{class_name}.zip -d ./datasets"
            )
            # move the prompt_dict.json and metafile.txt files to the current directory
            os.system(
                f"cp {STAGING_DIR}/ptd/ptd_final/prompt_dict.json ./datasets/ptd_final"
            )
            os.system(
                f"cp {STAGING_DIR}/ptd/ptd_final/metafile.txt ./datasets/ptd_final"
            )
        else:
            os.system(f"tar -xf {STAGING_DIR}/{d}.tar -C ./datasets")
    for m in models:
        options = glob.glob(f"{STAGING_DIR}/models/imagenet/*/{m}*.pt")
        for opt in options:
            os.system(f"cp {opt} {opt.split(STAGING_DIR + '/')[-1]}")


def main(
    model_name,
    threat_model,
    subset,
    dataset_name,
    batch_size=128,
):
    root_dir = Path("datasets")
    if device == "cuda":
        chtc_setup([dataset_name], [model_name])
    model, model_transforms = load_model_by_name(model_name, threat_model)
    dataset = dataloading.BetterImageFolder(
        root_dir / dataset_name,
        transform=model_transforms,
        split_folder=str(root_dir / dataset_name),
        return_paths=True,
    )
    classes = dataset.classes
    dataloader = dataloading.get_dataloader(
        dataset,
        subset=subset,
        batch_size=batch_size,
        data_kwargs=data_kwargs,
    )
    confidences_df = evaluate_models.evaluate_model(
        model,
        dataloader,
        device,
    )
    imagenet_num_to_name_dict = utilities.imagenet_num_to_name()
    # get the class id from classes[i] and replace it with the class name from imagenet_classes
    confidences_df["labels"] = confidences_df["labels"].replace(
        imagenet_num_to_name_dict
    )
    confidences_df["preds"] = confidences_df["preds"].replace(imagenet_num_to_name_dict)
    model_name = model_name.replace("_", "-")
    suffix = f"{model_name}_{dataset_name}"
    confidences_df.to_csv(f"confidences_{suffix}.csv", index=False)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_kwargs = {"num_workers": 4, "pin_memory": True} if device == "cuda" else {}
    print(f"Using device: {device}")
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    STAGING_DIR = os.getenv("STAGING_DIR")
    if device == "cuda":
        torch.hub.set_dir(".")

    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use. Default weights will be loaded.",
        default="resnet50",
    )
    parser.add_argument(
        "--threat_model",
        type=str,
        help="Threat model to use. Default is None.",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--subset",
        help="Use a subset of the dataset for testing. Either max and min or just min can be specified. By default, the entire dataset is used.",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use. Options are 'dtd', 'ptd', 'imagenet-a', 'imagenet-o'.",
        default="ptd",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to use for testing. Default is 128. If subset is used, the batch size will be adjusted if it exceeds the length of the subset.",
        default=128,
    )
    args = parser.parse_args()
    main(
        args.model,
        args.threat_model,
        args.subset,
        args.dataset,
        args.batch_size,
    )
