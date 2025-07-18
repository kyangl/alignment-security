import argparse
import logging
import os
from dotenv import load_dotenv
import torch
from brainscore_vision import load_model
from torchmetrics.classification import Accuracy
import csv
import dataloading
from utilities import progressBar
from torchattacks import AutoAttack


def chtc_setup(datasets):
    for d in datasets:
        if d is None:
            continue
        if "ptd" in d:
            class_name = d.split("/")[-2]
            print(f"Unzipping {class_name}.zip")
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
            os.system(
                f"cp {STAGING_DIR}/ptd/ptd_final/metafile_filtered.txt ./datasets/ptd_final"
            )
        else:
            os.system(f"tar -xf {STAGING_DIR}/{d}.tar -C ./datasets")


def load_dataset(
    threat_model, preprocess, corruption_type=None, corruption_severity=None
):
    """
    Load the appropriate dataset based on the threat model.
    """
    if threat_model in ["clean", "Linf"]:
        dataset_name = "imagenet-val"
        dataset_path = f"datasets/{dataset_name}"
    elif threat_model == "corruptions":
        if not corruption_type or not corruption_severity:
            raise ValueError(
                "Corruption type and severity must be provided for common corruptions."
            )
        dataset_name = "imagenet-c"
        dataset_path = (
            f"datasets/Tiny-ImageNet-C/{corruption_type}/{corruption_severity}"
        )
    else:
        raise ValueError(f"Unknown threat model: {threat_model}")
    if args.chtc:
        chtc_setup([dataset_name])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    dataset = dataloading.BetterImageFolder(root=dataset_path, transform=preprocess)
    return dataloading.get_dataloader(
        dataset,
        batch_size=args.batch_size,
        subset=args.subset,
        data_kwargs=({"num_workers": 4} if device == "cuda" else {}),
    )


def evaluate_model(model, data_loader, device, top_k=(1, 5)):
    """
    Evaluate a model using a single evaluation loop.
    """
    model.eval()
    model.to(device)

    metrics = {
        k: Accuracy(num_classes=1000, task="multiclass", top_k=k).to(device)
        for k in top_k
    }

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            progressBar(i, len(data_loader))
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            try:
                outputs = outputs.logits
            except:
                pass
            for k, metric in metrics.items():
                metric.update(outputs, labels)

    return {
        f"Top-{k} Accuracy": metric.compute().item() for k, metric in metrics.items()
    }


def attack_model(
    model, data_loader, device, top_k=(1, 5), eps=1 / 255, mean=None, std=None
):
    """
    Evaluate a model using a single evaluation loop.
    """
    model.eval()
    model.to(device)

    metrics = {
        k: Accuracy(num_classes=1000, task="multiclass", top_k=k).to(device)
        for k in top_k
    }
    clean_metrics = {
        k: Accuracy(num_classes=1000, task="multiclass", top_k=k).to(device)
        for k in top_k
    }

    attack = AutoAttack(model, norm="Linf", eps=eps, version="standard", n_classes=1000)
    attack.set_normalization_used(mean, std)
    mean, std = torch.tensor(mean).to(device), torch.tensor(std).to(device)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            progressBar(i, len(data_loader))
            images, labels = batch[0].to(device), batch[1].to(device)
            adv_images = attack(images, labels)
            outputs = model(adv_images)
            for k, metric in metrics.items():
                metric.update(outputs, labels)
            clean_outputs = model(images)
            for k, metric in clean_metrics.items():
                metric.update(clean_outputs, labels)
    return {
        f"Top-{k} Accuracy": metric.compute().item() for k, metric in metrics.items()
    } | {
        f"Top-{k} Subset Clean Accuracy": metric.compute().item()
        for k, metric in clean_metrics.items()
    }


def unnormalize(tensor, mean, std):
    # Unnormalize the tensor
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    return tensor * std + mean


def main():

    print(f"Loading model: {args.model_name}")
    # Load BrainScore model and preprocessing
    model_wrapper = load_model(args.model_name)
    model = model_wrapper.activations_model._model
    preprocess = model_wrapper.activations_model._extractor.preprocess
    if isinstance(preprocess, dict):
        preprocess = preprocess["transforms"]
    try:
        mean = preprocess.keywords["normalize_mean"]
        std = preprocess.keywords["normalize_std"]
    except:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    # Load the dataset
    data_loader = load_dataset(
        threat_model=args.threat_model,
        preprocess=preprocess,
        corruption_type=(
            args.corruption_type if args.threat_model == "corruptions" else None
        ),
        corruption_severity=(
            args.corruption_severity if args.threat_model == "corruptions" else None
        ),
    )
    if args.threat_model == "Linf":
        if args.eps > 0.3:
            print("Epsilon is too high, setting to eps/255")
            args.eps = args.eps / 255
        # Attack the model
        results = attack_model(
            model, data_loader, device, top_k=(1, 5), eps=args.eps, mean=mean, std=std
        )
    else:
        # Evaluate the model
        results = evaluate_model(model, data_loader, device, top_k=(1, 5))
    print(f"Results: {results}")

    # Add additional metadata to results
    results["Model Name"] = args.model_name
    results["Threat Model"] = args.threat_model
    if args.threat_model == "corruptions":
        results["Corruption Type"] = args.corruption_type
        results["Corruption Severity"] = args.corruption_severity
    if args.threat_model == "Linf":
        results["Epsilon"] = args.eps

    # Save results to CSV
    output_file = "evaluation_results.csv"
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()  # Write the header only if the file doesn't exist
        writer.writerow(results)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_kwargs = {"num_workers": 4, "pin_memory": True} if device == "cuda" else {}
    print(f"Using device: {device}")
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    STAGING_DIR = os.getenv("STAGING_DIR")
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a BrainScore model on ImageNet with AutoAttack."
    )
    # add a subparser for the threat model
    subparsers = parser.add_subparsers(dest="threat_model")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the BrainScore model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation."
    )
    parser.add_argument(
        "-s",
        "--subset",
        help="Use a subset of the dataset for testing. Either max and min or just min can be specified. By default, the entire dataset is used.",
        nargs=2,
        type=int,
        default=None,
    )
    parser.add_argument(
        "--chtc",
        action="store_true",
        help="Set this flag to run on the CHTC.",
    )
    clean_parser = subparsers.add_parser("clean")
    linf_parser = subparsers.add_parser("Linf")
    linf_parser.add_argument(
        "--eps",
        type=float,
        default=1 / 255,
        help="Value to use for the epsilon of the attack.",
    )
    corruptions_parser = subparsers.add_parser("corruptions")
    corruptions_parser.add_argument(
        "--corruption_type",
        type=str,
        required=True,
        help="Type of corruption for ImageNet-C (e.g., 'blur').",
    )
    corruptions_parser.add_argument(
        "--corruption_severity",
        type=int,
        required=True,
        help="Severity level of corruption (1-5).",
    )
    args = parser.parse_args()
    if args.chtc:
        torch.hub.set_dir(f"{STAGING_DIR}/.cache/torch")
    main()
