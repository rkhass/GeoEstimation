from argparse import ArgumentParser
from pathlib import Path
from math import ceil
import pandas as pd
import torch
from tqdm.auto import tqdm

from classification.train_base import MultiPartitioningClassifier
from classification.dataset import FiveCropImageDataset, FiveCropImageDatasetCustom


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/base_M/epoch=014-val_loss=18.4833.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams",
        type=Path,
        default=Path("models/base_M/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("resources/images/im2gps"),
        help="Folder containing images. Supported file extensions: (*.jpg, *.jpeg, *.png)",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for image loading and pre-processing",
    )
    return args


# args = parse_args().parse_args()


def load_model(args):
    print("Load model from ", args.checkpoint)
    model = MultiPartitioningClassifier.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        hparams_file=str(args.hparams),
        map_location=None,
    )
    model.eval()
    if args.gpu and torch.cuda.is_available():
        model.cuda()
    return model


def load_model_custom():
    checkpoint = Path("GeoEstimation/models/base_M/epoch=014-val_loss=18.4833.ckpt")
    hparams = Path("GeoEstimation/models/base_M/hparams.yaml")
    gpu = False

    print("Load model from ", checkpoint)
    model = MultiPartitioningClassifier.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        hparams_file=str(hparams),
        map_location=None,
    )
    model.eval()
    if gpu and torch.cuda.is_available():
        model.cuda()
    return model


def load_dataloader(args):
    print("Init dataloader")
    dataloader = torch.utils.data.DataLoader(
        FiveCropImageDataset(meta_csv=None, image_dir=args.image_dir),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print("Number of images: ", len(dataloader.dataset))
    if len(dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")
    return dataloader


def load_dataloader_custom(path):
    print("Init dataloader")
    dataloader = torch.utils.data.DataLoader(
        FiveCropImageDatasetCustom(path),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print("Number of images: ", len(dataloader.dataset))
    return dataloader


def main(args):
    model = load_model(args)
    dataloader = load_dataloader(args)

    rows = []
    for X in tqdm(dataloader):
        if args.gpu:
            X[0] = X[0].cuda()
        img_paths, pred_classes, pred_latitudes, pred_longitudes = model.inference(X)
        for p_key in pred_classes.keys():
            for img_path, pred_class, pred_lat, pred_lng in zip(
                img_paths,
                pred_classes[p_key].cpu().numpy(),
                pred_latitudes[p_key].cpu().numpy(),
                pred_longitudes[p_key].cpu().numpy(),
            ):
                rows.append(
                    {
                        "img_id": Path(img_path).stem,
                        "p_key": p_key,
                        "pred_class": pred_class,
                        "pred_lat": pred_lat,
                        "pred_lng": pred_lng,
                    }
                )
    df = pd.DataFrame.from_records(rows)
    df.set_index(keys=["img_id", "p_key"], inplace=True)
    print(df)
    fout = Path(args.checkpoint).parent / f"inference_{args.image_dir.stem}.csv"
    print("Write output to", fout)
    df.to_csv(fout)


# main(args)
