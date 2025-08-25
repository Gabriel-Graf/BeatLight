from math import ceil, floor

import lightning as light
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loggers import TensorBoardLogger
import logging

import customDataloader
import customTorchLightning

# Constants
DATASET_PATH = r"Dataset_GTZAN/metadata.csv"
DATA_PATH = r"Dataset_GTZAN/samples"
MODEL_SAVING_PATH = "../saved_models/"
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 100 # 118
TRAIN_SAMPLES_PER_CLASS = ceil(SAMPLES_PER_CLASS * 0.8)
VAL_SAMPLES_PER_CLASS = floor(SAMPLES_PER_CLASS * 0.2)
BATCH_SIZE = 32
NUM_WORKERS = 3


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def plot_confusion_matrix(conf_matrix, label_dict):
    """Creates a confusion matrix plot as a Matplotlib figure."""
    conf_matrix = conf_matrix.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_dict.keys(),
        yticklabels=label_dict.keys(),
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return fig


def prepare_datasets(df, dataset):
    train_indices = []
    val_indices = []
    for c in range(NUM_CLASSES):
        start_idx = c * SAMPLES_PER_CLASS
        train_indices.extend(range(start_idx, start_idx + TRAIN_SAMPLES_PER_CLASS))
        val_indices.extend(range(start_idx + TRAIN_SAMPLES_PER_CLASS, start_idx + SAMPLES_PER_CLASS))
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return train_ds, val_ds


def create_dataloaders(train_ds, val_ds):
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4
    )
    return train_dl, val_dl


def setup_training(num_classes, train_dl):
    torch.set_float32_matmul_precision('high')
    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="min", min_delta=0.001)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="saved_models/checkpoints",
        verbose=True
    )
    logger = TensorBoardLogger("logs", name="gtzan_audio_classification")

    model = customTorchLightning.CNNAudioClassifierLightning(
        customTorchLightning.CNNAudioClassifier(num_classes)
    )

    trainer = light.Trainer(
        callbacks=[early_stop, checkpoint_callback],
        min_epochs=30,
        max_epochs=200,
        accelerator="gpu",
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        log_every_n_steps=len(train_dl),
        logger=logger,
    )
    return model, trainer, checkpoint_callback


def save_and_export_model(checkpoint_callback):
    best_ckpt_path = checkpoint_callback.best_model_path
    logging.info(f"Loading best checkpoint from: {best_ckpt_path}")
    best_model = customTorchLightning.CNNAudioClassifierLightning.load_from_checkpoint(
        best_ckpt_path,
        output_dim=NUM_CLASSES  # or dataset.num_classes
    )
    best_model.eval()

    torch.save(best_model.state_dict(), f"{MODEL_SAVING_PATH}/best_model.pth")
    logging.info("Best model saved as best_model.pt")

    dummy_input = torch.randn(1, 1, 256, 10336, device="gpu")

    torch.onnx.export(
        best_model,
        dummy_input,
        f"{MODEL_SAVING_PATH}/best_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        export_params=True,
    )
    logging.info("Best model exported as best_model.onnx")
    return best_model


def evaluate_and_plot_confusion_matrix(model, val_dl, label_dict):
    from torchmetrics.classification import MulticlassConfusionMatrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model.to(device)
    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dl:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    conf_matrix = conf_matrix_metric(all_preds, all_labels)
    logging.info("Confusion Matrix:")
    logging.info(conf_matrix.cpu().numpy())

    plot_confusion_matrix(conf_matrix, label_dict)


def main():
    setup_logging()
    logging.info("Loading dataset metadata...")
    df = pd.read_csv(DATASET_PATH, encoding="latin")
    df['relative_path'] = df['label'].astype(str) + '/' + df['filename'].astype(str)

    logging.info("Initializing custom dataset...")
    dataset = customDataloader.SoundDSMel(df, DATA_PATH)

    train_ds, val_ds = prepare_datasets(df, dataset)
    logging.info(f"Number of training samples: {len(train_ds)}")
    logging.info(f"Number of validation samples: {len(val_ds)}")

    train_dl, val_dl = create_dataloaders(train_ds, val_ds)

    batch = next(iter(train_dl))
    logging.info(batch[0].shape)

    model, trainer, checkpoint_callback = setup_training(dataset.num_classes, train_dl)

    logging.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    best_model = save_and_export_model(checkpoint_callback)

    logging.info("Evaluating model on validation set...")
    trainer.test(model, dataloaders=val_dl)

    evaluate_and_plot_confusion_matrix(best_model, val_dl, dataset.label_dict)


if __name__ == "__main__":
    main()
