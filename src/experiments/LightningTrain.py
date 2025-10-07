import sys
import argparse
import os
import os.path as op

# Add project root to path
path = op.dirname(op.dirname(op.dirname(op.realpath(__file__))))
sys.path.append(path)

import numpy as np
import random
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

from src.misc.DynamicDataModule import DynamicDataModule
from src.models.GRUModels import GRU, StackedGRU
from src.models.LSTMModels import LSTM, StackedLSTM
from src.models.EncoderModels import TransformerEncoder, CNNEncoder, HybridEncoder
from src.models.MSCSGRUModels import MSCSGRU, MSCGRU, CNNGRU, CNNStackedGRU


parser = argparse.ArgumentParser(description="KLP training entrypoint (ASL-style)")

# General
parser.add_argument("-description", dest="description", type=str, required=False)
parser.add_argument("-test", dest="test", action="store_true", required=False)

# Data/optim
parser.add_argument("-time_steps", dest="time_steps", type=int, required=False)
parser.add_argument("-lr", dest="lr", type=float, required=False)
parser.add_argument("-batch_size", dest="batch_size", type=int, required=False)
parser.add_argument("-epochs", dest="epochs", type=int, required=False)

# Model select
parser.add_argument("-model", dest="model", type=str, required=False)
parser.add_argument("-model_type", dest="model_type", type=str, required=False)
parser.add_argument("-layers", dest="layers", type=int, required=False)
parser.add_argument("-hidden_size", dest="hidden_size", type=int, required=False)

# Encoder specific
parser.add_argument("-num_heads", dest="num_heads", type=int, required=False)
parser.add_argument("-num_layers", dest="num_layers", type=int, required=False)

parser.add_argument("-project_name", dest="project_name", type=str, required=False, help="wandb project name")

parser.set_defaults(
    batch_size=64,
    epochs=100,
    time_steps=87,
    lr=1e-3,
    model="GRU",
    model_type="RNN",
    layers=2,
    hidden_size=64,
    num_heads=8,
    num_layers=2,
    project_name="KLP-SignGlove",
    test=False,
)

args = parser.parse_args()


# Seeds
seed = 1337
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def get_model(model_name: str):
    params_common = {
        "learning_rate": args.lr,
        "input_size": 8,
        "hidden_size": args.hidden_size,
        "classes": 24,
    }

    mapping = {
        "GRU": (GRU, params_common | {"layers": args.layers}),
        "StackedGRU": (StackedGRU, params_common | {"layers": args.layers}),
        "LSTM": (LSTM, params_common | {"layers": args.layers}),
        "StackedLSTM": (StackedLSTM, params_common | {"layers": args.layers}),
        "TransformerEncoder": (TransformerEncoder, params_common | {"num_heads": args.num_heads, "num_layers": args.num_layers}),
        "CNNEncoder": (CNNEncoder, params_common),
        "HybridEncoder": (HybridEncoder, params_common | {"num_heads": args.num_heads, "num_layers": args.num_layers}),
        "MSCSGRU": (MSCSGRU, params_common | {"gru_layers": args.layers}),
        "MSCGRU": (MSCGRU, params_common),
        "CNNGRU": (CNNGRU, params_common),
        "CNNStackedGRU": (CNNStackedGRU, params_common),
    }

    if model_name not in mapping:
        raise ValueError(f"Unknown model: {model_name}")
    cls, kwargs = mapping[model_name]
    return cls(**kwargs)


def main():
    # DataModule
    dataset_params = {
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "kfold": 0,
        "splits": 5,
        "seed": seed,
        "shuffle": True,
        "use_test_split": True,
        "resampling_method": "padding",
    }

    datamodule = DynamicDataModule(**dataset_params)

    # Early Stopping Callback
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=15,
        mode='min',
        verbose=True,
        min_delta=0.001
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='./checkpoints/',
        filename='best_model_{epoch:02d}_{val/loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    # Trainer
    trainer_params = {
        "max_epochs": args.epochs,
        "log_every_n_steps": 15,
        "fast_dev_run": True if args.test else False,
        "callbacks": [early_stopping, checkpoint_callback],
    }

    wandb_logger = None
    if args.description:
        run_name = f"{args.model}_{args.description}_{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}"
        wandb_logger = WandbLogger(project=args.project_name, name=run_name)

    trainer = L.Trainer(**trainer_params, logger=wandb_logger) if wandb_logger else L.Trainer(**trainer_params)

    # Model
    model = get_model(args.model)

    # Fit and test
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()


