import os
from typing import Optional
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ts_qa.linearprobing_callback import LinearProbing2Fine
from ts_qa.qa_data import OppQADataModule, TimeQADataModule

import lightning as L
from .mm_transformer import MMTransformer, MMTransformerPretrain, TextOnlyTransformer, TimeOnlyTransformer
from argparse import ArgumentParser
import os 
import torch.nn as nn 


def load_wandb_secret():
    try:
        # Open the file containing your secret key
        with open("/.wandb_secret") as f:
            secret_key = f.read().split("=")[1].strip()

        # Set the environment variable
        os.environ["WANDB_API_KEY"] = secret_key
    except FileNotFoundError:
        print("The .wandb_secret file was not found.")


class L

def main():
    load_wandb_secret()

    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="pretrain")
    parser.add_argument("--kind", type=str, default="mm")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=1)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
    # os.environ['CUDA_VISIBLE_DEVICES'] = f"1,2"


    datamodule = OppQADataModule(batch_size=54, task="binary", short=True)
    # datamodule = OppQADataModule(batch_size=8, task="binary")
    max_len = 1500
    feat_dim = 77
    # datamodule = TimeQADataModule(name="time-qa-simple", task="binary", batch_size=64)
    # max_len =  300
    # feat_dim = 24
    # datamodule.prepare_data()
    num_classes = 2  # len(datamodule.dataset["val"].features["options"])
    # max_len = datamodule.max_len
    # feat_dim = datamodule.feat_dim
    patch_len = 32
    patch_stride = 16
    tok_len = 30



    # if args.task == "pretrain":
    #     # if args.kind == "mm":
    #     system = MMTransformerPretrain(feat_dim=feat_dim, seq_len=max_len, lr=1e-3, patch_len=patch_len, patch_stride=patch_stride)
    #     print("Start pretraining")
    # else:
    #      # finetune
    #     if args.kind == "text":
    #         system = TextOnlyTransformer(num_classes=2, time_grad_mul=1.0, feat_dim=feat_dim, seq_len=max_len, patch_len=patch_len, patch_stride=patch_stride, tok_len=tok_len)
    #         print("Text only")
    #     elif args.kind == "time":
    #         system = TimeOnlyTransformer(num_classes=2, time_grad_mul=1.0, feat_dim=feat_dim, seq_len=max_len, patch_len=patch_len, patch_stride=patch_stride, tok_len=tok_len)
    #         print("Time only")
    #     elif args.kind == "mm":
    #         system = MMTransformer(num_classes=2, time_grad_mul=1.5, feat_dim=feat_dim, seq_len=max_len, lr=5e-5, patch_len=patch_len, patch_stride=patch_stride, tok_len=tok_len)
    #         print("MM")
    #     else:
    #         raise ValueError("Invalid kind")
    #     if args.ckpt is not None:
    #         system.load_pretrained(args.ckpt)
    #         print("Loaded checkpoint")
    #     system.freeze()


    logger = WandbLogger(project="opp-qa", tags=[args.kind,args.task, "binary",])# "v0.0.1"])

    # LightningCLI(datamodule_class=TSRegressionDataModule, model_class=OFA)
    trainer = Trainer(
        max_epochs=1000,
        callbacks=[
            RichProgressBar(),
            # LinearProbing2Fine(5)
            ModelCheckpoint(monitor="val/loss", save_top_k=5, mode="min"),
        ], #gpus=args.gpu,
        #   strategy="ddp", devices=2,
        logger=logger,
        val_check_interval=0.5,
        limit_train_batches=0.10
        # overfit_batches=50,
        # train
        # limit_val_batches=0,
        # enable_checkpointing=False,
    )

    

    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
