import os
from lightning import Trainer
import numpy as np
from ts_qa.ofa_system import OFA
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from ts_qa.linearprobing_callback import LinearProbing2Fine
from ts_qa.qa_data import TimeQADataModule, OppQADataModule

def load_wandb_secret():
    try:
        # Open the file containing your secret key
        with open("/.wandb_secret") as f:
            secret_key = f.read().split("=")[1].strip()

        # Set the environment variable
        os.environ["WANDB_API_KEY"] = secret_key
    except FileNotFoundError:
        print("The .wandb_secret file was not found.")




def main():
    load_wandb_secret()
    datamodule = TimeQADataModule(name="time-qa-simple", task="binary", batch_size=10)
    datamodule.prepare_data()
    num_classes = 2 #len(datamodule.dataset["val"].features["options"])
    max_len = datamodule.max_len
    feat_dim = datamodule.feat_dim
    d_model = 768
    max_token_length =  40 #28
    patch_size = 16
    stride = 8

    # datamodule = OppQADataModule(task="multi", batch_size=32)
    # num_classes = 17
    # feat_dim = 1
    # d_model = 768
    # max_token_length = 40
    # max_len = 1500
    # patch_size = 256
    # stride = 128

    model = OFA(
        max_token_length=max_token_length,
        num_classes=num_classes, #2 if datamodule.task == "binary" else 3,
        max_seq_len=max_len,
        patch_size=patch_size,
        stride=stride,
        dropout=0.1,
        d_model=d_model,
        feat_dim=feat_dim,
        lr = 0.0001
    )
    logger = WandbLogger(project="time-qa", tags=["ofa","alignment", "simple"])
    # LightningCLI(datamodule_class=TSRegressionDataModule, model_class=OFA)
    trainer = Trainer(
        max_epochs=100, callbacks=[RichProgressBar(),
                                    LinearProbing2Fine(10)
                                    ], logger=logger,
                                    #   overfit_batches=20, limit_val_batches=0
    )

    # datamodule = TSRegressionDataModule(name="Heartbeat", batch_size=64)
    # datamodule.prepare_data()
    # model = OFA(num_classes=len(datamodule.meta["class_values"]), max_seq_len=datamodule.max_seq_len, patch_size=16, stride=8, dropout=0.1, d_model=768, feat_dim=datamodule.feat_dim)

    # trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
