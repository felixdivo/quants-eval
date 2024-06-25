import torch
from quants.action_encoder import QuantsBaseline
from quants.data import TSQADataModule
from lightning import Trainer
from aim.pytorch_lightning import AimLogger

if __name__ == "__main__":

    # Load data
    data_module = TSQADataModule(task="binary", batch_size=128)


    # Initialize model
    model = QuantsBaseline(num_classes=19) 

    logger = AimLogger(
        experiment="quants",
    )
    trainer = Trainer(max_epochs=3, log_every_n_steps=10, logger=logger)
    # trainer.fit(model, data_module)
    # trainer.save_checkpoint("ckpts/model.ckpt")
    model.load_state_dict(torch.load("ckpts/model.ckpt")["state_dict"])

    trainer.test(model, data_module)








