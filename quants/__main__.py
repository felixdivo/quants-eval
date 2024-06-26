from pathlib import Path
import torch
from quants.action_encoder import QuantsBaseline
from quants.data import TSQADataModule
from lightning import Trainer
from aim.pytorch_lightning import AimLogger
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="binary")

    cfg = parser.parse_args()

    task = cfg.task

    # Load data
    data_module = TSQADataModule(task=task, batch_size=128)


    # Initialize model
    model = QuantsBaseline(num_classes=19,task=task) 

    logger = AimLogger(
        experiment="quants-"+task,
    )
    trainer = Trainer(max_epochs=3, log_every_n_steps=10, logger=logger)
    if Path(f"/workspaces/ts-qa/ckpts/model-{task}.ckpt").exists():
        model.load_state_dict(torch.load(f"/workspaces/ts-qa/ckpts/model-{task}.ckpt")["state_dict"])
    else:
        trainer.fit(model, data_module)
        trainer.save_checkpoint(f"/workspaces/ts-qa/ckpts/model-{task}.ckpt")

    trainer.test(model, data_module)

    print("Done")
    print(f"Skipped {model.skipped_questions} questions")








