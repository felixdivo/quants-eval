from pathlib import Path
import rich
import torch
from quants.action_encoder import QuantsBaseline
from quants.data import TSQADataModule
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from aim.pytorch_lightning import AimLogger
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import wandb

from omegaconf import OmegaConf
from pathlib import Path
import os
import tempfile
from rich.console import Console
from rich.syntax import Syntax


def qa_builder(q_template,a_template, prompts): 
    final_prompts = ""
    should_add_id = len(prompts) > 1
    for idx, p in enumerate(prompts):
        final_prompts += "\nExample"
        if should_add_id:
            final_prompts += f" {idx+1}"
        final_prompts += f":\n{q_template.format(**p)}\n{a_template.format(**p)}"

    return final_prompts




# def qa_builder(q_template,a_template, prompts): 
#     final_prompts = []
#     for p in prompts:
#         final_prompts.append({"role":"user", "content": q_template.format(**p)})
#         final_prompts.append({"role": "assistant", "content": a_template.format(**p)})

#     return final_prompts

OmegaConf.register_new_resolver( "qa_builder", qa_builder, replace=True)

def load_and_merge_configs(config_path: Path):
    # Load the main configuration file
    config = OmegaConf.load(config_path)

    # Load and merge additional configs specified in the 'imports' section
    if 'imports' in config:
        for import_item in config.imports:
            if isinstance(import_item, str):
                # When only the file path is given
                relative_conf_file = import_item
                key = None
            else:
                key, relative_conf_file = next(iter(import_item.items()))
            conf_file_path = Path(config_path.parent) / relative_conf_file

            if conf_file_path.is_file():
                additional_config = load_and_merge_configs(conf_file_path)
                if key is not None:
                    if key not in config:
                        config[key] = OmegaConf.create()
                    # Merge the additional config under the key
                    config[key] = OmegaConf.merge( additional_config,config[key])
                else:
                    # Merge the additional config at the top level
                    config = OmegaConf.merge(additional_config, config)

            else:
                raise FileNotFoundError(f"Config file {conf_file_path} not found.")
    
    # Remove the imports section after merging
    if 'imports' in config:
        del config['imports']

    return config

# Path to the default configuration file
default_config_file = Path(os.getcwd()) / "quants" / "config" / "outlines.yaml"





def evaluate_question_type(task, t, idx, total_types, config):
    print(f"Evaluating question type {t} ({idx}/{total_types})")
    

    yaml_string = OmegaConf.to_yaml(config, resolve=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as temp_file:
        temp_file.write(yaml_string.encode('utf-8'))
        temp_file_path = temp_file.name

    exp = wandb.init(project="quants", entity="ml-research", group=f"{task}-{t}", job_type="eval", tags=["dbg"])

    artifact = wandb.Artifact('llm_template', type='config')
    artifact.add_file(temp_file_path)
    wandb.log_artifact(artifact)
    os.remove(temp_file_path)

    logger = WandbLogger(experiment=exp)

    trainer = Trainer(max_epochs=1, log_every_n_steps=10, logger=logger)
    
    model_ckpt_path = f"/workspaces/ts-qa/ckpts/model-{task}.ckpt"
    if Path(model_ckpt_path).exists():
        model = QuantsBaseline(num_classes=19, task=task, config=config, gt_annotations=True)
        model.load_state_dict(
            torch.load(model_ckpt_path)["state_dict"]
        )
    else:
        raise RuntimeError("Model not found, please restart the script")

    data_module = TSQADataModule(task=task, batch_size=128, q_type=t)
    trainer.test(model, data_module)
    print(f"Done evaluating question type {t}")
    print(f"Skipped {model.skipped_questions} questions")
    exp.log({"skipped_questions": model.skipped_questions})
    # logger.experiment.log_info(f"Done evaluating question type {t}")
    # logger.experiment.log_info(f"Skipped {model.skipped_questions} questions")

    
    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="binary")

    cfg = parser.parse_args()

    task = cfg.task
    task_data_folder = Path(f"/workspaces/ts-qa/preprocess_data/ts_qa/{task}")

    # Load data
    data_module = TSQADataModule(task=task, batch_size=128)

    # Initialize model
    model = QuantsBaseline(num_classes=19, task=task)

    model_ckpt_path = f"/workspaces/ts-qa/ckpts/model-{task}.ckpt"
    if Path(model_ckpt_path).exists():
        # model.load_state_dict(
        #     torch.load(model_ckpt_path)["state_dict"]
        # )
        pass
    else:
        logger = AimLogger(
            experiment=f"quants-{task}/pretrain",
            train_metric_prefix="train/",
            test_metric_prefix="test/",
            val_metric_prefix="val/",
        )
        trainer = Trainer(max_epochs=3, log_every_n_steps=10, logger=logger)
        trainer.fit(model, data_module)
        trainer.save_checkpoint(model_ckpt_path)

    types = [item.name for item in task_data_folder.iterdir() if item.is_dir()][13:15]
    print("Evaluating question types")


    # Load and merge configurations
    config = load_and_merge_configs(default_config_file)

    console = Console()

    syntax = Syntax(OmegaConf.to_yaml(config, resolve=True), 'yaml')
    console.print(syntax)

    
    # # Use map to apply the evaluate_question_type function sequentially
    for idx, t in enumerate(types):
        evaluate_question_type(task, t, idx, len(types), config)
        