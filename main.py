import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
from argparse import ArgumentParser
from importlib import import_module


def load_module(module_path, module_name):
    spec = import_module(module_path)
    return getattr(spec, module_name)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    seed = config.get("seed", 42)
    pl.seed_everything(seed)

    wandb_logger = WandbLogger(
        project=config["logging"]["project_name"],
        name=config["logging"]["experiment_name"],
        save_dir="./logs",
    )

    wandb_logger.experiment.config.update(config)

    data_module_path = config["data"]["module_path"]
    data_module_name = config["data"]["module_name"]
    data_module_class = load_module(data_module_path, data_module_name)
    data_module = data_module_class(**config["data"]["params"])
    
    model_module_path = config["model"]["module_path"]
    model_module_name = config["model"]["module_name"]
    model_class = load_module(model_module_path, model_module_name)
    model = model_class(**config["model"]["params"])

    trainer = Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        logger=wandb_logger,
        accelerator='gpu',  # Use 'cpu' if you're not using a GPU
        devices=config["trainer"]["gpus"],
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()

# TODO
# [X] compute acc
# [X] replace current data path by data folder in parent dir
# [X] understand EFVFL (auto-)folder and replace it with something clearer
# [] start repo
# [] adjust optimizer
# [] use VFL model
# [] add direct compression module
# [] add EF compression module
