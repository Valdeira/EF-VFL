import yaml
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from importlib import import_module
import wandb

def load_module(module_path, module_name):
    spec = import_module(module_path)
    return getattr(spec, module_name)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="mnist_fullbatch/svfl.yaml", help="Path to the config file for an experiment.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

    with open("configs/" + args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    seeds = config.get("seed", [0, 1, 2, 3, 4])

    for seed in seeds:
        L.seed_everything(seed)

        if not args.no_wandb:
            wandb_logger = WandbLogger(
                project=config["logging"]["project_name"],
                name=f"{config['logging']['experiment_name']}-s{seed}",
                save_dir="./logs",
            )
            wandb_logger.experiment.config.update(config)
        else:
            wandb_logger = None

        print(f"wandb_logger.experiment.name: {wandb_logger.experiment.name if wandb_logger else 'No Wandb'}")

        # Load the data module
        data_module_path = config["data"]["module_path"]
        data_module_name = config["data"]["module_name"]
        data_module_class = load_module(data_module_path, data_module_name)
        data_module = data_module_class(**config["data"]["params"])

        data_module.prepare_data()
        data_module.setup(stage='fit')
        num_samples = data_module.num_train_samples
        batch_size = data_module.train_dataloader().batch_size

        # Load the model
        model_module_path = config["model"]["module_path"]
        model_module_name = config["model"]["module_name"]
        model_class = load_module(model_module_path, model_module_name)
        model = model_class(**config["model"]["params"], num_samples=num_samples, batch_size=batch_size)

        # Initialize the Trainer
        trainer = Trainer(
            max_epochs=config["trainer"]["max_epochs"],
            logger=wandb_logger if wandb_logger is not None else False,
            accelerator='gpu',
            devices=config["trainer"]["gpus"],
            log_every_n_steps=1,
        )
        # Train the model
        trainer.fit(model, data_module)
        # trainer.test(model, data_module)

        if wandb_logger:
            wandb.finish()

if __name__ == "__main__":
    main()

# TODO
# [] make plots more similar to the ones in the paper (but replace "test" in the plots with "validation")
# [] include "metrics to compute" argument in config -> in particular: we only care about computing sqd gd norm for mnist_fullbatch experiments
# . test metrics are not being saved
# why can't i plot comm cost vs val acc
# create a table for final test metrics
