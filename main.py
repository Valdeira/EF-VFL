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


def main(args):

    with open("configs/" + args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    config['gpu'] = args.gpu
    config['seeds'] = args.seeds

    for seed in args.seeds:
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

        data_module_class = load_module(config["data"]["module_path"], config["data"]["module_name"])
        data_module = data_module_class(**config["data"]["params"])

        data_module.prepare_data()
        data_module.setup(stage='fit')
        data_module.setup(stage='validate')
        num_samples = data_module.num_train_samples
        batch_size = data_module.train_dataloader().batch_size

        model_class = load_module(config["model"]["module_path"], config["model"]["module_name"])
        model = model_class(**config["model"]["params"], num_samples=num_samples, batch_size=batch_size)

        trainer = Trainer(
            max_epochs=config["trainer"]["max_epochs"],
            logger=wandb_logger if wandb_logger is not None else False,
            accelerator='gpu',
            devices=args.gpu,
            log_every_n_steps=1,
        )
        trainer.validate(model, data_module)
        trainer.fit(model, data_module)
        trainer.test(model, data_module)

        if wandb_logger:
            wandb.finish()

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="mnist_fullbatch/svfl.yaml", help="The config file for the experiment.")
    parser.add_argument('--gpu', type=lambda s: [int(s)], required=True, help='GPU device id.')
    parser.add_argument('--seeds', type=int, nargs='+', help='List of seed values', required=True)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

    main(args)

# TODO redo plots and replace them in paper

# TODO in data/cifar10.py, match cifar10_transform to those used in paper
# TODO test ShallowSplitNN
# TODO test ResNetSplitNN
# TODO commit "implemented the cifar10"

# TODO change configs/cifar10/svfl.yaml
# TODO commit "matched the parameters of the cifar10 experiment"

# TODO currently SplitNN is specific to cross_entropy and SGD (I think this holds for all our exps, but should be generalized)
