import os
import torch
import logging
import random
import numpy as np
import json
from argparse import ArgumentParser, Namespace

from src import lightning
from src.collecting import collecting
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def set_global_seed(seed: int):
    """
    Reason from pytorch-lightning maintainer William Falcon:
    "setting the seed inside lightning seems like it hides too much according to ppl I’ve discussed with.
    i don’t want to have people wondering if lightning is doing anything with seeds.
    setting seeds and such is deferred to the user."
    proof: https://github.com/PyTorchLightning/pytorch-lightning/issues/37
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args() -> Namespace:

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_path', type=str, default='./data/checkpoint')
    parser.add_argument('--project_name', type=str, default='LightningDistillation')
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--distributed_backend', type=str, default='dp')
    parser.add_argument('--gpu', type=int, default=0 if torch.cuda.is_available() else None)
    parser.add_argument('--n_batch_accumulate', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)

    parser = lightning.LightningDistillation.add_model_specific_args(parser)

    parsed_args = parser.parse_args()

    return parsed_args


def train():

    logger = logging.getLogger(__file__)

    args = get_args()

    set_global_seed(args.seed)

    logging.basicConfig(level=logging.INFO)

    logger.info('Start collecting data')
    collecting(args.data_dir)

    model = lightning.LightningDistillation(args)

    try:
        with open('comet_api_key.txt') as f:
            comet_api_key = f.read().strip()
    except FileNotFoundError:
        comet_api_key = None

    # ddp don't work with comet logger
    if comet_api_key is not None and args.distributed_backend == 'dp':
        from pytorch_lightning.loggers import CometLogger
        pl_logger = CometLogger(api_key=comet_api_key,
                                project_name=args.project_name)
        pl_logger.experiment.log_parameters(args.__dict__)
        logger.info('Use CometML Logger')
    else:
        pl_logger = TensorBoardLogger(save_dir=os.path.join(os.getcwd(), args.data_dir),
                                      name=args.project_name)
        logger.info('Use TensorBoard Logger')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=f'distillation_{args.gpu}'
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    try:
        import apex
        use_amp = True
        precision = 16
    except ModuleNotFoundError:
        use_amp = False
        precision = 32
        logger.info('Train without amp, you can install it with command: make install-apex')

    with open(os.path.join(args.data_dir, 'hparams.json'), mode='w') as file_object:
        json.dump(args.__dict__, file_object)

    trainer = pl.Trainer(logger=pl_logger,
                         accumulate_grad_batches=args.n_batch_accumulate,
                         use_amp=use_amp,
                         precision=precision,
                         gradient_clip=args.max_norm,
                         distributed_backend=args.distributed_backend,
                         gpus=[args.gpu] if args.gpu is not None else None,
                         val_check_interval=0.25,
                         num_sanity_val_steps=0,
                         log_save_interval=10,
                         progress_bar_refresh_rate=10,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback)

    trainer.fit(model)


if __name__ == '__main__':
    train()
