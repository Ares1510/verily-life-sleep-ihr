import os
import time
from utils.parser import parser
from utils.data import SleepDataLoader
from models import HeartRateNetwork
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def main():
    args = parser()
    pl.seed_everything(42)
    sub_ids = [x.split('_')[0] for x in os.listdir('data')]
    data_loader = SleepDataLoader(sub_ids)
    train_loader = data_loader.train(batch_size=args.batch_size)
    val_loader = data_loader.val(batch_size=args.batch_size)
    test_loader = data_loader.test(batch_size=args.batch_size)

    callbacks = [EarlyStopping(monitor='val_BinaryF1Score', patience=args.patience, mode='max', verbose=True),
                 ModelCheckpoint(monitor='val_BinaryF1Score', mode='max')]

    run_name = 'MESA_' + time.strftime("%Y%m%d-%H%M%S")
    model = HeartRateNetwork(lr=args.lr, weight_decay=args.weight_decay)

    logger = WandbLogger(project='sleep-ihr', name=run_name, log_model='all')
    logger.experiment.config.update(args)

    trainer = Trainer(accelerator='gpu', max_epochs=args.epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()