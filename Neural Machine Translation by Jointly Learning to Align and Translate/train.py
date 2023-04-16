from __future__ import unicode_literals, print_function, division
import language as lang
from model import Encoder, Decoder, Model, SentenceData
import gc
import argparse
import utils as ut

import pytorch_lightning as pl
import torch
import torch.nn as nn
import mlflow

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gc.collect()
torch.cuda.empty_cache()


criterion = nn.NLLLoss()
optimizer = torch.optim.SGD

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--input_file')
    parser.add_argument('--hidden_size', default=256)
    parser.add_argument('--epochs', default=5)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--test_split', default=0.2)
    parser.add_argument('--num_workers', default=5)
    args = parser.parse_args()

    if args.input_file is not None:
        params = ut.read_yaml(args.input_file)['model_parameters']
        run_name = params['run_name']
        mlflow.set_tag('mlflow.runName', run_name)
        hidden_size = params['hidden_size']
        epochs = params['epochs']
        learning_rate = params['learning_rate']
        test_split = params['test_split']
        num_workers = params['num_workers']
        kwargs = {'num_workers': num_workers}
        
    else:
        run_name = args.run_name
        mlflow.set_tag('mlflow.runName', run_name)
        hidden_size = args.hidden_size
        epochs = args.epochs
        learning_rate = args.learning_rate
        test_split = args.test_split
        num_workers = args.num_workers
        kwargs = {'num_workers': num_workers}

    input_lang, output_lang, pairs = lang.prepareData('eng', 'fra', True)
    sentencedata = SentenceData(input_lang, output_lang)
    sentencedata.prepare_data(pairs=pairs, test_split=test_split)
    train_dataloader = sentencedata.train_dataloader(**kwargs)
    test_dataloader = sentencedata.test_dataloader(**kwargs)
    val_dataloader = sentencedata.val_dataloader(**kwargs)
    
    encoder = Encoder(input_lang.n_words, hidden_size=hidden_size)
    decoder = Decoder(hidden_size=hidden_size, output_size=output_lang.n_words)
    model = Model(input_lang=input_lang, output_lang=output_lang, 
                  encoder=encoder, decoder=decoder, criterion=criterion, 
                  optimizer=optimizer, learning_rate=learning_rate)
    
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)
    tblogger = pl.loggers.TensorBoardLogger('.', version = run_name)
    
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', logger=tblogger)
    trainer.fit(model=model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
