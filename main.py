import pandas as pd
import torch

from classifiers.LSTMClassifier import LSTMClassifier

from transformers import AutoModelForSequenceClassification

from utils.trainer import Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data.dataset import Dataset

from transformers import AutoTokenizer

import argparse

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return max([len(x) for x in self.data.values()])
    
    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}

def main():
    parser = argparse.ArgumentParser(description="Train a spam classification model.")

    parser.add_argument("input_data_path", type=str, help="Path to the input data csv file.")
    parser.add_argument("--do-test", action="store_true", default=False, help="Perform a test on the model.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "bert", "roberta", "t5", "deberta"], help="The model to use.")
    parser.add_argument("--embedding_size", type=int, default=64, help="The embedding size to use for the LSTM model.")
    parser.add_argument("--hidden_size", type=int, default=128, help="The hidden size to use for the LSTM model.")
    parser.add_argument("--num_layers", type=int, default=2, help="The number of layers to use for the LSTM model.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use for training.")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="The optimizer to use.")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate to use.")
    parser.add_argument("--momentum", type=float, default=0.9, help="The momentum to use for the optimizer.")
    parser.add_argument("--betas", type=str, default="(0.9, 0.999)", help="The betas to use for the Adam optimizer.")

    args = parser.parse_args()

    lr              = args.lr
    model_type      = args.model
    epochs          = args.epochs
    do_test         = args.do_test
    momentum        = args.momentum
    optimizer_type  = args.optimizer
    num_layers      = args.num_layers
    batch_size      = args.batch_size
    betas           = eval(args.betas)
    hidden_size     = args.hidden_size
    embedding_size  = args.embedding_size
    input_path      = args.input_data_path

    df = pd.read_csv(input_path)

    ohe_encoder = OneHotEncoder(sparse_output=False)

    X_data = df["text"].to_list()
    y_data = ohe_encoder.fit_transform(df["label"].values.reshape(-1, 1))

    X_train, X_val, y_train, y_val      = train_test_split(X_data,  y_data,  test_size=0.2, random_state=42, stratify=y_data)
    X_train, X_test, y_train, y_test    = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    model_name = {
        "bert": "google-bert/bert-base-uncased",
        "roberta": "cardiffnlp/twitter-roberta-base-emotion",
        "deberta": "microsoft/deberta-base",
        "t5": "google-t5/t5-small",
    }

    if model_type == "lstm":
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name[model_type])

    X_train = tokenizer(X_train,  padding=True,  return_tensors="pt", truncation=True)
    X_test  = tokenizer(X_test,   padding=True,  return_tensors="pt", truncation=True)
    X_val   = tokenizer(X_val,    padding=True,  return_tensors="pt", truncation=True)

    X_train["labels"] = torch.Tensor(y_train).float()
    X_test["labels"] = torch.Tensor(y_test).float()
    X_val["labels"] = torch.Tensor(y_val).float()

    if model_type == "lstm":
        model = LSTMClassifier(input_size=30522, embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name[model_type], 
            num_labels = 2,   
            output_attentions = False, 
            output_hidden_states = False
        )

    trainer = Trainer(
        model=model,
        train_dataset=CustomDataset(X_train),
        test_dataset=CustomDataset(X_test),
        val_dataset=CustomDataset(X_val),
    )

    trainer.train(batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum, betas=betas, optimizer_type=optimizer_type)

    if do_test:
        trainer.evaluate(batch_size=batch_size)

if __name__ == "__main__":
    main()