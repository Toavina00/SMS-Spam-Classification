import pandas as pd
import torch

from classifiers.LSTMClassifier import LSTMClassifier
from classifiers.BERTClassifier import BERTClassifier

from utils.trainer import Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from transformers import AutoTokenizer

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a spam classification model.")

    parser.add_argument("input_data_path", type=str, help="Path to the input data csv file.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "bert"], help="The model to use.")
    parser.add_argument("--do-test", action="store_true", default=False, help="Perform a test on the model.")
    parser.add_argument("--stop_words", action="store_true", default=False, help="Remove English stop words from the text data.")
    parser.add_argument("--max_df", type=float, default=1.0, help="The maximum document frequency for the BoW, Ngram and Tf-Idf models.")
    parser.add_argument("--min_df", type=float, default=1, help="The minimum document frequency for the BoW, Ngram and Tf-Idf models.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to use for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate to use.")
    parser.add_argument("--momentum", type=float, default=0.9, help="The momentum to use for the optimizer.")
    parser.add_argument("--betas", type=str, default="(0.9, 0.999)", help="The betas to use for the Adam optimizer.")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="The optimizer to use.")
    parser.add_argument("--embedding_size", type=int, default=64, help="The embedding size to use for the LSTM model.")
    parser.add_argument("--hidden_size", type=int, default=128, help="The hidden size to use for the LSTM model.")
    parser.add_argument("--num_layers", type=int, default=2, help="The number of layers to use for the LSTM model.")

    args = parser.parse_args()

    input_path = args.input_data_path
    max_df = args.max_df
    min_df = args.min_df
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    betas = eval(args.betas)
    optimizer_type = args.optimizer
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    stop_words = args.stop_words
    do_test    = args.do_test
    model_type = args.model

    df = pd.read_csv(input_path)

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    ohe_encoder = OneHotEncoder(sparse_output=False)

    X = df["text"].to_list()
    y = ohe_encoder.fit_transform(df["label"].values.reshape(-1, 1))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    X_train = tokenizer(X_train, return_tensors="pt", max_length=512, padding=True, truncation=True)
    X_test = tokenizer(X_test, return_tensors="pt", max_length=512, padding=True, truncation=True)
    X_val = tokenizer(X_val, return_tensors="pt", max_length=512, padding=True, truncation=True)



    model = LSTMClassifier(input_size=30522, embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=2)
    if model_type == "bert":
        model = BERTClassifier(2)

    trainer = Trainer(
        model=model,
        train_dataset=torch.utils.data.TensorDataset(
            X_train["input_ids"], X_train["token_type_ids"], X_train["attention_mask"], torch.Tensor(y_train).float()
        ),
        test_dataset=torch.utils.data.TensorDataset(
            X_test["input_ids"], X_test["token_type_ids"], X_test["attention_mask"], torch.Tensor(y_test).float()
        ),
        val_dataset=torch.utils.data.TensorDataset(
            X_val["input_ids"], X_val["token_type_ids"], X_val["attention_mask"], torch.Tensor(y_val).float()
        ),
    )

    trainer.train(batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum, betas=betas, optimizer_type=optimizer_type)

    if do_test:
        trainer.evaluate(batch_size=batch_size)

if __name__ == "__main__":
    main()