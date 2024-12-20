import torch

from classifiers.LSTMClassifier import LSTMClassifier

from typing import Tuple, Literal
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, val_dataset):
        """
        Initializes a Trainer instance.

        Parameters
        ----------
        model : torch.nn.Module or sklearn.base.BaseEstimator
            The model to be trained.
        X_train : torch.Tensor or np.ndarray
            The input for training.
        y_train : torch.Tensor or np.ndarray
            The target for training.
        X_test : torch.Tensor or np.ndarray
            The input for testing.
        y_test : torch.Tensor or np.ndarray
            The target for testing.
        """
        
        self.train_dataset  = train_dataset
        self.test_dataset   = test_dataset
        self.val_dataset    = val_dataset
        self.model = model    

    def train(
        self,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 0.01,
        momentum: float = 0.9,
        betas: Tuple[float, float] = (0.9, 0.999),
        optimizer_type: Literal["sgd", "adam"] = "sdg",
    ):
        
        """
        Train the model.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for training. The default is 64.
        epochs : int, optional
            The number of epochs for training. The default is 10.
        lr : float, optional
            The learning rate for the optimizer. The default is 0.01.
        momentum : float, optional
            The momentum for the optimizer. The default is 0.9.
        betas : Tuple[float, float], optional
            The betas for the Adam optimizer. The default is (0.9, 0.999).
        optimizer_type : OptimizerType, optional
            The type of optimizer to use. The default is OptimizerType.SGD.

        Returns
        -------
        self
            The instance of the class.
        """

        if self.model is None: raise Exception("Model is not initialized")
        
        print("-" * 50)
        print("Training model...")
        print("-" * 50)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        trainloader = DataLoader(self.train_dataset,    batch_size=batch_size, shuffle=True)
        valloader   = DataLoader(self.val_dataset,      batch_size=batch_size, shuffle=False)

        if optimizer_type == "sgd":
            optimizer = optim.SGD(self.model.parameters(),  lr=lr, momentum=momentum)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

        for epoch in range(epochs):
            running_loss        = 0.0
            running_accuracy    = 0.0
            running_f1          = 0.0

            for i, inputs in enumerate(trainloader):
                loss    = None
                outputs = None

                if isinstance(self.model, LSTMClassifier):
                    text, labels  = inputs["input_ids"].to(device), inputs["labels"].to(device)
                    outputs       = self.model(text)
                    loss          = criterion(outputs, labels)
                else:
                    inputs      = {k: v.to(device) for k, v in inputs.items()}
                    outputs     = self.model(**inputs)
                    loss        = outputs.loss
                    outputs     = outputs.logits
                    labels      = inputs["labels"]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss        += loss.item()
                running_accuracy    += torch.sigmoid(outputs).argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
                running_f1          += f1_score(labels.argmax(dim=1).tolist(), torch.sigmoid(outputs).argmax(dim=1).tolist(), zero_division=1)
            
            print(f"Epoch {epoch+1} Train ~ loss: {running_loss / (i+1):.4f} accuracy: {running_accuracy / len(trainloader.dataset):.4f} f1_score: {running_f1 / (i+1):.4f}", end=" | ")

            running_loss        = 0.0
            running_accuracy    = 0.0
            running_f1          = 0.0

            with torch.no_grad():
                for i, inputs in enumerate(valloader):
                    loss    = None
                    outputs = None

                    if isinstance(self.model, LSTMClassifier):
                        text, labels  = inputs["input_ids"].to(device), inputs["labels"].to(device)
                        outputs       = self.model(text)
                        loss          = criterion(outputs, labels)
                    else:
                        inputs      = {k: v.to(device) for k, v in inputs.items()}
                        outputs     = self.model(**inputs)
                        loss        = outputs.loss
                        outputs     = outputs.logits
                        labels      = inputs["labels"]

                    running_loss        += loss.item()
                    running_accuracy    += torch.sigmoid(outputs).argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
                    running_f1          += f1_score(labels.argmax(dim=1).tolist(), torch.sigmoid(outputs).argmax(dim=1).tolist(), zero_division=1)
            
            print(f"Validation ~ loss: {running_loss / (i+1):.4f} accuracy: {running_accuracy / len(trainloader.dataset):.4f} f1_score: {running_f1 / (i+1):.4f}")

        return self.model

    def evaluate(self, batch_size: int = 64):

        """
        Evaluate the model on the test set.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for evaluation. The default is 64.

        Returns
        -------
        None
        """

        if self.model is None: raise Exception("Model is not initialized")

        print("-" * 50)
        print("Evaluating model...")
        print("-" * 50)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        testloader  = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        criterion   = nn.CrossEntropyLoss()
        
        running_loss        = 0.0
        running_accuracy    = 0.0
        running_f1          = 0.0

        with torch.no_grad():
            for i, inputs in enumerate(testloader):
                loss    = None
                outputs = None

                if isinstance(self.model, LSTMClassifier):
                    text, labels  = inputs["input_ids"].to(device), inputs["labels"].to(device)
                    outputs       = self.model(text)
                    loss          = criterion(outputs, labels)
                else:
                    inputs      = {k: v.to(device) for k, v in inputs.items()}
                    outputs     = self.model(**inputs)
                    loss        = outputs.loss
                    outputs     = outputs.logits
                    labels      = inputs["labels"]

                running_loss        += loss.item()
                running_accuracy    += torch.sigmoid(outputs).argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
                running_f1          += f1_score(labels.argmax(dim=1).tolist(), torch.sigmoid(outputs).argmax(dim=1).tolist(), zero_division=1)

        print(f"Test loss: {running_loss / (i+1):.4f} accuracy: {running_accuracy / len(testloader.dataset):.4f} f1_score: {running_f1 / (i+1):.4f}")
        
        

