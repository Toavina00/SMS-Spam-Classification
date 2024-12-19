from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, num_classes):
        
        """
        Initializes the LSTMClassifier model.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input.
        embedding_size : int
            The size of the embedding layer.
        hidden_size : int
            The number of features in the hidden state of the LSTM.
        num_layers : int
            The number of recurrent layers in the LSTM.
        num_classes : int
            The number of classes for the output layer.
        """

        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        
        """
        The forward pass of the LSTMClassifier model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        x = self.embedding(x)
        h, _ = self.lstm(x)
        x = self.fc(h[:, -1, :])
        return x