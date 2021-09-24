# Data Engineering and Modelling using pytorch

## Process of data engineering

## Code

- Write pytorch code for tabular data set
- Import libraries

```
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```

```
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

```
print(torch.__version__)
```

```
import pandas as pd
import seaborn as sn
```

```
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

```
df = pd.read_csv('Sample_IssueDataset.csv') 
```

```
df.head()
```

```
df1 = pd.get_dummies(df)
```

```
df1.head()
```

```
y = df1.iloc[:,1]
X = df1.iloc[:,:18]
```

```
X = X.drop(columns=['EmployeeLeft'])
```

```
from sklearn.model_selection import train_test_split
```

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

```
X_train_torch = torch.tensor(X_train.values)
X_test_torch = torch.tensor(X_test.values)
y_train_torch = torch.tensor(y_train.values)
y_test_torch = torch.tensor(y_test.values)
```

```
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
```

```
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9380*17, 10),
            nn.ReLU(),
            #nn.Linear(9380*17, 10),
            #nn.ReLU(),
            nn.Linear(9380*17, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

```
model = NeuralNetwork().to(device)
print(model)
```

```
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

```
import torch
 
train_features = torch.tensor(X_train.to_numpy())
train_labels = torch.tensor(y_train.to_numpy())
 
validation_features = torch.tensor(X_test.to_numpy())
validation_labels = torch.tensor(y_test.to_numpy())
```

```
n_features = X_train.shape[1]
# 31
model = torch.nn.Sequential(torch.nn.Linear(n_features, 18),
                            torch.nn.ReLU(),
                            torch.nn.Linear(18, 1),
                            torch.nn.Sigmoid())
```

```
criterion = torch.nn.BCELoss()
 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
```

```
X_train_torch.shape
y_train_torch.shape
```

```
n_batches = 2
train_features_batched = train_features.reshape(n_batches,
                                               int(train_features.shape[0]/n_batches),
                                               train_features.shape[1])
train_labels_batched = train_labels.reshape(n_batches,
                                            int(train_labels.shape[0]/n_batches))

```

```
n_epochs = 100
loss_list = []
validate_loss_list = []
 
for epoch in range(n_epochs):
    for batch_idx in range(n_batches):
        optimizer.zero_grad()
         
        outputs = model(train_features_batched[batch_idx].float())
         
     
        loss = criterion(outputs.flatten().float(),
                         train_labels_batched[batch_idx].float())
     
         
        loss.backward()
         
        optimizer.step()
         
    outputs = model(train_features.float())
     
    validation_outputs = model(validation_features.float())
     
         
    loss = criterion(outputs.flatten().float(),
                     train_labels.float())
     
    validate_loss = criterion(validation_outputs.flatten().float(),
                              validation_labels.float())
     
    loss_list.append(loss.item())
     
    validate_loss_list.append(validate_loss)
 
print('Finished Training')
 
import matplotlib.pyplot as plt
plt.plot(loss_list, linewidth=3)
plt.plot(validate_loss_list, linewidth=3)
plt.legend(("Training Loss", "Validation Loss"))
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
```

```
y_pred = model(validation_features[1].flatten().float())
print(y_pred)
```

```
import matplotlib.pyplot as plt
import numpy as np
```

```
type(validation_features.flatten().float())
```

```
model.eval()
```

```
y_pred = model(validation_features[0].flatten().float())
```

```
len(y_pred)
```

```
len(validation_features[0].flatten().float())
```

```
model.train()
epochs = 5
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(train_features.float())
    # Compute Loss
    loss = criterion(y_pred.squeeze(), train_labels.float())
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()
```

```
model.train()
epochs = 500
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(validation_features.float())
    # Compute Loss
    loss = criterion(y_pred.squeeze(), validation_labels.float())
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()
```

```
import matplotlib.pyplot as plt
import numpy as np
def plotcharts(errors):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(train_labels.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    plt.legend(loc=7)
    plt.show()
plotcharts(errors)
```

```
model.eval()
y_pred = model(validation_features.float())
after_train = criterion(y_pred.squeeze(), validation_labels.float())
print('Test loss after Training' , after_train.item())
```

```
import matplotlib.pyplot as plt
import numpy as np
def plotcharts(errors):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(train_labels.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    plt.legend(loc=7)
    plt.show()
plotcharts(errors)
```

```
probs = torch.sigmoid(y_pred)
print(probs)
```

- More to come