import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics 

iris = datasets.load_iris()

iris_model = nn.Sequential(
    nn.Linear(iris.data.shape[1], 20),
    nn.ReLU(),
    nn.Linear(20, len(iris.target_names))
)

print(iris_model)

idx = torch.randperm(150)

iris_train_data = iris.data[idx[:120]]
iris_train_labels = iris.target[idx[:120]]

iris_val_data = iris.data[idx[120:135]]
iris_val_labels = iris.target[idx[120:135]]

iris_test_data = iris.data[idx[135:]]
iris_test_labels = iris.target[idx[135:]]

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(iris_model.parameters(), lr=0.01)

train_dataset = TensorDataset(torch.tensor(iris_train_data, dtype=torch.float), torch.tensor(iris_train_labels, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(iris_val_data, dtype=torch.float), torch.tensor(iris_val_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(iris_test_data, dtype=torch.float), torch.tensor(iris_test_labels, dtype=torch.long))

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False)

epochs = 50

for epoch in range(epochs):
    train_loss = 0.
    for batch in train_dataloader:
        optim.zero_grad()
        x, y = batch
        y_pred = iris_model(x)
        # WRONG: loss = criterion(y, y_pred) 
        loss = criterion(y_pred, y)
        loss.backward()
        train_loss += loss.item() * x.size(0)
        optim.step()
    train_loss /= iris_train_data.shape[0]
    val_loss = 0.
    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            y_pred = iris_model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item() * x.size(0)
    val_loss /= len(val_dataloader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")


def calculate_confusion_matrix(test_dataloader, model, file=None):
    with torch.no_grad():
        labels, predictions = [], []
        for vdata in test_dataloader:
            vinputs, vtargets = vdata[0], vdata[1]
            labels.extend(vtargets)
            voutputs = model(vinputs)
            predictions.extend(torch.argmax(voutputs, dim=1).numpy())
    cm = metrics.confusion_matrix(labels, predictions)
    if file:
        cm_display = metrics.ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.savefig(file)
    return cm


print("\nValidation")
print(calculate_confusion_matrix(val_dataloader, iris_model, file="cm_iris_val.png"))
print("\nTest")
print(calculate_confusion_matrix(test_dataloader, iris_model, file="cm_iris_test.png"))
