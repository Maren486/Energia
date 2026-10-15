import torch
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def calculate_confusion_matrix(dataloader, model, file=None):
    with torch.no_grad():
        labels, predictions = [], []
        for vdata in dataloader:
            vinputs, vtargets = vdata[0], vdata[1]
            labels.extend(vtargets)
            voutputs = model(torch.flatten(vinputs.to(torch.float), start_dim=1))
            predictions.extend(torch.argmax(voutputs, dim=1).numpy())
    f1 = metrics.f1_score(labels, predictions, average='micro')
    print(f'F1 score: {f1:.2%}')
    cm = metrics.confusion_matrix(labels, predictions)
    if file:
        cm_display = metrics.ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.savefig(file)
    return cm


mnist = MNIST(root='./data', download=True)

print(mnist.data.size())
print(mnist.targets.size())

BATCH_SIZE = 16

idx = torch.randperm(mnist.data.size()[0])

n = int(mnist.data.size()[0]*.9)

dataset_train = TensorDataset(mnist.data[idx[:n]], mnist.targets[idx[:n]])
dataset_val = TensorDataset(mnist.data[idx[n:]], mnist.targets[idx[n:]])
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)


INPUT_SIZE = 28*28  

LR = 0.01

EPOCHS = 50

PATIENCE = 5

BEST_MODEL_PATH = './mnist_best_model.pt'


model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_SIZE, 20),
    torch.nn.ReLU(20),
    torch.nn.Linear(20, 10)
)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.train()

writer = SummaryWriter() if SummaryWriter is not None else None

min_val_loss = float('inf')
count = 0
for epoch in range(EPOCHS):

    training_loss = 0.
    for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1} train"):
        optimizer.zero_grad()
        inputs, labels = batch
        pred = model(torch.flatten(inputs.to(torch.float), start_dim=1) / 255.)
        loss = criterion(pred, labels)
        training_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    training_loss /= len(dataloader_train)
    print(f'  Train loss: {training_loss:.4f}')
    if writer is not None:
        writer.add_scalar("Loss/train", training_loss, epoch+1)

    val_loss = 0.
    with torch.no_grad():
        labels_all, predictions_all = [], []
        for batch in dataloader_val:
            inputs, labels = batch
            pred = model(torch.flatten(inputs.to(torch.float), start_dim=1) / 255.)
            labels_all.extend(labels)
            predictions_all.extend(torch.argmax(pred, dim=1).numpy())
            loss = criterion(pred, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(dataloader_val)
    print(f'  Valid loss: {val_loss:.4f}')
    f1 = metrics.f1_score(labels_all, predictions_all, average='micro')
    print(f'F1 score: {f1:.2%}')
    if writer is not None:
        writer.add_scalar("Loss/val", val_loss, epoch+1)
        cm = metrics.confusion_matrix(labels_all, predictions_all)
        cm_display = metrics.ConfusionMatrixDisplay(cm)
        cm_display.plot()
        writer.add_figure(f"Confusion Matrix {epoch+1}", cm_display.figure_)
        plt.clf()
        writer.flush()
    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        count = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        count += 1
        print(f'  Not improving for {count}/{PATIENCE} epochs')
        if count >= PATIENCE:
            print('Stopping training')
            break

if writer is not None:
    writer.close()

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

print("\nValidation")
print(calculate_confusion_matrix(dataloader_val, model, file="cm_mnist_val.png"))
