"""

  - mnist_best_model.pt   -> pesos del mejor modelo de todos los intentos
  - search_results.csv    -> tabla resumen de cada configuracion probada
  - cm_mnist_val.png       -> matriz de confusion del mejor modelo en validacion

"""

import copy
import csv

import torch
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from model import build_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def calculate_confusion_matrix(dataloader, model, file=None):
    with torch.no_grad():
        labels, predictions = [], []
        for vdata in dataloader:
            vinputs, vtargets = vdata[0], vdata[1]
            labels.extend(vtargets)
            voutputs = model(torch.flatten(vinputs.to(torch.float), start_dim=1) / 255.)
            predictions.extend(torch.argmax(voutputs, dim=1).numpy())
    f1 = metrics.f1_score(labels, predictions, average='micro')
    print(f'F1 score: {f1:.2%}')
    cm = metrics.confusion_matrix(labels, predictions)
    if file:
        cm_display = metrics.ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.savefig(file)
        plt.close()
    return cm


def evaluate(dataloader, model, criterion):
    model.eval()
    total_loss = 0.
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            pred = model(torch.flatten(inputs.to(torch.float), start_dim=1) / 255.)
            loss = criterion(pred, labels)
            total_loss += loss.item() * inputs.size(0)
            labels_all.extend(labels)
            predictions_all.extend(torch.argmax(pred, dim=1).numpy())
    total_loss /= len(dataloader)
    f1 = metrics.f1_score(labels_all, predictions_all, average='micro')
    return total_loss, f1


def train_one_config(config, dataloader_train, dataloader_val, run_id, writer=None):
    """Entrena un modelo con una configuracion concreta usando early stopping."""
    torch.manual_seed(config.get("seed", 0))

    model = build_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    min_val_loss = float('inf')
    best_state = None
    count = 0
    history = []

    for epoch in range(config["epochs"]):
        model.train()
        training_loss = 0.
        desc = f"[{run_id}] lr={config['lr']} bs={config['batch_size']} epoch {epoch + 1}"
        for inputs, labels in tqdm(dataloader_train, desc=desc):
            optimizer.zero_grad()
            pred = model(torch.flatten(inputs.to(torch.float), start_dim=1) / 255.)
            loss = criterion(pred, labels)
            training_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()
        training_loss /= len(dataloader_train)

        val_loss, val_f1 = evaluate(dataloader_val, model, criterion)
        print(f'  [{run_id}] epoch {epoch + 1}: train_loss={training_loss:.4f} '
              f'val_loss={val_loss:.4f} val_f1={val_f1:.2%}')

        if writer is not None:
            writer.add_scalar(f"Loss/train_{run_id}", training_loss, epoch + 1)
            writer.add_scalar(f"Loss/val_{run_id}", val_loss, epoch + 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": training_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
        })

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            count = 0
        else:
            count += 1
            print(f'    Sin mejora {count}/{config["patience"]} epocas')
            if count >= config["patience"]:
                print(f'  [{run_id}] Parada temprana en epoca {epoch + 1}')
                break

    return best_state, min_val_loss, history


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

mnist = MNIST(root='./data', download=True)

print(mnist.data.size())
print(mnist.targets.size())

idx = torch.randperm(mnist.data.size()[0])
n = int(mnist.data.size()[0] * .9)

BATCH_SIZE_DEFAULT = 16
dataset_train = TensorDataset(mnist.data[idx[:n]], mnist.targets[idx[:n]])
dataset_val = TensorDataset(mnist.data[idx[n:]], mnist.targets[idx[n:]])

# --------------------------------------------------------------------------
# Configuraciones a probar: varias combinaciones de epocas/lr/batch size.
# Amplia o reduce esta lista segun lo que os pida el profesor.
# --------------------------------------------------------------------------

CONFIGS = [
    {"lr": 0.01,  "batch_size": 16, "epochs": 50, "patience": 5, "seed": 0},
    {"lr": 0.01,  "batch_size": 64, "epochs": 50, "patience": 5, "seed": 0},
    {"lr": 0.001, "batch_size": 16, "epochs": 50, "patience": 5, "seed": 0},
    {"lr": 0.001, "batch_size": 64, "epochs": 50, "patience": 5, "seed": 0},
]

BEST_MODEL_PATH = './mnist_best_model.pt'
RESULTS_CSV = './search_results.csv'

writer = SummaryWriter() if SummaryWriter is not None else None

global_best_val_loss = float('inf')
global_best_state = None
global_best_config = None
results_rows = []

for i, config in enumerate(CONFIGS, start=1):
    dataloader_train = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config["batch_size"])

    run_id = f"run{i}"
    best_state, best_val_loss, history = train_one_config(
        config, dataloader_train, dataloader_val, run_id, writer=writer
    )

    results_rows.append({
        "run": run_id,
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "epochs_run": history[-1]["epoch"] if history else 0,
        "best_val_loss": best_val_loss,
        "final_val_f1": history[-1]["val_f1"] if history else None,
    })

    if best_val_loss < global_best_val_loss:
        global_best_val_loss = best_val_loss
        global_best_state = best_state
        global_best_config = config

if writer is not None:
    writer.close()

# --------------------------------------------------------------------------
# Guardar resultados y mejor modelo
# --------------------------------------------------------------------------

with open(RESULTS_CSV, "w", newline="") as f:
    fieldnames = ["run", "lr", "batch_size", "epochs_run", "best_val_loss", "final_val_f1"]
    writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
    writer_csv.writeheader()
    writer_csv.writerows(results_rows)

print("\nResumen de todos los intentos:")
for row in results_rows:
    print(row)

print(f"\nMejor configuracion: {global_best_config} "
      f"(val_loss={global_best_val_loss:.4f})")

torch.save(global_best_state, BEST_MODEL_PATH)
print(f"Pesos del mejor modelo guardados en {BEST_MODEL_PATH}")

# --------------------------------------------------------------------------
# Evaluacion final del mejor modelo con matriz de confusion
# --------------------------------------------------------------------------

best_model = build_model()
best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
best_model.eval()

dataloader_val_final = DataLoader(dataset_val, batch_size=BATCH_SIZE_DEFAULT)

print("\nValidacion (mejor modelo)")
print(calculate_confusion_matrix(dataloader_val_final, best_model, file="cm_mnist_val.png"))
