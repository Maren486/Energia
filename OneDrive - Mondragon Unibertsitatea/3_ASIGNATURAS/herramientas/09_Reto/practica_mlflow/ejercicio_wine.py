from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Cargar los datos
wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

# Separar entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
# Configuración del modelo
config = {
    "n_neighbors": 3,
    "weights": "uniform",
    "p": 2
}

# Escalado y clasificación
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        **config,
        algorithm="brute"
    ))
])

# Entrenamiento
model.fit(X_train, y_train)

# Predicciones
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Métricas
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)
f1_test = f1_score(
    y_test,
    pred_test,
    average="weighted"
)
print("Accuracy train:", accuracy_train)
print("Accuracy test:", accuracy_test)
print("F1 test:", f1_test)
