import time
from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


# Conexión con MLflow
# --------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment(
    "Practica Wine - Unai Orive"
)


# Datos
# --------------------------------------------------

wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Tablas con variables y objetivo
tabla_train = X_train.copy()
tabla_train["target"] = y_train

tabla_test = X_test.copy()
tabla_test["target"] = y_test


# Copias de los datos utilizadas por todas las pruebas
carpeta_datos = Path("datos_wine")
carpeta_datos.mkdir(exist_ok=True)

ruta_train = carpeta_datos / "wine_training.csv"
ruta_test = carpeta_datos / "wine_testing.csv"

tabla_train.to_csv(ruta_train, index=False)
tabla_test.to_csv(ruta_test, index=False)


# Datasets para registrar en MLflow
dataset_train = mlflow.data.from_pandas(
    tabla_train,
    targets="target",
    name="wine_training"
)

dataset_test = mlflow.data.from_pandas(
    tabla_test,
    targets="target",
    name="wine_testing"
)


# Configuraciones
# --------------------------------------------------

configuraciones = [
    {
        "n_neighbors": 3,
        "weights": "uniform",
        "p": 2
    },
    {
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2
    },
    {
        "n_neighbors": 7,
        "weights": "uniform",
        "p": 1
    }
]


# Entrenamientos
# --------------------------------------------------

for numero, config in enumerate(configuraciones, start=1):

    with mlflow.start_run(run_name=f"knn_prueba_{numero}"):

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                **config,
                algorithm="brute"
            ))
        ])


        # Entrenamiento y tiempo
        inicio = time.perf_counter()

        model.fit(X_train, y_train)

        tiempo_entrenamiento = (
            time.perf_counter() - inicio
        )


        # Predicciones
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)


        # Métricas
        accuracy_train = accuracy_score(
            y_train,
            pred_train
        )

        accuracy_test = accuracy_score(
            y_test,
            pred_test
        )

        f1_test = f1_score(
            y_test,
            pred_test,
            average="weighted"
        )


        # Parámetros
        mlflow.log_params(config)

        mlflow.log_params({
            "algorithm": "brute",
            "numero_features": X_train.shape[1],
            "numero_salidas": 1,
            "numero_clases": int(y_train.nunique()),
            "test_size": 0.25,
            "random_state": 42
        })


        # Métricas registradas
        mlflow.log_metrics({
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "f1_test": f1_test,
            "tiempo_entrenamiento_segundos": tiempo_entrenamiento
        })


        # Tags
        mlflow.set_tag(
            "modelo",
            "KNeighborsClassifier"
        )

        mlflow.set_tag(
            "preprocesado",
            "StandardScaler"
        )


        # Datasets
        mlflow.log_input(
            dataset_train,
            context="training"
        )

        mlflow.log_input(
            dataset_test,
            context="testing"
        )


        # Archivos de datos
        mlflow.log_artifact(
            str(ruta_train),
            artifact_path="datos"
        )

        mlflow.log_artifact(
            str(ruta_test),
            artifact_path="datos"
        )


        # Firma del modelo
        signature = infer_signature(
            X_train,
            pred_train
        )


        # Guardar y registrar el pipeline
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name="wine_knn"
        )


        # Mostrar resultados
        print(f"\nPrueba {numero}")
        print("Configuración:", config)
        print("Accuracy train:", accuracy_train)
        print("Accuracy test:", accuracy_test)
        print("F1 test:", f1_test)
        print("Tiempo:", tiempo_entrenamiento)
