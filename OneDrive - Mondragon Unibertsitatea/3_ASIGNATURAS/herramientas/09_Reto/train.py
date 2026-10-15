import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# 1. CONFIGURACIÓN DE MLFLOW

# Dirección donde está ejecutándose el servidor de MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Nombre del experimento donde se guardarán las ejecuciones
mlflow.set_experiment("Clasificacion Iris")


# 2. CARGA DE DATOS

iris = load_iris()

X = iris.data
y = iris.target


# Dividimos los datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# 3. CONFIGURACIONES QUE VAMOS A PROBAR

configuraciones = [
    {
        "n_estimators": 10,
        "max_depth": 2
    },
    {
        "n_estimators": 50,
        "max_depth": 3
    },
    {
        "n_estimators": 100,
        "max_depth": 5
    },
    {
        "n_estimators": 200,
        "max_depth": None
    }
]


# 4. ENTRENAMIENTO DE LOS DIFERENTES MODELOS

for config in configuraciones:

    # Cada start_run representa una ejecución diferente
    with mlflow.start_run():

        
        # Crear el modelo
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42
        )


        
        # Entrenar
        model.fit(
            X_train,
            y_train
        )


        
        # Predicción
        predictions = model.predict(
            X_test
        )


        
        # Calcular métrica
        accuracy = accuracy_score(
            y_test,
            predictions
        )


        
        # 5. REGISTRAR INFORMACIÓN EN MLFLOW
        
        # Parámetros
        mlflow.log_param(
            "n_estimators",
            config["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            config["max_depth"]
        )


        # Métrica
        mlflow.log_metric(
            "accuracy",
            accuracy
        )


        
        # Guardar matriz de confusión como artefacto
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions
        )

        nombre_imagen = (
            f"matriz_confusion_"
            f"{config['n_estimators']}_"
            f"{config['max_depth']}.png"
        )

        plt.savefig(
            nombre_imagen
        )

        plt.close()


        mlflow.log_artifact(
            nombre_imagen
        )


        
        # Guardar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )


        
        # Mostrar resultado por terminal
        print(
            f"n_estimators={config['n_estimators']} | "
            f"max_depth={config['max_depth']} | "
            f"accuracy={accuracy:.3f}"
        )