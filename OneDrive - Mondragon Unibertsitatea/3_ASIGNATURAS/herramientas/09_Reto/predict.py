import mlflow
import mlflow.sklearn


# 1. CONEXIÓN CON MLFLOW
mlflow.set_tracking_uri(
    "http://localhost:5000"
)



# 2. MODELO QUE QUEREMOS UTILIZAR

# Este nombre debe coincidir con el nombre que demos
# al modelo cuando lo registremos en MLflow.

model_uri = "models:/iris_random_forest@champion"


# 3. CARGAR MODELO

model = mlflow.sklearn.load_model(
    model_uri
)


# 4. NUEVO DATO

# Medidas de una flor Iris:
#
# longitud sépalo
# anchura sépalo
# longitud pétalo
# anchura pétalo

nueva_flor = [[
    5.1,
    3.5,
    1.4,
    0.2
]]


# 5. HACER PREDICCIÓN

prediccion = model.predict(
    nueva_flor
)


print(
    "Clase predicha:",
    prediccion[0]
)