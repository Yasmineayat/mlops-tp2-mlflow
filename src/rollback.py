
import mlflow
import pandas as pd
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Transaction a tester
transaction = pd.DataFrame([{
    "montant": 8500,
    "anciennete": 3,
    "freq_24h": 9,
    "nuit": 1,
}])

# Tester la version actuellement en production
modele = mlflow.sklearn.load_model("models:/fraude@production")
prediction = modele.predict(transaction)
print("Avant rollback :", prediction)

# Effectuer le rollback vers la version 2
client.set_registered_model_alias("fraude", "production", "2")

modele = mlflow.sklearn.load_model("models:/fraude@production")
prediction = modele.predict(transaction)
print("Apres rollback vers version 2 :", prediction)

# Restaurer la version 1 en production
client.set_registered_model_alias("fraude", "production", "1")

modele = mlflow.sklearn.load_model("models:/fraude@production")
prediction = modele.predict(transaction)
print("Apres restauration version 1 :", prediction)
