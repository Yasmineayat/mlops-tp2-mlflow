
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Associer les alias aux versions
client.set_registered_model_alias(
    "fraude", "staging", "2"
)

client.set_registered_model_alias(
    "fraude", "production", "1"
)

# Ajouter une description a la version 1
client.update_model_version(
    name="fraude",
    version="1",
    description="Version initiale du modele de detection de fraude",
)

# Verifier les alias
for alias in ["staging", "production"]:
    mv = client.get_model_version_by_alias("fraude", alias)
    print(alias, "-> version", mv.version)
