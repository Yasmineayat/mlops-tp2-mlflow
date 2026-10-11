import sys
import os
import json
import subprocess

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# Recuperer les parametres depuis la ligne de commande
n_estimators = int(sys.argv[1])
max_depth = int(sys.argv[2])

# Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp2-registre")

os.makedirs("reports", exist_ok=True)

# Charger les donnees
df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

# Entrainer et enregistrer l'experience
with mlflow.start_run(
    run_name=f"rf-{n_estimators}-{max_depth}"
):
    modele = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    modele.fit(Xtr, ytr)

    pred = modele.predict(Xte)
    proba = modele.predict_proba(Xte)[:, 1]

    # Parametres et metriques
    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    })

    f1 = f1_score(yte, pred)

    mlflow.log_metrics({
        "f1": f1,
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "roc_auc": roc_auc_score(yte, proba),
    })

    # Informations de tracabilite
    commit = subprocess.getoutput(
        "git rev-parse --short HEAD"
    )

    mlflow.set_tags({
        "auteur": "VOTRE-NOM",
        "commit": commit,
        "jeu_de_donnees": "transactions v1",
    })

    # Matrice de confusion
    ConfusionMatrixDisplay.from_predictions(yte, pred)
    plt.savefig(
        "reports/confusion.png",
        dpi=110,
        bbox_inches="tight",
    )
    plt.close()

    mlflow.log_artifact("reports/confusion.png")

    # Importance des variables
    importances = {
        colonne: round(float(valeur), 4)
        for colonne, valeur in zip(
            X.columns,
            modele.feature_importances_,
        )
    }

    with open(
        "reports/importances.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(importances, f, indent=2)

    mlflow.log_artifact("reports/importances.json")

    # Enregistrer le modele dans MLflow
    mlflow.sklearn.log_model(
        modele,
        name="model",
        input_example=Xte.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("F1 =", round(f1, 4))
