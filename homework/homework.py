# flake8: noqa: E501
"""
Credit Card Default Prediction Model
Clasificación de default de pagos usando SVM con pipeline optimizado
"""

import gzip
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def load_and_clean_data():
    """
    Paso 1: Cargar y limpiar los datasets
    """
    # Cargar datos
    train_data = pd.read_csv(
        "files/input/train_data.csv.zip",
        compression="zip",
        index_col=False,
    )
    test_data = pd.read_csv(
        "files/input/test_data.csv.zip",  # CORREGIDO: quitado el .csv duplicado
        compression="zip",
        index_col=False,
    )

    # Renombrar columna objetivo
    train_data = train_data.rename(
        columns={"default payment next month": "default"}
    )
    test_data = test_data.rename(
        columns={"default payment next month": "default"}
    )

    # Remover columna ID
    if "ID" in train_data.columns:
        train_data = train_data.drop(columns=["ID"])
    if "ID" in test_data.columns:
        test_data = test_data.drop(columns=["ID"])

    # Eliminar registros con información no disponible
    # EDUCATION: 0 es N/A
    # MARRIAGE: 0 es N/A
    train_data = train_data[
        (train_data["EDUCATION"] != 0) & (train_data["MARRIAGE"] != 0)
    ]
    test_data = test_data[
        (test_data["EDUCATION"] != 0) & (test_data["MARRIAGE"] != 0)
    ]

    # Agrupar EDUCATION > 4 en categoría "others" (4)
    train_data.loc[train_data["EDUCATION"] > 4, "EDUCATION"] = 4
    test_data.loc[test_data["EDUCATION"] > 4, "EDUCATION"] = 4

    return train_data, test_data


def split_features_target(train_data, test_data):
    """
    Paso 2: Dividir datasets en X e y
    """
    x_train = train_data.drop(columns=["default"])
    y_train = train_data["default"]

    x_test = test_data.drop(columns=["default"])
    y_test = test_data["default"]

    return x_train, y_train, x_test, y_test


def create_pipeline():
    """
    Paso 3: Crear pipeline de clasificación
    """
    # Identificar columnas categóricas
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    # Crear transformador para one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    # Crear pipeline optimizado
    pipeline = Pipeline(
        [
            ("onehotencoder", preprocessor),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("selectkbest", SelectKBest(score_func=f_classif)),
            ("svc", SVC(random_state=42, class_weight="balanced")),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimizar hiperparámetros usando GridSearchCV
    """
    # Definir grilla de hiperparámetros más amplia y optimizada
    param_grid = {
        "pca__n_components": [0.85, 0.90, 0.95, None],
        "selectkbest__k": [10, 15, 20, 25, "all"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__kernel": ["rbf", "linear"],
        "svc__gamma": ["scale", "auto", 0.1, 0.01],
    }

    # Usar StratifiedKFold para mejor manejo de clases desbalanceadas
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Crear GridSearchCV optimizado
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",  # Cambiar a f1 para mejor balance entre precision y recall
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    # Ajustar modelo
    print("Entrenando modelo con validación cruzada...")
    grid_search.fit(x_train, y_train)

    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score (f1): {grid_search.best_score_:.4f}")

    return grid_search


def save_model(model, filename):
    """
    Paso 5: Guardar modelo comprimido
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with gzip.open(filename, "wb") as file:
        pickle.dump(model, file)

    print(f"Modelo guardado en: {filename}")


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6: Calcular métricas de clasificación
    """
    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Métricas para conjunto de entrenamiento
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_train, y_train_pred)
        ),
        "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
        "f1_score": float(f1_score(y_train, y_train_pred, zero_division=0)),
    }

    # Métricas para conjunto de prueba
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
    }

    return train_metrics, test_metrics, y_train_pred, y_test_pred


def calculate_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred):
    """
    Paso 7: Calcular matrices de confusión
    """
    # Matriz de confusión para entrenamiento
    cm_train = confusion_matrix(y_train, y_train_pred)
    train_cm = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }

    # Matriz de confusión para prueba
    cm_test = confusion_matrix(y_test, y_test_pred)
    test_cm = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }

    return train_cm, test_cm


def save_metrics(metrics_list, filename):
    """
    Guardar métricas en archivo JSON
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as file:
        for metrics in metrics_list:
            file.write(json.dumps(metrics) + "\n")

    print(f"Métricas guardadas en: {filename}")


def main():
    """
    Función principal que ejecuta todo el flujo
    """
    print("=" * 70)
    print("CREDIT CARD DEFAULT PREDICTION MODEL")
    print("=" * 70)

    # Paso 1: Cargar y limpiar datos
    print("\n1. Cargando y limpiando datos...")
    train_data, test_data = load_and_clean_data()
    print(f"   Train shape: {train_data.shape}")
    print(f"   Test shape: {test_data.shape}")

    # Paso 2: Dividir en X e y
    print("\n2. Dividiendo features y target...")
    x_train, y_train, x_test, y_test = split_features_target(
        train_data, test_data
    )

    # Paso 3: Crear pipeline
    print("\n3. Creando pipeline...")
    pipeline = create_pipeline()

    # Paso 4: Optimizar hiperparámetros
    print("\n4. Optimizando hiperparámetros...")
    model = optimize_hyperparameters(pipeline, x_train, y_train)

    # Paso 5: Guardar modelo
    print("\n5. Guardando modelo...")
    save_model(model, "files/models/model.pkl.gz")

    # Paso 6: Calcular métricas
    print("\n6. Calculando métricas...")
    train_metrics, test_metrics, y_train_pred, y_test_pred = calculate_metrics(
        model, x_train, y_train, x_test, y_test
    )

    # Paso 7: Calcular matrices de confusión
    print("\n7. Calculando matrices de confusión...")
    train_cm, test_cm = calculate_confusion_matrices(
        y_train, y_train_pred, y_test, y_test_pred
    )

    # Guardar todas las métricas
    all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics(all_metrics, "files/output/metrics.json")

    # Resumen de resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"\nTRAIN - Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}")
    print(f"        Precision: {train_metrics['precision']:.4f}")
    print(f"        Recall: {train_metrics['recall']:.4f}")
    print(f"        F1-Score: {train_metrics['f1_score']:.4f}")

    print(f"\nTEST  - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"        Precision: {test_metrics['precision']:.4f}")
    print(f"        Recall: {test_metrics['recall']:.4f}")
    print(f"        F1-Score: {test_metrics['f1_score']:.4f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()