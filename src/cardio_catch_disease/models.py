"""Training, evaluation and batch prediction entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

from .config import load_config, resolve_project_path
from .data import load_training_frame, read_csv
from .features import model_matrix, prepare_features


def _one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _column_types(X):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return numeric, categorical


def _preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric, categorical = _column_types(X)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def _classifier(config: dict):
    from sklearn.linear_model import LogisticRegression

    modeling = config.get("modeling", {})
    class_weight = modeling.get("class_weight", "balanced")
    return LogisticRegression(
        max_iter=int(modeling.get("max_iter", 1000)),
        class_weight=class_weight,
        n_jobs=-1,
        random_state=int(modeling.get("random_state", 42)),
    )


def _regressor(config: dict):
    from sklearn.ensemble import RandomForestRegressor

    modeling = config.get("modeling", {})
    return RandomForestRegressor(
        n_estimators=int(modeling.get("n_estimators", 160)),
        min_samples_leaf=int(modeling.get("min_samples_leaf", 2)),
        n_jobs=-1,
        random_state=int(modeling.get("random_state", 42)),
    )


def _candidate_classifiers(config: dict) -> list[tuple[str, object]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    modeling = config.get("modeling", {})
    rs = int(modeling.get("random_state", 42))
    cw = modeling.get("class_weight", "balanced")

    candidates = [
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight=cw, n_jobs=-1, random_state=rs)),
        ("random_forest", RandomForestClassifier(n_estimators=100, class_weight=cw, n_jobs=-1, random_state=rs)),
    ]
    try:
        from xgboost import XGBClassifier

        candidates.append(
            ("xgboost", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=rs, n_jobs=-1))
        )
    except ImportError:
        pass
    return candidates


def _cross_val_comparison(X_train, y_train, preprocessor, candidates: list, cv: int = 5, scoring: str = "roc_auc") -> dict:
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    results = {}
    for name, estimator in candidates:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {"mean": float(scores.mean()), "std": float(scores.std()), "cv_scores": scores.tolist()}
    return results


def _tune_random_forest(X_train, y_train, preprocessor, config: dict):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    modeling = config.get("modeling", {})
    rs = int(modeling.get("random_state", 42))
    cw = modeling.get("class_weight", "balanced")

    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(class_weight=cw, n_jobs=-1, random_state=rs)),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=int(modeling.get("n_search_iter", 20)),
        scoring="roc_auc",
        cv=5,
        random_state=rs,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _feature_importance(model, X) -> dict:
    try:
        step = model.named_steps.get("model") or model.named_steps.get("estimator")
        preprocessor = model.named_steps.get("preprocess")
        if step is None or preprocessor is None:
            return {}
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out().tolist()
        else:
            feature_names = X.columns.tolist()
        if hasattr(step, "feature_importances_"):
            importances = step.feature_importances_.tolist()
        elif hasattr(step, "coef_"):
            import numpy as np
            importances = np.abs(step.coef_[0]).tolist() if step.coef_.ndim > 1 else np.abs(step.coef_).tolist()
        else:
            return {}
        if len(feature_names) != len(importances):
            return {}
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return {name: round(imp, 6) for name, imp in ranked[:20]}
    except Exception:
        return {}


def _pipeline(X, estimator):
    from sklearn.pipeline import Pipeline

    return Pipeline(steps=[("preprocess", _preprocessor(X)), ("model", estimator)])


def _classification_metrics(y_true, y_pred, y_proba, config: dict):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = sorted(set(y_true.dropna().tolist()))
    average = "binary" if len(labels) == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None and len(labels) == 2:
        positive = config.get("data", {}).get("positive_label", labels[-1])
        classes = list(getattr(y_proba, "classes_", []))
        scores = y_proba
        if hasattr(y_proba, "shape") and len(y_proba.shape) == 2:
            pos_idx = 1
            scores = y_proba[:, pos_idx]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["average_precision"] = float(average_precision_score(y_true, scores))
        except ValueError:
            pass
        for k in config.get("evaluation", {}).get("top_k", []):
            k = min(int(k), len(scores))
            if k <= 0:
                continue
            order = np.argsort(scores)[::-1][:k]
            positive_mask = (y_true.reset_index(drop=True).iloc[order] == positive).astype(int)
            base_rate = float((y_true == positive).mean())
            precision_at_k = float(positive_mask.mean())
            metrics[f"precision_at_{k}"] = precision_at_k
            metrics[f"lift_at_{k}"] = precision_at_k / base_rate if base_rate else None
    return metrics


def _regression_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(y_true == 0, np.nan, y_true)
    rmspe = np.sqrt(np.nanmean(((y_true - y_pred) / denominator) ** 2))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": float(rmspe),
    }


def _try_log_mlflow(config: dict, model, metrics: dict, model_path) -> str | None:
    if not config.get("modeling", {}).get("use_mlflow", False):
        return None
    try:
        import mlflow
    except ImportError:
        return "MLflow is not installed. Install requirements-mlflow.txt to enable tracking."

    experiment = config.get("modeling", {}).get("mlflow_experiment", config.get("project", {}).get("slug", "kaggle_project"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=config.get("project", {}).get("slug", "training")):
        mlflow.log_params(
            {
                "problem_type": config.get("project", {}).get("problem_type"),
                "family": config.get("project", {}).get("family"),
                "target": config.get("data", {}).get("target", ""),
            }
        )
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        mlflow.log_artifact(str(model_path))
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass
    return None


def train_supervised(config: dict):
    import joblib
    from sklearn.model_selection import train_test_split

    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column is not available in the training data.")

    modeling = config.get("modeling", {})
    problem_type = config.get("project", {}).get("problem_type", "binary_classification")
    stratify = y if problem_type in {"binary_classification", "multiclass_classification", "ranking"} else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        stratify=stratify,
    )

    preprocessor = _preprocessor(X_train)

    if problem_type == "regression":
        estimator = _regressor(config)
        model = _pipeline(X_train, estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        metrics = _regression_metrics(y_valid, y_pred)
    else:
        # Compare candidate classifiers with cross-validation
        candidates = _candidate_classifiers(config)
        cv_results = _cross_val_comparison(X_train, y_train, preprocessor, candidates)

        best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
        print(f"[train] Cross-validation results (ROC-AUC):")
        for name, res in cv_results.items():
            print(f"  {name}: {res['mean']:.4f} ± {res['std']:.4f}")
        print(f"[train] Best model: {best_name}")

        # Tune the best model if it's RandomForest, else use as-is
        if best_name == "random_forest" and modeling.get("tune", True):
            model, best_params, best_cv_score = _tune_random_forest(X_train, y_train, preprocessor, config)
            print(f"[train] Best tuned ROC-AUC: {best_cv_score:.4f}")
        else:
            best_estimator = dict(candidates)[best_name]
            from sklearn.pipeline import Pipeline
            model = Pipeline(steps=[("preprocess", preprocessor), ("model", best_estimator)])
            model.fit(X_train, y_train)
            best_params = {}

        y_pred = model.predict(X_valid)
        y_proba = model.predict_proba(X_valid) if hasattr(model, "predict_proba") else None
        metrics = _classification_metrics(y_valid, y_pred, y_proba, config)
        metrics["model_selected"] = best_name
        metrics["cv_comparison"] = cv_results
        metrics["feature_importance"] = _feature_importance(model, X_train)
        if best_params:
            metrics["best_params"] = {k: str(v) for k, v in best_params.items()}

    model_path = resolve_project_path(config, modeling.get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    mlflow_note = _try_log_mlflow(config, model, metrics, model_path)
    if mlflow_note:
        metrics["mlflow_note"] = mlflow_note
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "metrics": metrics}


def train_clustering(config: dict):
    import json
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.pipeline import Pipeline

    df = load_training_frame(config)
    X, _, prepared = model_matrix(df, config, training=True)
    preprocessor = _preprocessor(X)
    X_matrix = preprocessor.fit_transform(X)
    cluster_range = config.get("modeling", {}).get("cluster_range", [3, 4, 5, 6, 7])
    sample_size = int(config.get("modeling", {}).get("silhouette_sample_size", 10000))
    results = []
    best = None
    for k in cluster_range:
        model = KMeans(n_clusters=int(k), n_init="auto", random_state=int(config.get("modeling", {}).get("random_state", 42)))
        labels = model.fit_predict(X_matrix)
        if len(labels) > sample_size:
            rng = np.random.default_rng(int(config.get("modeling", {}).get("random_state", 42)))
            idx = rng.choice(len(labels), size=sample_size, replace=False)
            sil = silhouette_score(X_matrix[idx], labels[idx])
        else:
            sil = silhouette_score(X_matrix, labels)
        metrics = {
            "k": int(k),
            "silhouette": float(sil),
            "calinski_harabasz": float(calinski_harabasz_score(X_matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_matrix, labels)),
        }
        results.append(metrics)
        if best is None or metrics["silhouette"] > best["silhouette"]:
            best = metrics

    selected_model = KMeans(
        n_clusters=best["k"],
        n_init="auto",
        random_state=int(config.get("modeling", {}).get("random_state", 42)),
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", selected_model)])
    labels = pipeline.fit_predict(X)
    prepared = prepared.copy()
    prepared["cluster"] = labels

    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    clusters_path = resolve_project_path(config, "reports/cluster_assignments.csv")
    for path in [model_path, metrics_path, clusters_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps({"selected": best, "trials": results}, indent=2), encoding="utf-8")
    prepared.to_csv(clusters_path, index=False)
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "clusters_path": str(clusters_path), "metrics": best}


def train(config_path: str | Path | None = None):
    config = load_config(config_path)
    problem_type = config.get("project", {}).get("problem_type")
    if problem_type == "clustering":
        return train_clustering(config)
    if problem_type in {"ab_testing", "price_elasticity"}:
        from .analysis import run_analysis

        return run_analysis(config_path)
    return train_supervised(config)


def predict(config_path: str | Path | None, input_path: str | Path, output_path: str | Path | None = None):
    import joblib
    import pandas as pd

    config = load_config(config_path)
    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    model = joblib.load(model_path)
    raw = read_csv(input_path)
    prepared = prepare_features(raw, config, training=False)
    X, _, _ = model_matrix(prepared, config, training=False)
    result = raw.copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            result["score"] = proba[:, 1]
        else:
            for idx, klass in enumerate(model.classes_):
                result[f"score_{klass}"] = proba[:, idx]
    result["prediction"] = model.predict(X)
    output = Path(output_path) if output_path else resolve_project_path(config, "data/processed/predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
