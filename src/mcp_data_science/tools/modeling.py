from datetime import datetime

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def train_test_split(
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
        df_name: str = "",
    ) -> str:
        """Split dataframe into train/test sets. Stores them as '{name}_train' and '{name}_test'.
        stratify=True preserves target class proportions (classification only).
        Run BEFORE training any model. Always use fixed random_state for reproducibility. Use stratify=True for imbalanced classification datasets.
        Example: train_test_split(target_column="Revenue", test_size=0.2)"""
        try:
            from sklearn.model_selection import train_test_split as sklearn_split

            name = store.resolve_name(df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}"

            strat = df[target_column] if stratify else None
            train_df, test_df = sklearn_split(
                df, test_size=test_size, random_state=random_state, stratify=strat
            )

            train_name = f"{name}_train"
            test_name = f"{name}_test"
            store.add(train_name, train_df.reset_index(drop=True), set_current=False)
            store.add(test_name, test_df.reset_index(drop=True), set_current=False)

            return (
                f"Split '{name}' into train/test (test_size={test_size}, random_state={random_state}).\n"
                f"  '{train_name}': {train_df.shape[0]} rows ({train_df.shape[0]/len(df)*100:.0f}%)\n"
                f"  '{test_name}': {test_df.shape[0]} rows ({test_df.shape[0]/len(df)*100:.0f}%)"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def train_model(
        target_column: str,
        model_type: str = "random_forest",
        model_name: str = "",
        hyperparams: dict | None = None,
        train_df_name: str = "",
    ) -> str:
        """Train a model. Types: 'linear_regression', 'logistic_regression', 'random_forest',
        'gradient_boosting', 'decision_tree'. The model is stored for later predict/evaluate.
        Start with simple model (linear/logistic) as baseline. Only move to complex models (random_forest, gradient_boosting) if baseline is insufficient. Uses ALL numeric columns as features.
        Example: train_model(target_column="Revenue", model_type="random_forest")"""
        try:
            name = store.resolve_name(train_df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}"

            params = hyperparams if hyperparams else {}

            # Select numeric features only (excluding target)
            feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
            if not feature_cols:
                return "Error: No numeric feature columns found (excluding target)."

            X = df[feature_cols].fillna(0)
            y = df[target_column]

            model_classes = {
                "linear_regression": ("sklearn.linear_model", "LinearRegression"),
                "logistic_regression": ("sklearn.linear_model", "LogisticRegression"),
                "random_forest": ("sklearn.ensemble", "RandomForestRegressor"),
                "random_forest_classifier": ("sklearn.ensemble", "RandomForestClassifier"),
                "gradient_boosting": ("sklearn.ensemble", "GradientBoostingRegressor"),
                "gradient_boosting_classifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
                "decision_tree": ("sklearn.tree", "DecisionTreeRegressor"),
                "decision_tree_classifier": ("sklearn.tree", "DecisionTreeClassifier"),
            }

            if model_type not in model_classes:
                return f"Error: Unknown model type '{model_type}'. Available: {list(model_classes.keys())}"

            module_name, class_name = model_classes[model_type]
            import importlib
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
            model = model_cls(**params)
            model.fit(X, y)

            # Training score
            train_score = model.score(X, y)

            resolved_model_name = model_name if model_name else f"{model_type}_{name}"
            store.add_model(resolved_model_name, {
                "model": model,
                "type": model_type,
                "features": feature_cols,
                "target": target_column,
                "train_df": name,
                "trained_at": datetime.now().isoformat(),
                "train_score": train_score,
            })

            return (
                f"Trained '{model_type}' model as '{resolved_model_name}'.\n"
                f"  Features: {len(feature_cols)} columns\n"
                f"  Training score (R²/Accuracy): {train_score:.4f}\n"
                f"  Features used: {feature_cols}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def predict(
        model_name: str = "",
        prediction_column: str = "prediction",
        df_name: str = "",
    ) -> str:
        """Generate predictions on a dataframe using a stored model. Stores predictions as a new column.
        Generate predictions on new data. The dataframe must contain the same feature columns used during training.
        Example: predict(model_name="random_forest_data_train", df_name="data_test")"""
        try:
            models = store.list_model_names()
            if not models:
                return "Error: No models trained. Use train_model first."
            resolved_model = model_name if model_name else models[-1]

            model_info = store.get_model(resolved_model)
            model = model_info["model"]
            feature_cols = model_info["features"]

            name = store.resolve_name(df_name)
            df = store.get(name)

            missing_features = [c for c in feature_cols if c not in df.columns]
            if missing_features:
                return f"Error: Missing feature columns: {missing_features}"

            X = df[feature_cols].fillna(0)
            preds = model.predict(X)
            df[prediction_column] = preds
            store.set(name, df)

            if pd.api.types.is_numeric_dtype(preds):
                stats = f"min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}"
            else:
                stats = f"unique={len(set(preds))}"

            return (
                f"Predictions stored as '{prediction_column}' in '{name}' using model '{resolved_model}'.\n"
                f"  {len(preds)} predictions. {stats}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def evaluate_model(
        model_name: str = "",
        test_df_name: str = "",
        target_column: str = "",
    ) -> str:
        """Evaluate a model on test data. Returns classification metrics (accuracy, precision, recall, F1,
        confusion matrix) or regression metrics (MAE, MSE, RMSE, R²).
        Always evaluate on TEST set, never training set. Compare metrics against baseline. Test score much worse than training = overfitting.
        Example: evaluate_model(model_name="random_forest_data_train", test_df_name="data_test")"""
        try:
            models = store.list_model_names()
            if not models:
                return "Error: No models trained. Use train_model first."
            resolved_model = model_name if model_name else models[-1]

            model_info = store.get_model(resolved_model)
            model = model_info["model"]
            feature_cols = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(test_df_name)
            df = store.get(name)

            if target not in df.columns:
                return f"Error: Target column '{target}' not found in '{name}'."

            X = df[feature_cols].fillna(0)
            y_true = df[target]
            y_pred = model.predict(X)

            # Determine if classification or regression
            is_classification = hasattr(model, "predict_proba") or hasattr(model, "classes_")

            if is_classification:
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                )
                acc = accuracy_score(y_true, y_pred)
                avg = "weighted" if len(set(y_true)) > 2 else "binary"
                prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
                rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
                f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
                return (
                    f"Classification Evaluation for '{resolved_model}' on '{name}':\n"
                    f"  Accuracy:  {acc:.4f}\n"
                    f"  Precision: {prec:.4f}\n"
                    f"  Recall:    {rec:.4f}\n"
                    f"  F1-score:  {f1:.4f}\n"
                    f"\nConfusion Matrix:\n{cm}"
                )
            else:
                from sklearn.metrics import (
                    mean_absolute_error, mean_squared_error, r2_score
                )
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                return (
                    f"Regression Evaluation for '{resolved_model}' on '{name}':\n"
                    f"  MAE:  {mae:.4f}\n"
                    f"  MSE:  {mse:.4f}\n"
                    f"  RMSE: {rmse:.4f}\n"
                    f"  R²:   {r2:.4f}"
                )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def list_models() -> str:
        """List all trained models with their type and training metadata.
        Check which models have been trained and their training scores before deciding next steps."""
        names = store.list_model_names()
        if not names:
            return "No models trained."
        lines = ["Trained models:"]
        for n in names:
            info = store.get_model(n)
            lines.append(
                f"  - {n}: type={info['type']}, features={len(info['features'])}, "
                f"target={info['target']}, train_score={info.get('train_score', 'N/A'):.4f}"
            )
        return "\n".join(lines)
