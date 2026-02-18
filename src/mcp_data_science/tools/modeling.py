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

    @mcp.tool()
    def cross_validate(
        target_column: str,
        model_type: str = "random_forest",
        n_folds: int = 5,
        hyperparams: dict | None = None,
        df_name: str = "",
    ) -> str:
        """K-fold cross-validation. Returns mean and std of scores across folds.
        More reliable than a single train/test split. Detects overfitting when train >> test scores.
        Example: cross_validate(target_column="Revenue", model_type="random_forest", n_folds=5)"""
        try:
            import importlib
            from sklearn.model_selection import cross_val_score

            name = store.resolve_name(df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}"

            params = hyperparams if hyperparams else {}
            feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
            if not feature_cols:
                return "Error: No numeric feature columns found."

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
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
            model = model_cls(**params)

            scores = cross_val_score(model, X, y, cv=n_folds, n_jobs=-1)

            lines = [
                f"Cross-Validation ({n_folds}-fold) for '{model_type}' on '{name}':\n",
                f"  Features: {len(feature_cols)} columns",
                f"  Target: {target_column}",
                "",
                f"  {'Fold':<8} {'Score':>10}",
                "  " + "-" * 20,
            ]
            for i, s in enumerate(scores):
                lines.append(f"  Fold {i+1:<4} {s:>10.4f}")
            lines.extend([
                "",
                f"  Mean score: {scores.mean():.4f} (+/- {scores.std():.4f})",
                f"  Min: {scores.min():.4f}, Max: {scores.max():.4f}",
            ])
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def compare_models(
        target_column: str,
        model_types: list[str] | None = None,
        n_folds: int = 5,
        df_name: str = "",
    ) -> str:
        """Compare multiple model types using cross-validation. Returns a ranked table.
        Quick way to find the best model type before fine-tuning hyperparameters.
        Default models: linear_regression, random_forest, gradient_boosting (or classifiers if target is categorical).
        Example: compare_models(target_column="Revenue", model_types=["linear_regression","random_forest","gradient_boosting"])"""
        try:
            import importlib
            from sklearn.model_selection import cross_val_score

            name = store.resolve_name(df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}"

            feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
            if not feature_cols:
                return "Error: No numeric feature columns found."

            X = df[feature_cols].fillna(0)
            y = df[target_column]

            is_classification = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20

            if model_types:
                types = model_types
            elif is_classification:
                types = ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]
            else:
                types = ["linear_regression", "random_forest", "gradient_boosting"]

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

            results = []
            for mt in types:
                if mt not in model_classes:
                    continue
                module_name, class_name = model_classes[mt]
                module = importlib.import_module(module_name)
                model_cls = getattr(module, class_name)
                model = model_cls()
                scores = cross_val_score(model, X, y, cv=n_folds, n_jobs=-1)
                results.append({
                    "model": mt,
                    "mean_score": scores.mean(),
                    "std_score": scores.std(),
                    "min_score": scores.min(),
                    "max_score": scores.max(),
                })

            results.sort(key=lambda x: x["mean_score"], reverse=True)

            lines = [
                f"Model Comparison ({n_folds}-fold CV) on '{name}':",
                f"  Features: {len(feature_cols)}, Target: {target_column}\n",
                f"  {'Rank':<6} {'Model':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}",
                "  " + "-" * 76,
            ]
            for i, r in enumerate(results):
                lines.append(
                    f"  {i+1:<6} {r['model']:<35} {r['mean_score']:>8.4f} {r['std_score']:>8.4f} "
                    f"{r['min_score']:>8.4f} {r['max_score']:>8.4f}"
                )

            winner = results[0]
            lines.append(f"\n  Best model: {winner['model']} (score={winner['mean_score']:.4f})")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def grid_search(
        target_column: str,
        model_type: str,
        param_grid: dict[str, list],
        n_folds: int = 5,
        model_name: str = "",
        train_df_name: str = "",
    ) -> str:
        """Grid search for hyperparameter tuning. Tests all combinations and stores the best model.
        Returns the best parameters and score. The best model is saved for later predict/evaluate.
        Example: grid_search(target_column="Revenue", model_type="random_forest",
                 param_grid={"n_estimators": [50,100,200], "max_depth": [5,10,20]})"""
        try:
            import importlib
            from sklearn.model_selection import GridSearchCV

            name = store.resolve_name(train_df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found."

            feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
            if not feature_cols:
                return "Error: No numeric feature columns found."

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
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)

            grid = GridSearchCV(model_cls(), param_grid, cv=n_folds, n_jobs=-1, return_train_score=True)
            grid.fit(X, y)

            resolved_name = model_name if model_name else f"grid_{model_type}_{name}"
            store.add_model(resolved_name, {
                "model": grid.best_estimator_,
                "type": model_type,
                "features": feature_cols,
                "target": target_column,
                "train_df": name,
                "trained_at": datetime.now().isoformat(),
                "train_score": grid.best_score_,
            })

            n_combos = len(grid.cv_results_["params"])
            lines = [
                f"Grid Search for '{model_type}' on '{name}':",
                f"  Combinations tested: {n_combos}",
                f"  Best score ({n_folds}-fold CV): {grid.best_score_:.4f}",
                f"  Best parameters: {grid.best_params_}",
                f"  Model saved as '{resolved_name}'",
                "",
                "  Top 5 parameter combinations:",
            ]

            results_df = pd.DataFrame(grid.cv_results_)
            results_df = results_df.sort_values("rank_test_score").head(5)
            for _, row in results_df.iterrows():
                lines.append(
                    f"    Rank {int(row['rank_test_score'])}: "
                    f"score={row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f}) "
                    f"params={row['params']}"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
