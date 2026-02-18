import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mcp.server.fastmcp import FastMCP, Image

from mcp_data_science.state import DataStore
from mcp_data_science.tools._plot_helpers import fig_to_image, fig_to_image_and_store


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def plot_feature_importance_model(
        model_name: str = "",
        top_n: int = 20,
        save_path: str = "",
    ) -> Image:
        """Bar plot of feature importance from a trained model.
        Works with tree-based models (random_forest, gradient_boosting, decision_tree).
        Use after train_model to understand which features drive predictions.
        Example: plot_feature_importance_model(model_name="random_forest_data_train")"""
        try:
            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained. Use train_model first.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_.ravel())
                if len(importances) != len(features):
                    importances = importances[:len(features)]
            else:
                raise ValueError(f"Model '{resolved}' does not expose feature importances.")

            imp_df = pd.DataFrame({"feature": features, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=True).tail(top_n)

            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
            ax.barh(imp_df["feature"], imp_df["importance"])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {resolved}")
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"feature_importance_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_residuals(
        model_name: str = "",
        test_df_name: str = "",
        target_column: str = "",
        save_path: str = "",
    ) -> Image:
        """Residuals vs predicted values plot for regression models.
        Use to diagnose model quality: random scatter = good, patterns = systematic error.
        Also shows a histogram of residuals to check normality.
        Example: plot_residuals(model_name="linear_regression_data_train", test_df_name="data_test")"""
        try:
            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(test_df_name)
            df = store.get(name)
            X = df[features].fillna(0)
            y_true = df[target]
            y_pred = model.predict(X)
            residuals = y_true - y_pred

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
            axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs Predicted")

            # Residual distribution
            axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
            axes[1].set_xlabel("Residual")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Residual Distribution")

            fig.suptitle(f"Residual Analysis — {resolved}", fontsize=14)
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"residuals_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_confusion_matrix(
        model_name: str = "",
        test_df_name: str = "",
        target_column: str = "",
        save_path: str = "",
    ) -> Image:
        """Visual confusion matrix heatmap for classification models.
        Shows true vs predicted labels with counts. Use after evaluate_model for deeper diagnosis.
        Example: plot_confusion_matrix(model_name="rf_classifier", test_df_name="data_test")"""
        try:
            from sklearn.metrics import confusion_matrix as sk_cm

            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(test_df_name)
            df = store.get(name)
            X = df[features].fillna(0)
            y_true = df[target]
            y_pred = model.predict(X)

            labels = sorted(set(y_true) | set(y_pred))
            cm = sk_cm(y_true, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {resolved}")
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"confusion_matrix_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_roc_curve(
        model_name: str = "",
        test_df_name: str = "",
        target_column: str = "",
        save_path: str = "",
    ) -> Image:
        """ROC curve with AUC score for binary classification models.
        Plots the trade-off between true positive rate and false positive rate.
        AUC = 0.5 means random, AUC = 1.0 means perfect.
        Example: plot_roc_curve(model_name="logistic_data_train", test_df_name="data_test")"""
        try:
            from sklearn.metrics import roc_curve, auc

            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(test_df_name)
            df = store.get(name)
            X = df[features].fillna(0)
            y_true = df[target]

            if not hasattr(model, "predict_proba"):
                raise ValueError(f"Model '{resolved}' does not support predict_proba (needed for ROC).")

            y_prob = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {resolved}")
            ax.legend(loc="lower right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"roc_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_precision_recall_curve(
        model_name: str = "",
        test_df_name: str = "",
        target_column: str = "",
        save_path: str = "",
    ) -> Image:
        """Precision-Recall curve for binary classification models.
        Better than ROC for imbalanced datasets. Shows trade-off between precision and recall.
        Example: plot_precision_recall_curve(model_name="logistic_data_train", test_df_name="data_test")"""
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(test_df_name)
            df = store.get(name)
            X = df[features].fillna(0)
            y_true = df[target]

            if not hasattr(model, "predict_proba"):
                raise ValueError(f"Model '{resolved}' does not support predict_proba.")

            y_prob = model.predict_proba(X)[:, 1]
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, label=f"PR (AP = {ap:.4f})", linewidth=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve — {resolved}")
            ax.legend(loc="upper right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"pr_curve_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_learning_curve(
        model_name: str = "",
        train_df_name: str = "",
        target_column: str = "",
        n_splits: int = 5,
        save_path: str = "",
    ) -> Image:
        """Learning curve: training and validation scores vs training set size.
        Diagnoses overfitting (gap between train/val) and underfitting (both scores low).
        Example: plot_learning_curve(model_name="rf_model", train_df_name="data_train")"""
        try:
            from sklearn.model_selection import learning_curve

            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(train_df_name)
            df = store.get(name)
            X = df[features].fillna(0)
            y = df[target]

            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=n_splits,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring=None, n_jobs=-1,
            )

            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
            ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
            ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation score")
            ax.set_xlabel("Training Set Size")
            ax.set_ylabel("Score")
            ax.set_title(f"Learning Curve — {resolved}")
            ax.legend(loc="lower right")
            ax.grid(True)
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"learning_curve_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def permutation_importance(
        model_name: str = "",
        test_df_name: str = "",
        target_column: str = "",
        n_repeats: int = 10,
        top_n: int = 20,
        save_path: str = "",
    ) -> Image:
        """Permutation importance: model-agnostic feature importance measured on test data.
        More reliable than built-in feature_importances_ because it measures actual impact on predictions.
        Shows importance with error bars from multiple random shuffles.
        Example: permutation_importance(model_name="rf_model", test_df_name="data_test")"""
        try:
            from sklearn.inspection import permutation_importance as sk_perm_imp

            models = store.list_model_names()
            if not models:
                raise ValueError("No models trained.")
            resolved = model_name if model_name else models[-1]
            model_info = store.get_model(resolved)
            model = model_info["model"]
            features = model_info["features"]
            target = target_column if target_column else model_info["target"]

            name = store.resolve_name(test_df_name)
            df = store.get(name)
            X = df[features].fillna(0)
            y = df[target]

            result = sk_perm_imp(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)

            imp_df = pd.DataFrame({
                "feature": features,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }).sort_values("importance_mean", ascending=True).tail(top_n)

            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
            ax.barh(imp_df["feature"], imp_df["importance_mean"], xerr=imp_df["importance_std"])
            ax.set_xlabel("Mean Importance (decrease in score)")
            ax.set_title(f"Permutation Importance — {resolved}")
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"perm_importance_{resolved}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)
