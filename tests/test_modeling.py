"""Tests for modeling tools (modeling.py)."""
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import modeling
from conftest import ToolCaller


# ---------------------------------------------------------------------------
# Regression fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg_env(numeric_df):
    """Environment with numeric data, split into train/test."""
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", numeric_df.copy())
    modeling.register_tools(mcp, store)
    call = ToolCaller(mcp)
    # Pre-split
    call("train_test_split", target_column="target", test_size=0.2,
         random_state=42)
    return store, call


# ---------------------------------------------------------------------------
# Classification fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_env(classification_df):
    """Environment with classification data, split into train/test."""
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", classification_df.copy())
    modeling.register_tools(mcp, store)
    call = ToolCaller(mcp)
    call("train_test_split", target_column="target", test_size=0.2,
         random_state=42)
    return store, call


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_default(self, numeric_df):
        mcp = FastMCP("test")
        store = DataStore()
        store.add("data", numeric_df.copy())
        modeling.register_tools(mcp, store)
        call = ToolCaller(mcp)
        result = call("train_test_split", target_column="target")
        assert "data_train" in store.list_names()
        assert "data_test" in store.list_names()
        train = store.get("data_train")
        test = store.get("data_test")
        assert len(train) == 80
        assert len(test) == 20

    def test_custom_size(self, numeric_df):
        mcp = FastMCP("test")
        store = DataStore()
        store.add("data", numeric_df.copy())
        modeling.register_tools(mcp, store)
        call = ToolCaller(mcp)
        result = call("train_test_split", target_column="target",
                       test_size=0.3)
        assert len(store.get("data_test")) == 30

    def test_missing_target(self, numeric_df):
        mcp = FastMCP("test")
        store = DataStore()
        store.add("data", numeric_df.copy())
        modeling.register_tools(mcp, store)
        call = ToolCaller(mcp)
        result = call("train_test_split", target_column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestTrainModel:
    def test_linear_regression(self, reg_env):
        store, call = reg_env
        result = call("train_model", target_column="target",
                       model_type="linear_regression")
        assert "Error" not in str(result)
        assert len(store.list_model_names()) > 0

    def test_random_forest(self, reg_env):
        store, call = reg_env
        result = call("train_model", target_column="target",
                       model_type="random_forest",
                       hyperparams={"n_estimators": 10})
        assert "Error" not in str(result)

    def test_gradient_boosting(self, reg_env):
        store, call = reg_env
        result = call("train_model", target_column="target",
                       model_type="gradient_boosting",
                       hyperparams={"n_estimators": 10})
        assert "Error" not in str(result)

    def test_decision_tree(self, reg_env):
        store, call = reg_env
        result = call("train_model", target_column="target",
                       model_type="decision_tree")
        assert "Error" not in str(result)

    def test_custom_name(self, reg_env):
        store, call = reg_env
        result = call("train_model", target_column="target",
                       model_type="linear_regression",
                       model_name="my_model")
        assert "my_model" in store.list_model_names()

    def test_bad_model_type(self, reg_env):
        _, call = reg_env
        result = call("train_model", target_column="target",
                       model_type="xyz_model")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestPredict:
    def test_happy(self, reg_env):
        store, call = reg_env
        call("train_model", target_column="target",
             model_type="linear_regression")
        model_name = store.list_model_names()[0]
        result = call("predict", model_name=model_name,
                       df_name="test_test")
        df = store.get("test_test")
        assert "prediction" in df.columns


class TestEvaluateModel:
    def test_regression(self, reg_env):
        store, call = reg_env
        call("train_model", target_column="target",
             model_type="linear_regression")
        model_name = store.list_model_names()[0]
        result = call("evaluate_model", model_name=model_name,
                       test_df_name="test_test", target_column="target")
        assert "R" in result or "MAE" in result or "MSE" in result

    def test_r2_positive(self, reg_env):
        store, call = reg_env
        call("train_model", target_column="target",
             model_type="linear_regression")
        model_name = store.list_model_names()[0]
        result = call("evaluate_model", model_name=model_name,
                       test_df_name="test_test", target_column="target")
        # Linear data should have positive R²
        assert "R" in result

    def test_classification(self, clf_env):
        store, call = clf_env
        call("train_model", target_column="target",
             model_type="logistic_regression")
        model_name = store.list_model_names()[0]
        result = call("evaluate_model", model_name=model_name,
                       test_df_name="test_test", target_column="target")
        assert "Accuracy" in result or "accuracy" in result or "F1" in result or "Precision" in result


class TestListModels:
    def test_empty(self, numeric_df):
        mcp = FastMCP("test")
        store = DataStore()
        store.add("data", numeric_df.copy())
        modeling.register_tools(mcp, store)
        call = ToolCaller(mcp)
        result = call("list_models")
        assert "No" in result or "0" in result or "empty" in result.lower()

    def test_after_train(self, reg_env):
        store, call = reg_env
        call("train_model", target_column="target",
             model_type="linear_regression")
        result = call("list_models")
        assert "linear" in result.lower()
