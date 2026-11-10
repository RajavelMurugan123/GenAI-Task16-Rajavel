from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_data(path=DATA):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {path}. Download data.csv from "
            "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data "
            "and put it in the data folder."
        )
    df = pd.read_csv(path)
    print("\\nDATASET SHAPE:", df.shape)
    print("\\nCOLUMNS:", df.columns.tolist())
    print("\\nFIRST FIVE ROWS:\\n", df.head())
    print("\\nDATASET INFO:")
    df.info()
    print("\\nMISSING VALUES:\\n", df.isnull().sum())

    target = next((c for c in ["diagnosis", "target", "class"] if c in df.columns), None)
    if target is None:
        raise ValueError("Target column not found. Expected diagnosis, target or class.")

    y_raw = df[target]
    X = df.drop(columns=[target])
    drop_cols = [c for c in X.columns if c.lower() == "id" or c.lower().startswith("unnamed")]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = LabelEncoder().fit_transform(y_raw.astype(str)) if y_raw.dtype == "object" else y_raw.to_numpy()
    if X.empty or pd.Series(y).nunique() < 2:
        raise ValueError("Dataset needs numeric features and at least two target classes.")
    print("\\nTARGET:", target)
    print("FEATURE SHAPE:", X.shape)
    print("CLASS COUNTS:\\n", pd.Series(y).value_counts())
    return X, np.asarray(y), list(X.columns)


def save_cm(y_true, y_pred, title, filename):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150)
    plt.close()


def fit_report(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    train_acc = accuracy_score(ytr, model.predict(Xtr))
    pred = model.predict(Xte)
    test_acc = accuracy_score(yte, pred)
    print(f"\\n{name}: train accuracy={train_acc:.4f}, test accuracy={test_acc:.4f}")
    print(classification_report(yte, pred, zero_division=0))
    return model, train_acc, test_acc, pred


def main():
    X, y, feature_names = load_data()

    # Keep the final test set separate. Overall split: 60% train, 20% validation, 20% test.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # TASK 1: SVM linear and RBF kernels. Scaling is fitted inside each pipeline.
    svm_rows = []
    for kernel in ["linear", "rbf"]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel=kernel, C=1.0))
        ])
        model.fit(X_trainval, y_trainval)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        svm_rows.append({"Model": f"SVM {kernel}", "Test Accuracy": score})
        print(f"\\nSVM {kernel} test accuracy: {score:.4f}")
        save_cm(y_test, pred, f"SVM {kernel} confusion matrix", f"svm_{kernel}_cm.png")
    pd.DataFrame(svm_rows).to_csv(OUT/"svm_comparison.csv", index=False)

    # TASK 2: shallow tree versus unrestricted tree.
    tree_rows, tree_models = [], {}
    for depth, name in [(2, "Shallow tree (max_depth=2)"), (None, "Unrestricted tree")]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_trainval, y_trainval)
        tr = accuracy_score(y_trainval, tree.predict(X_trainval))
        te = accuracy_score(y_test, tree.predict(X_test))
        tree_rows.append({"Model": name, "Train Accuracy": tr, "Test Accuracy": te})
        tree_models[name] = tree
        print(f"\\n{name}: train={tr:.4f}, test={te:.4f}")
    plt.figure(figsize=(20, 10))
    plot_tree(tree_models["Shallow tree (max_depth=2)"], feature_names=feature_names,
              class_names=["Class 0", "Class 1"], filled=True, rounded=True, fontsize=7)
    plt.title("Decision Tree (max_depth=2)")
    plt.tight_layout()
    plt.savefig(OUT/"decision_tree_depth_2.png", dpi=160)
    plt.close()
    pd.DataFrame(tree_rows).to_csv(OUT/"decision_tree_comparison.csv", index=False)

    # TASK 3: train/validation/test split and tune max_depth on validation only.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    val_rows = []
    for depth in [2, 3, 4, 5, 6, 8]:
        candidate = DecisionTreeClassifier(max_depth=depth, random_state=42)
        candidate.fit(X_train, y_train)
        val_score = accuracy_score(y_val, candidate.predict(X_val))
        val_rows.append({"max_depth": depth, "Validation Accuracy": val_score})
        print(f"Validation max_depth={depth}: {val_score:.4f}")
    best_depth = max(val_rows, key=lambda r: r["Validation Accuracy"])["max_depth"]
    final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    final_tree.fit(X_train, y_train)
    final_test = accuracy_score(y_test, final_tree.predict(X_test))
    print(f"Selected max_depth={best_depth}; final held-out test accuracy={final_test:.4f}")
    pd.DataFrame(val_rows).to_csv(OUT/"validation_tuning.csv", index=False)

    # TASK 4: 5-fold stratified cross-validation on the development (train+validation) data.
    cv_model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_trainval, y_trainval, cv=cv, scoring="accuracy")
    print("\\n5-fold CV scores:", np.round(cv_scores, 4))
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}; standard deviation: {cv_scores.std():.4f}")
    print("Single SVM RBF test accuracy:", svm_rows[1]["Test Accuracy"])

    # TASK 5: Bagging versus boosting.
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                            n_estimators=100, random_state=42, n_jobs=-1)
    boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                               n_estimators=100, learning_rate=0.5, random_state=42)
    ensemble_rows = []
    for name, model in [("Bagging", bag), ("AdaBoost", boost)]:
        fitted, tr, te, pred = fit_report(name, model, X_trainval, y_trainval, X_test, y_test)
        ensemble_rows.append({"Model": name, "Train Accuracy": tr, "Test Accuracy": te})
        save_cm(y_test, pred, f"{name} confusion matrix", f"{name.lower()}_cm.png")
    pd.DataFrame(ensemble_rows).to_csv(OUT/"bagging_boosting_comparison.csv", index=False)

    # TASK 6: Single tree, bagging and random forest comparison.
    models = [
        ("Single Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Bagging Classifier", BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=100, random_state=42, n_jobs=-1)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1))
    ]
    comparison = []
    fitted_rf = None
    for name, model in models:
        fitted, tr, te, _ = fit_report(name, model, X_trainval, y_trainval, X_test, y_test)
        comparison.append({"Model": name, "Train Accuracy": tr, "Test Accuracy": te})
        if name == "Random Forest":
            fitted_rf = fitted
    pd.DataFrame(comparison).to_csv(OUT/"tree_bagging_random_forest.csv", index=False)

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": fitted_rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\\nRANDOM FOREST FEATURE IMPORTANCE (TOP 10):\\n", importance.head(10).to_string(index=False))
    importance.to_csv(OUT/"random_forest_feature_importance.csv", index=False)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance.head(10), x="Importance", y="Feature")
    plt.title("Random Forest - Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(OUT/"random_forest_feature_importance.png", dpi=160)
    plt.close()

    print("\\nAll result tables and charts are saved in:", OUT)


if __name__ == "__main__":
    main()
