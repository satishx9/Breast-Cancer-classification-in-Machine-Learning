import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, f1 and confusion‑matrix."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, cm


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ---------- 1. Retrieve & save the uploaded CSV ----------
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # ---------- 2. Read data ----------
        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            return render_template("index.html", error=f"Error reading CSV: {e}")

        if data.shape[1] < 2:
            return render_template(
                "index.html", error="CSV must have at least two columns (features and a target)."
            )

        # ---------- 3. Pre‑processing (mirror bc.ipynb) ----------
        # 3.1 Drop unneeded columns
        columns_to_drop = ["id", "Unnamed: 32"]
        data = data.drop(columns=[c for c in columns_to_drop if c in data.columns], errors="ignore")

        # 3.2 Drop rows with missing values (bc.ipynb does this instead of filling)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            return render_template("index.html", error="No data left after dropping rows with missing values.")

        # 3.3 Separate features (X) and target (y)
        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

        # 3.4 Encode target labels to match bc.ipynb (LabelEncoder)
        le = LabelEncoder()
        y = le.fit_transform(y)

        # 3.5 Ensure features are numeric, then scale with StandardScaler
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna()
        if X.empty or len(y) == 0:
            return render_template("index.html", error="Not enough usable data after cleaning.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Need at least two classes in the target
        if len(np.unique(y)) < 2:
            return render_template("index.html", error="Target column must contain at least two unique classes.")

        # ---------- 4. Train / test split ----------
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        # ---------- 5. Train models ----------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }

        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy, precision, recall, f1, cm = calculate_metrics(y_test, y_pred)
                results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cm": cm.tolist(),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        return render_template("index.html", results=results, filename=file.filename)

    # GET request
    return render_template("index.html", results=None)


if __name__ == "__main__":
    app.run(debug=True)

