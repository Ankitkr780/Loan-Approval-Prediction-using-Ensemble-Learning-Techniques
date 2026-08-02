import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay

#This is by AI
class Visualizer:

    def __init__(self):
        self.plot_dir = "../outputs/plots"
        self.model_dir = "../models"

        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    # -------------------------------------------------
    # Accuracy Comparison
    # -------------------------------------------------

    def plot_accuracy(self, results):

        plt.figure(figsize=(8, 5))

        plt.bar(
            results["Model"],
            results["Accuracy"]
        )

        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xlabel("Models")
        plt.xticks(rotation=15)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.plot_dir,
                "accuracy_comparison.png"
            )
        )

        plt.close()

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    def plot_confusion_matrix(
        self,
        model,
        model_name,
        X_test,
        y_test
    ):

        fig, ax = plt.subplots(figsize=(6, 6))

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            cmap="Blues",
            ax=ax
        )

        ax.set_title(f"{model_name} Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.plot_dir,
                f"{model_name.lower().replace(' ','_')}_cm.png"
            )
        )

        plt.close()

    # -------------------------------------------------
    # ROC Curve
    # -------------------------------------------------

    def plot_roc_curve(
        self,
        models,
        X_test,
        y_test
    ):

        plt.figure(figsize=(8, 6))

        ax = plt.gca()

        for name, model in models.items():

            RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_test,
                ax=ax,
                name=name
            )

        plt.title("ROC Curve Comparison")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.plot_dir,
                "roc_curve.png"
            )
        )

        plt.close()

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    def plot_feature_importance(
        self,
        model,
        feature_names,
        model_name
    ):

        if not hasattr(model, "feature_importances_"):
            return

        importance = model.feature_importances_

        indices = importance.argsort()[::-1]

        plt.figure(figsize=(10, 6))

        plt.barh(
            range(len(feature_names)),
            importance[indices]
        )

        plt.yticks(
            range(len(feature_names)),
            feature_names[indices]
        )

        plt.gca().invert_yaxis()

        plt.title(f"{model_name} Feature Importance")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.plot_dir,
                f"{model_name.lower().replace(' ','_')}_feature_importance.png"
            )
        )

        plt.close()

    # -------------------------------------------------
    # Save Best Model
    # -------------------------------------------------

    def save_best_model(
        self,
        best_model
    ):

        joblib.dump(
            best_model,
            os.path.join(
                self.model_dir,
                "best_model.pkl"
            )
        )

        print("\nBest model saved successfully.\n")