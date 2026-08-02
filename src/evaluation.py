from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd


class Evaluator:

    def evaluate(self, models, X_test, y_test):

        results = []

        for name, model in models.items():

            prediction = model.predict(X_test)

            results.append({

                "Model": name,

                "Accuracy": accuracy_score(
                    y_test,
                    prediction
                ),

                "Precision": precision_score(
                    y_test,
                    prediction
                ),

                "Recall": recall_score(
                    y_test,
                    prediction
                ),

                "F1": f1_score(
                    y_test,
                    prediction
                )

            })
            
            print("=" * 50)
            print(name)
            print(classification_report(y_test, prediction))
            print(confusion_matrix(y_test, prediction))

        return pd.DataFrame(results)