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
    def __init__(self):
        self.best_score = 0
        self.best_model = None
        self.best_model_name = None
        
    def evaluate(self, models, X_test, y_test):

        results = []

        for name, model in models.items():

            prediction = model.predict(X_test)

            accuracy = accuracy_score(y_test, prediction)
            precision = precision_score(y_test, prediction)
            recall = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })

            if f1>self.best_score:
                self.best_score = f1
                self.best_model = model
                self.best_model_name = name
                
            print("=" * 50)
            print(name)
            print(classification_report(y_test, prediction))
            print(confusion_matrix(y_test, prediction))

        return pd.DataFrame(results)