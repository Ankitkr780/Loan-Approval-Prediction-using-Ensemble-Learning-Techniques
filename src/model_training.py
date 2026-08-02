from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

class ModelTrainer:

    def __init__(self):
        self.models = {}

    def train(self, X_train, y_train):

        dt = DecisionTreeClassifier(random_state=42)

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        et = ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        )

        gb = GradientBoostingClassifier(
            random_state=42
        )

        voting = VotingClassifier(
            estimators=[
                ("rf", rf),
                ("et", et),
                ("gb", gb)
            ],
            voting="soft"
        )

        self.models = {
            "Decision Tree": dt,
            "Random Forest": rf,
            "Extra Trees": et,
            "Gradient Boosting": gb,
            "Voting": voting
        }

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)

        

        return self.models