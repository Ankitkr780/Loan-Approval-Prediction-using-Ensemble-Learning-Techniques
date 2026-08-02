from preprocessing import Preprocessor
from data_loader import get_data
from model_training import ModelTrainer
from evaluation import Evaluator
from visualization import Visualizer

dataset=get_data()
preprocess_object=Preprocessor(dataset=dataset)
preprocess_object.preprocess()
print(dataset.head())
print(preprocess_object.xtrain_final)


trainer = ModelTrainer()

models = trainer.train(
    preprocess_object.xtrain_final,
    preprocess_object.ytrain_final
)

evaluator = Evaluator()

results = evaluator.evaluate(
    models,
    preprocess_object.xtest_final,
    preprocess_object.ytest_final
)

visualizer=Visualizer()
visualizer.plot_accuracy(results)

    # Confusion Matrix for every model
for name, model in models.items():

    visualizer.plot_confusion_matrix(
            model,
            name,
            preprocess_object.xtest_final,
            preprocess_object.ytest_final
    )

    # ROC Curve
visualizer.plot_roc_curve(
        models,
        preprocess_object.xtest_final,
        preprocess_object.ytest_final
)

    # Feature Importance (Best Model)
visualizer.plot_feature_importance(
        evaluator.best_model,
        preprocess_object.xtrain_final.columns,
        evaluator.best_model_name
)

print("Visualizations Saved")

    # -------------------------------------------------
    # Save Best Model
    # -------------------------------------------------

print("\n[6] Saving Best Model...")

visualizer.save_best_model(
        evaluator.best_model
)

print(f"Best Model : {evaluator.best_model_name}")
print(f"Best F1 Score : {evaluator.best_score:.4f}")

print("\nProject Completed Successfully!")

