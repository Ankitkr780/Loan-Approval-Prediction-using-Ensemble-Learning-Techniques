from preprocessing import Preprocessor
from data_loader import get_data
from model_training import ModelTrainer
from evaluation import Evaluator
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

print(results.sort_values(
    by="Accuracy",
    ascending=False
))
