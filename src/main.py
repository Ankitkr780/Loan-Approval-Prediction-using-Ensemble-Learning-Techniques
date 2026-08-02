from preprocessing import Preprocessor
from data_loader import get_data

dataset=get_data()
preprocess_object=Preprocessor(dataset=dataset)
preprocess_object.preprocess()
print(dataset.head())
print(preprocess_object.xtrain_final)



