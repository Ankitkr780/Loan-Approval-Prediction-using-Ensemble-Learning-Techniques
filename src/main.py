from preprocessing import Preprocessor
from data_loader import get_data

dataset=get_data()
preprocess=Preprocessor(dataset=dataset)

print(preprocess.xtrain_final)
