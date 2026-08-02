import pandas as pd

def get_data()->pd.DataFrame:
    dataset=pd.read_csv("../data/loan_approval_dataset.csv")
    return dataset