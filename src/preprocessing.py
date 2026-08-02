from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self,dataset):
        self.dataset=dataset
        self.xtrain_final=None
        self.ytrain_final=None
        self.xval_final=None
        self.yval_final=None
        self.xtest_final=None
        self.ytest_final=None

        self.dataset.columns=self.dataset.columns.str.strip()
        for col in self.dataset.select_dtypes(include="object").columns:
            self.dataset[col] = self.dataset[col].str.strip()

    def encodings(self):
        encoder=OrdinalEncoder(categories=[["Not Graduate","Graduate"]])

        self.dataset["education"]=encoder.fit_transform(self.dataset[["education"]]) 

        self.dataset["self_employed"] = self.dataset["self_employed"].map({
            "Yes": 1,
            "No": 0
        }).astype(int)

        self.dataset["loan_status"] = self.dataset["loan_status"].map({
            "Approved": 1,
            "Rejected": 0
        }).astype(int)       

    def splitting(self):
        input_data=self.dataset.iloc[:,:-1]
        ouptut_data=self.dataset.iloc[:,-1]
        xtrain,xeval,ytrain,yeval=train_test_split(input_data,ouptut_data,stratify=ouptut_data,test_size=0.30,random_state=15)
        xval,xtest,yval,ytest=train_test_split(xeval,yeval,test_size=0.50,stratify=yeval,random_state=15)

        return xtrain,ytrain,xval,yval,xtest,ytest 
        

    def preprocess(self):
        self.encodings()
        xtrain,ytrain,xval,yval,xtest,ytest=self.splitting()
        self.ytrain_final=ytrain
        self.xtrain_final=xtrain
        self.yval_final=yval
        self.xval_final=xval
        self.ytest_final=ytest
        self.xtest_final=xtest
        
