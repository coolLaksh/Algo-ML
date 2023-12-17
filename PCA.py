import numpy as np

class PCA:
    def __init__(self, flatten_Matrix):
        self.Data_Matrix = np.array(flatten_Matrix)
        self.row = np.shape(self.Data_Matrix)[0]
        self.colum = np.shape(self.Data_Matrix)[1]
        self.Stand_Matrix = np.zeros((self.row, self.colum))
        self.CovMatrix = np.zeros((self.colum, self.colum))
        self.Result = []
        self.eigVal = None

    def Standardize(self):
        Mean = np.mean(self.Data_Matrix,axis=0)
        Std_Dev = np.std(self.Data_Matrix,axis=0)
        for i in range(0, self.colum):
            for j in range(0, self.row):
                self.Stand_Matrix[j][i] = (self.Data_Matrix[j][i] - Mean[i]) / Std_Dev[i]

    def CoVarience_Matrix(self):
        temp = self.Stand_Matrix
        self.CovMatrix = np.cov(temp, rowvar=False)
        return

    def CalcEigen(self):
        temp = self.CovMatrix
        eigVal, eigVector = np.linalg.eig(temp)
        self.eigVal = eigVal
        Temp_Dict = {}

        for i in range(0, len(eigVal)):
            Temp_Dict[eigVal[i]] = eigVector[i]

        self.Result = [Temp_Dict[k] for k in sorted(Temp_Dict.keys(),reverse=True)]

    def getPCA(self,num):
        return np.asarray([self.Result[i] for i in range(0,num)])
