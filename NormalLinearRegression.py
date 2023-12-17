import numpy as np
# Labels and feature are dataFrame 
class Normal_Linear_Regression:
    def __init__(self,Feature,Labels):
        
        ones_list = [1 for i in range(0,Feature.shape[0])]
        Feature.insert(0, 'New Feature', ones_list)
        self.Feature = Feature.values
        self.Labels = Labels.values
        self.minimize_Objective = None
    
    def fit(self):
        Feature_T = np.transpose(self.Feature)
        
        a = np.dot(Feature_T, self.Feature)     
        
        a = np.linalg.inv(a)
        
        b = np.dot(Feature_T, self.Labels)
        
        self.minimize_Objective = np.dot(a,b)
    
    def predict(self, Test_Data):
        Y = []
        Test_Data = Test_Data.values
        
        for i in range(0,Test_Data.shape[0]):
            Y.append(np.dot(Test_Data[i], self.minimize_Objective))
            
        return np.array(Y)
        