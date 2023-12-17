class LDA:
    def __init__(self,Data):
        self.Data = Data
        self.labels = Data['class_label'].copy()
        self.coef_ = None
        self.transformed_data = None
        self.S_W = None
        
    def fit_transform(self):
        #Calculating total within-class scatter
        X = self.Data
        y = self.labels
        
        labels_list = []
        
        # Seprating the labels from the dataframe
        
        for i in np.unique(self.labels):
            labels_list.append(X[X['class_label'] == i].values)
          
        # Calculating between class scatter
        temp_scatter = np.zeros((X.shape[1]-1, X.shape[1]-1))
        for i in labels_list:
            i = np.delete(i, i.shape[1]-1,axis=1)
            mean = np.mean(i,axis=0)
            for j in i:
                diff = (j-mean).reshape(X.shape[1]-1,1)
                temp_scatter += diff.dot(diff.T)
        
        self.S_W = temp_scatter
        
        temp_bt_scatter = np.zeros((X.shape[1]-1, X.shape[1]-1))
        
        X = X.drop('class_label', axis=1)
        
        global_mean = np.mean(X,axis = 0)
        global_mean = np.array(global_mean)
        for i in labels_list:
            i = np.delete(i, i.shape[1]-1,axis=1)
            mean = np.array(np.mean(i,axis=0))
            diff = (mean - global_mean).reshape(X.shape[1],1)
            temp_bt_scatter += (i.shape[0])*(diff.dot(diff.T))

        SW_inv = np.linalg.inv(self.S_W)
        
        product = np.dot(SW_inv, temp_bt_scatter)
        
        eigenvalues, eigenvectors = np.linalg.eig(product)
        
        max_index = np.argmax(eigenvalues)
        
        max_eigenvector = eigenvectors[:,max_index]
        
        self.coef_ = max_eigenvector
        
        transform_data = np.dot(X,self.coef_.reshape(X.shape[1],1))
        
        return transform_data