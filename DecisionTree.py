import graphviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mth


class Tree_Node:
    def __init__(self):
        self.leaf = False
        self.impurity = None
        self.Feature_Name = None
        self.Feature_Value = None
        self.Best_Gain = None
        self.Index = None
        self.Left_Child = None
        self.Right_Child = None
        self.d_type = None
        self.Label = None


def Calc_Entropy(Y):
    tot_Entropy = 0
    uniques_ = Y.iloc[:, 0].unique()
    if len(uniques_) == 1:
        return 0
    for i in uniques_:
        P_i = (Y == i).sum()[0] / len(Y)
        tot_Entropy += P_i * mth.log2(P_i)

    return -1 * tot_Entropy


def calc_Info_Gain(parent, child):
    return parent - child


def create_Tree(root, X, Y, min_samples):
    global calc_Info_gain
    if len(Y['target'].unique()) == 1 or len(Y) < min_samples or len(X) <= 1:
        root.leaf = True
        value_counts = Y.value_counts()

        # Get the value with the highest count
        root.Label = value_counts.idxmax()
        return

    root.imputity = Calc_Entropy(Y)

    # Split test indicate feature having feature which hae to be choosen at this stage
    # index 0 --> Info Gain
    # index 1 --> Split_Value
    # index 2 --> Which column
    # index 3 --> 0 implies d-type in continuous and 1 implies d-type in discrete
    split_list = [0, 0, -1, 0]

    # Now iterate over each feature to find which one is best for splitting
    for i in range(X.shape[1]):

        Parent_impurity = root.imputity
        d_type = -1
        # There can be a case when a feature is continuous or discrete, so we have to handle it accordingly
        if X.iloc[:, 0].dtype == np.bool_ or X.iloc[:, 0].dtype == np.str_:
            # Case1 --> Feature is discrete
            d_type = 1
            best_split_value = 0
            best_gain = 0
            Unique_values = X.iloc[:, i].unique()
            for j in Unique_values:
                df1 = Y.loc[X.iloc[:, i] == j]
                df2 = Y.loc[X.iloc[:, i] != j]

                impurity1 = Calc_Entropy(df1)
                impurity2 = Calc_Entropy(df2)

                Avg_impurity = len(df1) * impurity1 + len(df2) * impurity2
                Avg_impurity /= len(Y)

                calc_Info_gain = calc_Info_gain(Parent_impurity, Avg_impurity)

                if best_gain < calc_Info_gain:
                    best_gain = calc_Info_gain
                    best_split_value = j

        else:
            # Case2 --> Feature is continuous
            d_type = 0
            sorted_feature = sorted(X.iloc[:, i])
            # Selecting the best split fot this continuous feature
            best_split_value = 0
            best_gain = 0
            for j in range(len(sorted_feature) - 1):
                temp_split = (sorted_feature[j] + sorted_feature[j + 1]) / 2

                df1 = Y.loc[X.iloc[:, i] < temp_split]
                df2 = Y.loc[X.iloc[:, i] >= temp_split]

                impurity1 = Calc_Entropy(df1)
                impurity2 = Calc_Entropy(df2)

                Avg_impurity = (len(df1) * impurity1 + len(df2) * impurity2) / len(Y)

                calc_Info_gain = calc_Info_Gain(Parent_impurity, Avg_impurity)
                if calc_Info_gain > best_gain:
                    best_gain = calc_Info_gain
                    best_split_value = temp_split

        if best_gain > split_list[0]:
            split_list[0] = best_gain
            split_list[1] = best_split_value
            split_list[2] = i
            split_list[3] = d_type

    if split_list[3] == 0:
        # Feature having maximum info gain is continuous
        root.Feature_Name = X.columns[split_list[2]]
        root.Feature_Value = split_list[1]
        root.Best_Gain = split_list[0]
        root.Index = split_list[2]
        root.d_type = 0
        # Creating Left and Right Child's

        df1 = X[X[root.Feature_Name] < root.Feature_Value]
        Y1 = Y.loc[X.iloc[:, root.Index] < root.Feature_Value]

        df2 = X[X[root.Feature_Name] >= root.Feature_Value]
        Y2 = Y.loc[X.iloc[:, root.Index] >= root.Feature_Value]

        left_Child = Tree_Node()
        right_Child = Tree_Node()
        create_Tree(left_Child, df1, Y1, min_samples)
        create_Tree(right_Child, df2, Y2, min_samples)

        root.Left_Child = left_Child
        root.Right_Child = right_Child
        return root
    else:
        root.Feature_Name = X.columns[split_list[2]]
        root.Feature_Value = split_list[1]
        root.Best_Gain = split_list[0]
        root.Index = split_list[2]
        root.d_type = 1
        # Creating Left and Right Child's

        df1 = X[X[root.Feature_Name] == root.Feature_Value]
        Y1 = Y.loc[X.iloc[:, root.Index] == root.Feature_Value]

        df2 = X[X[root.Feature_Name] != root.Feature_Value]
        Y2 = Y.loc[X.iloc[:, root.Index] != root.Feature_Value]

        left_Child = Tree_Node()
        right_Child = Tree_Node()
        create_Tree(left_Child, df1, Y1, min_samples)
        create_Tree(right_Child, df2, Y2, min_samples)

        root.Left_Child = left_Child
        root.Right_Child = right_Child
        return root


def fit(X, Y, min_samples):
    root = Tree_Node()

    return create_Tree(root, X, Y, min_samples)


def helper(root, X):
    # Base Case
    if root.leaf:
        return root.Label

    if root.d_type == 0:
        if (X[root.Feature_Name] < root.Feature_Value).all():
            return helper(root.Left_Child, X)
        else:
            return helper(root.Right_Child, X)
    else:
        if (X[root.Feature_Name] == root.Feature_Value).all():
            return helper(root.Left_Child, X)
        else:
            return helper(root.Right_Child, X)


def predict(root, X):
    return helper(root, X)


from sklearn.datasets import load_iris

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column to the dataframe
iris_df['target'] = iris.target


def split_Train_Test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_length = int(len(data) * test_ratio)
    test_indices = shuffled[:test_length]
    train_indices = shuffled[test_length:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_data, test_data = split_Train_Test(iris_df, 0.3)

iris_df = train_data

target_column = iris_df.pop('target')

# create a new DataFrame with the extracted column
target_df = pd.DataFrame({'target': target_column})

root_ = fit(iris_df, target_df, 1)

root_

test_colum = test_data.pop('target').values

scores = 0
id = 0
for index, row in test_data.iterrows():
    # create a new dataframe with the current row data
    row_df = pd.DataFrame(row).transpose()
    label_ = predict(root_, row_df)
    if label_ == test_colum[id]:
        scores += 1
    id += 1
print((scores / len(test_colum)) * 100)


def draw_tree(root, name):
    dot = graphviz.Digraph(comment='Tree')
    add_node(dot, root, name)
    dot.render('tree.gv', view=True)


def add_node(dot, node, name):
    if node.leaf:
        dot.node(name, str(node.Label))
    else:
        dot.node(name, str(node.Feature_Name))
        left_name = name + 'l'
        add_node(dot, node.Left_Child, left_name)
        dot.edge(name, left_name)
        right_name = name + 'r'
        add_node(dot, node.Right_Child, right_name)
        dot.edge(name, right_name)


draw_tree(root_, 'root')
