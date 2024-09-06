import pandas as pd
from sklearn import tree

data = pd.read_csv('iris.csv')
y = data.variety
ch = {'Setosa': 0, 'Versicolor': 1 ,'Virginica': 2}
y=y.map(ch)
X = data.iloc[:,:-1]

# Xây dựng mô hình với giải thuật Cây quyết định
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X, y)
# Dự đoán nhãn tập kiểm tra

import pickle
# save
with open('model_iris.pkl','wb') as f:
    pickle.dump(model,f)
