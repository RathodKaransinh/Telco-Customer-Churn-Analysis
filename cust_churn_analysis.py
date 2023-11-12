import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import graphviz

tc = pd.read_csv('./customer_churn.csv')

# print("Row:", tc.shape[0])
# print("\nColumn:", tc.shape[1])
# print("\nFeatures: \n", tc.columns.tolist())
# print("\nMissing Values: \n", tc.isnull().sum().values.sum())
# print("\nUnique Values: \n", tc.nunique())

tc['TotalCharges'] = tc['TotalCharges'].replace(" ", np.nan)
tc =tc[tc['TotalCharges'].notnull()]
tc= tc.reset_index()[tc.columns]
tc['TotalCharges'] = tc['TotalCharges'].astype(float)
# tc.head()

replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols:
    tc[i] = tc[i].replace({"No internet service" : "No"})
# print(tc['OnlineSecurity'].head(15))

tc["SeniorCitizen"] = tc["SeniorCitizen"].replace({1:"Yes", 0:"No"})

def tenure_cat(tc):
    
    if tc["tenure"] <= 12:
        return "Tenure_0-12"
    
    elif (tc["tenure"] > 12) & (tc["tenure"] <= 24 ):
        return "Tenure_12-24"
    
    elif (tc["tenure"] > 24) & (tc["tenure"] <= 48) :
        return "Tenure_24-48"
    
    elif (tc["tenure"] > 48) & (tc["tenure"] <= 60) :
        return "Tenure_48-60"
    
    elif tc["tenure"] > 60 :
        return "Tenure_gt_60"
    
tc["tenure_grp"] = tc.apply(lambda tc:tenure_cat(tc),
                                      axis = 1)

#customer id col
Id_col     = ['customerID']
#Target columns
target_col = ["Churn"]
#categorical columns
cat_cols   = tc.nunique()[tc.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
# print(cat_cols)

#numerical columns
num_cols   = [x for x in tc.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values
bin_cols   = tc.nunique()[tc.nunique() == 2].keys().tolist()
# print("   ")

multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
# print(bin_cols)
for i in bin_cols :
    # print(i)
    tc[i] = le.fit_transform(tc[i])
    
#Duplicating columns for multi value columns
tc = pd.get_dummies(data = tc,columns = multi_cols )
# print(tc.head())

std = StandardScaler()
scaled = std.fit_transform(tc[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)
# print(scaled)

tc = tc.drop(columns=['tenure_grp_Tenure_12-24', 'tenure_grp_Tenure_0-12', 'tenure_grp_Tenure_24-48', 'tenure_grp_Tenure_48-60', 'tenure_grp_Tenure_gt_60'])

df_tc_og = tc.copy()
tc = tc.drop(columns = num_cols,axis = 1)
tc = tc.merge(scaled,left_index=True,right_index=True,how = "left")

bin_cols   = tc.nunique()[tc.nunique() == 2].keys().tolist()
le = LabelEncoder()
# print(bin_cols)
for i in bin_cols :
    # print(i)
    tc[i] = le.fit_transform(tc[i])
    
# print(tc.head())
# print(tc.columns)

Id_col = ['customerID']
target_col = ['Churn']

# print(tc["tenure_grp"])

cat_cols = tc.nunique()[tc.nunique() < 6]
# print(cat_cols)

cols = [i for i in tc.columns if i not in Id_col + target_col ]
# print(cols)

x = df_tc_og[cols]
y = df_tc_og[target_col]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# print(tc)
model_dt_2 = DecisionTreeClassifier(random_state = 1, max_depth = 2)
model_dt_2.fit(x_train, y_train)
model_dt_2_score_train = model_dt_2.score(x_train, y_train)
print("Training Score depth-2 : ", model_dt_2_score_train)
model_dt_2_score_test = model_dt_2.score(x_test, y_test)
print("Testing Score depth-2 : ", model_dt_2_score_test)

# depth-8
model_dt_8 = DecisionTreeClassifier(random_state=1, max_depth=8, criterion = "entropy")
model_dt_8.fit(x_train, y_train)
model_dt_8_score_train = model_dt_8.score(x_train, y_train)
print("Training score depth-8 : ",model_dt_8_score_train)
model_dt_8_score_test = model_dt_8.score(x_test, y_test)
print("Testing score depth-8 : ",model_dt_8_score_test)

dot_data = tree.export_graphviz(model_dt_8, out_file=None, 
                                feature_names=cols,  
                                class_names=target_col,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph