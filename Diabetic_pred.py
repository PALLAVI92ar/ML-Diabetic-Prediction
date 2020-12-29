# Project is to Predicting the diabetic patients
# Model to be developed based on the given sample information of identifying the patients having diabetic or not 
# Number of variables used here is 9
# Data Preparation & Exploration, applying predictive modeling techniques Model Development during train and test phase, Interpreting the Results.
# Missing values treatment using mean and median
# Applied Logistic Regression, K- Nearest neighbor hood

# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing dataset
df=pd.read_csv("diabetes.csv")

# Preview data
pd.set_option('display.max_columns', None)
df.head()

# Dataset dimensions - (rows, columns)
df.shape

# Features data-type
df.dtypes

# List of features
list(df)

# Statistical summary
df.describe()

# Count of null values 
df.isnull().sum()

# Outcome countplot
sns.countplot(x = 'Outcome',data = df)

# Pairplot 
sns.pairplot(data = df, hue = 'Outcome')
plt.show()

# Heatmap
# Finds correlation between Independent and dependent attributes
# Selecting the variables using correlation
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True)
plt.show()

# split X and Y variables
X=df.iloc[:,0:8]
list(X)

# Finds correlation between Independent features
X.corr()

Y=df['Outcome']
Y

# standardize the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scale=scaler.fit_transform(X)

# Implementing train n test cases
from sklearn.model_selection._split import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.30,random_state=40)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

# Implementing the Logistic regression model
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)

print(logreg.intercept_)
print(logreg.coef_)

Y_pred_lreg=logreg.predict(X_test)

# Import the metrics class
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,Y_pred_lreg)
print("Confusion matrix on Y_test and Y_pred_lreg =",cm)

acc=metrics.accuracy_score(Y_pred_lreg, Y_test).round(3)
print("Diabetic patient or not prediction accuracy is:",acc )

# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_lreg))

# Graphically visualize actual values with the predicted values
import matplotlib.pyplot as plt
plt.matshow(cm)

plt.title("Confusion matrix")
plt.colorbar()

plt.xlabel("True label")
plt.ylabel("Predicted label")

plt.show()

##############################################################################
# Implementing the KNN model for same dataset to select and fit the best model

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)

Y_pred_knn=knn.predict(X_test)

cm = metrics.confusion_matrix(Y_test,Y_pred_knn)
print("Confusion matrix on Y_test and Y_pred_knn =",cm)

knn_acc = metrics.accuracy_score(Y_test,Y_pred_knn)
print("Diabetic patient or not prediction accuracy where k=7 is:",acc )

# Applying cross validation for KNN
from sklearn.model_selection import GridSearchCV

k=range(1,51)
p_val=[1,2]

hyperparameters=dict(n_neighbors=k, p=p_val)

clf=GridSearchCV(KNeighborsClassifier(),hyperparameters,cv=10)
knn1=clf.fit(X_train,Y_train)

knn1.cv_results_

print(knn1.best_score_)
print(knn1.best_params_)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_knn))

#Patient is Diabetic or not prediction by fit the model logistic regression we get accuracy result of prediction is 74%
#where in KNN model also same result but after tuning by applying CV technique a small improvement in predicting the diabetic or not by 1%
#.e, for n-neighbors=39 we get accuracy result as 75.4%
