# ML-Diabetic-Prediction

Description:

Model to be developed based on the given sample information of identifying the patients having diabetes or not, based on certain diagnostic measurements included in the dataset. The datasets consists of several medical predictor variables and one target variable. Data preparation & exploration, applying predictive modeling techniques, model development during train and test phase and then interpreting the Results.

Data preprocessing: Observations:
1.	There are a total of 768 records and 9 features in the dataset.
2.	Each feature can be either of integer or float dataype, there is no object datatype in this dataset.
3.	There are no zero and NaN values in the dataset.
4.	In the outcome target variable column, 1- represents having diabetes and 0- represents not having diabetes.

Data Visualization: Observations:
1.	The countplot graphically shows the imbalanced in the dataset ie., the number of patients who don't have diabetes is more than the patients having diabetes.
2.	From the heatmap correlation check between dependent and independent features and from this we get to know that there is a high correlation between target variable Outcome and independent variables Glucose, BMI ,Age and Insulin and predict the outcome.

Model Accuracy : Observations:
•	Patient is Diabetes or not prediction by fitting the logistic regression model, we get accuracy result of prediction Outcome is 74%
•	where in KNN model also almost same accuracy but after tuning by applying Cross Validation technique a small improvement in predicting the diabetes or not by 1%, for n-neighbors=39 we get accuracy Outcome as 75.4%
