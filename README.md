# Predicting a chronical kidney disease using logistic regression
## Abstract
Chronic kidney disease (CKD) is common among adults in the United States. More than 37 million American adults may have CKD. CKD means your kidneys are damaged and can’t filter blood the way they should. The disease is called “chronic” because the damage to your kidneys happens slowly over a long period of time. This damage can cause wastes to build up in your body. CKD can also cause other health problems. The sooner you know you have kidney disease, the sooner you can make changes to protect your kidneys. 
We use logistic regression to predict the probability that the patient suffers from CKD.

## Introduction
As we implied in the abstract, our objective is to predict whether the patient suffers from ckd in time to save his life.
We will be using logistic regression, which is the go-to linear classification algorithm for two-class problems.
![logistic-model](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/linear-model.png)  
Logistic regression uses an equation as the representation, very much like linear regression. Input values (X) are combined linearly using weights or coefficient values to predict an output value (y).  
A key difference from linear regression is that the output value being modeled is a binary value (0 or 1) rather than a numeric value.  

## Materials and Methods
### Gradient Descent
In order to make our prediction we have to determine the value of each slope, we can do this using an efficient implementation of linear regression, named gradient descent.  
Gradient descent update the parameters (slopes) by calculating over and over its values until the predicted value is the same as the real value, the error is less than the learning rate or the number or iterations reach a limit. In a nutshell, gradient descent does big steps when far way, and does baby steps when close to the optimal value.
![bias-gradient-descent](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/gradient-descent.png)  
Where theta is each one of the parameters (theta 0 is the bias), alpha is the learning rate, m is the number of parameters, h0 is a prediction, y(i) is the real value and finally, xij is the value of the samples.   
### Mean Squared Error
In order to calculate our error, in each epoch we will be using the mean squared error
![mean-squared-error](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/mean-squared-error.png)  
Thus means the sumatory of the squares of the prediction minus the real value.
### Dataset
Finally the stature_hand_foot.csv dataset has the following features:
1. Age(numerical)
  	  	age in years
2. Blood Pressure(numerical)
	       	bp in mm/Hg
3. Specific Gravity(nominal)
	  	sg - (1.005,1.010,1.015,1.020,1.025)
4. Albumin(nominal)
		al - (0,1,2,3,4,5)
5. Sugar(nominal)
		su - (0,1,2,3,4,5)
6. Red Blood Cells(nominal)
		rbc - (normal,abnormal)
7. Pus Cell (nominal)
		pc - (normal,abnormal)
8. Pus Cell clumps(nominal)
		pcc - (present,notpresent)
9. Bacteria(nominal)
		ba  - (present,notpresent)
10. Blood Glucose Random(numerical)		
		bgr in mgs/dl
11. Blood Urea(numerical)	
		bu in mgs/dl
12. Serum Creatinine(numerical)	
		sc in mgs/dl
13. Sodium(numerical)
		sod in mEq/L
14. Potassium(numerical)	
		pot in mEq/L
15. Hemoglobin(numerical)
		hemo in gms
16. Packed  Cell Volume(numerical)
17 .White Blood Cell Count(numerical)
		wc in cells/cumm
18. Red Blood Cell Count(numerical)	
		rc in millions/cmm
19. Hypertension(nominal)	
		htn - (yes,no)
20. Diabetes Mellitus(nominal)	
		dm - (yes,no)
21. Coronary Artery Disease(nominal)
		cad - (yes,no)
22. Appetite(nominal)	
		appet - (good,poor)
23. Pedal Edema(nominal)
		pe - (yes,no)	
24. Anemia(nominal)
		ane - (yes,no)
25. Class (nominal)		
		class - (ckd,notckd)
  
We make a bit of **preprocessing** before we train the model with the dataframe. We filled the missing values with the mean of each column and scaled down the features using the StandardScaler of sklearn.
### How to run it
You need python 3.9.2 or later
You also need to pip install: pandas, numpy, matplotlib and sklearn.
1. Clone the repository
2. Run the file "main.py"
3. Wait for the model to finish
4. Review the prediction
## Results
||Mean Squared Error|Coeficient of Determination|
|-|-|-|
|By hand|0.075|0.6703|
|Framework|0.0125|0.945|

![errors-epochs](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/errors.png)
![feature-importance](https://raw.githubusercontent.com/AlonsoOropeza/Kidney-Logistic-Regression/main/feature_importance.png)
## Discussion
As we can see there is a relationship between handLen, footLen and height, but it is not as strong as we initially believed. Sure, we can make predictions, but the coeficient of determination is between 64% and 77%. The reason could be any of the following:
- We have few data points in out dataset (compared with the industry), therefore the model isn't as good as we want.
- The model is not as complex as the problem requires, maybe we could do better using trees or neural networks.

Altough we can't predict lenghts of human features with high accuracy, we can make good aproximations. Based on the graphs, the error is low enough to be sure the model was trained, but when I compared it with the linear regression of sklearn the accuracy buffed up to 81%. My best guess, is that for low-size datasets, its better to use linear regression than gradient descent. Since the difference in performance is minimal, and the accuracy is higher, I think it's worth it. 
## Limitations
Because we only analyze data from one source, it may be too soon to make generalized conclusions. Also the dataset contained stature, hand length, and foot length among 80 males and 75 females, which gives a total of 155 rows and that in the machine learning community is considered as a small sample. We definitily need more data (maybe records from different people around the world) to make better predicitions. 
## References
[1]"UCI Machine Learning Repository: Chronic_Kidney_Disease Data Set", Archive.ics.uci.edu, 2022. [Online]. Available: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease#. [Accessed: 04- Mar- 2022].
[2]W. Disease? and N. Health, "What Is Chronic Kidney Disease? | NIDDK", National Institute of Diabetes and Digestive and Kidney Diseases, 2022. [Online]. Available: https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/what-is-chronic-kidney-disease. [Accessed: 05- Mar- 2022].
