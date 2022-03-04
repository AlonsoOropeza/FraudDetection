# Predicting a chronical kidney disease using logistic regression
## Abstract
Have you ever wonder if the stature of a person determines the lenght of their feet and hands? For Lord of the Rings fans, what about hobbits, they are smaller than humans, nevertheless they have larger feet. Is there any relation between these variables or is it mere coincidence? Thankfully, using linear regression, we can make a prediction based on its dependance of other features to finally answer if its true our hypothesis or not. 

## Introduction
As we implied in the abstract, our objective is to determine if there is any relation between the stature of a human and the lenght of their feet and hands.  
We will be using linear regression, which is a machine learning algorithm thats tries to fit data into a linear model. 
![linear-model](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/linear-model.png)  
But for those of you who just read that line and didn't understand what the heck I am talking about, let me rephrase it.  
Do you remember your highschool math clases, when you saw the equation of a line?  
The famous **y = mx + b**  
Where y is the output, m is the slope of the line, x is the input and b is the intercept.  
Well this is similar, but with minor modifications. Now we can have multiple features, that means multiple pairs of slopes and x's.    
In other words: **prediction = m1x1 + m2x2 + m3x3 + ... + mnxn + bias**    
As you saw, b is now the **bias**, which is the difference between our actual and predicted values. The model is also affected by how "noisy" the data is, the so called **variance** is the model’s sensitivity to fluctuations in the data. Analyize the image below for further explanation.  
![bias-variance](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/bias-variance.png)  

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
Finally the stature_hand_foot.csv dataset has the following variables:
- idGen (w/in gender)  
- gender (1=M, 2=F)
- height (mm)  
- handLen (mm)   
- footLen (mm)
  
We make a bit of **preprocessing** before we train the model with the dataframe. We scaled down a little bit the height, handLen and footLen in order to be in centimeters, we also used hot-econding with the gender. 
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
