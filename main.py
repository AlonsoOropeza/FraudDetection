import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logisticModule as lm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

#get the data
# this is because some pokemons are dual type and they ones who are not have empty values
columns = ['type1','hp','attack','defense','sp_attack','sp_defense','speed','generation','legendary']
df = pd.read_csv('pokemon.csv', usecols=columns)

#clean the data
df = df.dropna()
df = pd.get_dummies(df, columns=['type1', 'legendary'])

#Bulbasaur,Grass,Poison,318,45,49,49,65,65,45,1,FALSE
#Squirtle,Water,,314,44,48,65,50,64,43,1,FALSE
#Charizard,Fire,Flying,534,78,84,78,109,85,100,1,FALSE
print('Welcome to the pokemon-type guesser')
print('Please fill the following stats (integer numbers)')
print('TRUE=1 and FALSE=0')
prediction = [45,49,49,65,65,45,1,1,0]
questions=['hp: ','attack: ','defense: ','sp_attack: ','sp_defense: ','speed ','generation: ','legendary: ']
# for x in range(8):
#     ans = int(input(questions[x]))
#     if x == 7:
#         prediction.append(1 if ans == 0 else 0)
#         prediction.append(1 if ans == 1 else 0)
#     else:
#         prediction.append(ans)
# Reshaping into 2Ds
prediction = np.array(prediction).reshape(1,-1)

#visualizing data
# sns.regplot(x = 'sp_attack', y = 'type1_Psychic', data = df, logistic=True, ci=None)
# plt.show()

types=['type1_Bug', 'type1_Dark', 'type1_Dragon','type1_Electric', 'type1_Fairy', 'type1_Fighting', 
'type1_Fire', 'type1_Flying', 'type1_Ghost', 'type1_Grass', 'type1_Ground','type1_Ice', 'type1_Normal', 
'type1_Poison', 'type1_Psychic', 'type1_Rock', 'type1_Steel', 'type1_Water']

for type in types:
    #split into X and y
    X = df[['hp','attack','defense','sp_attack','sp_defense','speed','generation','legendary_False','legendary_True']]
    y = df[[type]]
    X = X.values
    y = y.values.flatten()
    #print(np.shape(X),np.shape(y))

    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Normalization
    normalizer = preprocessing.StandardScaler()
    X_train = normalizer.fit_transform(X_train)
    X_test= normalizer.transform(X_test)
    prediction = normalizer.transform(prediction)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    regressor = lm.LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    # print(X_test)
    # print(predictions,y_test)
    #clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    #print(type,': ',clf.predict(prediction))
    #print(clf.score(X_test,y_test))
    print(type,': ',regressor.predict(prediction))
    print('LR classification accuracy:', accuracy(y_test, predictions))