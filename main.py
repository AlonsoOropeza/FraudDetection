from locale import normalize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logisticModule as lm
from sklearn import preprocessing

#get the data
# this is because some pokemons are dual type and they ones who are not have empty values
columns = ["type1","total","hp","attack","defense","sp_attack","sp_defense","speed","generation","legendary"]
df = pd.read_csv('pokemon.csv', usecols=columns)

#clean the data
df = df.dropna()
df = pd.get_dummies(df, columns=['type1', 'legendary'])

#        'type1_Bug', 'type1_Dark', 'type1_Dragon',
#        'type1_Electric', 'type1_Fairy', 'type1_Fighting', 'type1_Fire',
#        'type1_Flying', 'type1_Ghost', 'type1_Grass', 'type1_Ground',
#        'type1_Ice', 'type1_Normal', 'type1_Poison', 'type1_Psychic',
#        'type1_Rock', 'type1_Steel', 'type1_Water'
type="type1_Rock"

#visualizing data
# sns.regplot(x = 'sp_attack', y = "type1_Psychic", data = df, logistic=True, ci=None)
# plt.show()

#split into X and y
X = df[["total","hp","attack","defense","sp_attack","sp_defense","speed","generation",'legendary_False','legendary_True']]
y = df[[type]]
X = X.values
y = y.values.flatten()
#print(np.shape(X),np.shape(y))

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Normalization
normalizer = preprocessing.Normalizer()
X_train = normalizer.fit_transform(X_train)
X_test= normalizer.transform(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = lm.LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(X_test)
print(predictions,y_test)

print("LR classification accuracy:", accuracy(y_test, predictions))