# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
DATA PREPROCESSING
'''
# Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical Variables
# Label Encoding
lab_enc_geog = LabelEncoder()
X[:, 1] = lab_enc_geog.fit_transform(X[:, 1])
lab_enc_gend = LabelEncoder()
X[:, 2] = lab_enc_gend.fit_transform(X[:, 2])
# One Hot Encoding - Remove Dummy Var Trap
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

# Test - Train Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feture Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


'''
BUILDING ANN
'''
# Importing keras as required packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
classifier = Sequential()

# Adding Layers
classifier.add(Dense(units=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, init='uniform', activation='relu'))
classifier.add(Dense(units=1, init='uniform', activation='sigmoid'))    # Output Layer
print(classifier.summary())

# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to data
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predict
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Checking Results - Confusion Matrix & Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
acc = accuracy_score(y_test, y_pred)
print(acc)



'''
HOMEWORK ASSIGNMENT:

Use our ANN model to predict if the customer with the following informations will leave the bank:Â 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ?Â Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000

So should we say goodbye to that customer ?
'''

X_check = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
X_check = sc.transform(X_check)
y_check = classifier.predict(X_check)
y_check = (y_check > 0.5)


'''
EVALUATION - K-FOLDS CROSS VALIDATION
'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

def build_classifier():
    model_clf = Sequential()
    model_clf.add(Dense(units=6, init='uniform', activation='relu', input_dim=11))
    model_clf.add(Dense(units=6, init='uniform', activation='relu'))
    model_clf.add(Dense(units=1, init='uniform', activation='sigmoid'))
    model_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model_clf

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=kfold, n_jobs=-1)

mean_acc = accuracies.mean()
var_acc = accuracies.std()


'''
WITH DROPOUT
'''
from keras.layers import Dropout

model_clf = Sequential()
model_clf.add(Dense(units=6, init='uniform', activation='relu', input_dim=11))
model_clf.add(Dropout(rate = 0.1))
model_clf.add(Dense(units=6, init='uniform', activation='relu'))
model_clf.add(Dropout(rate = 0.1))
model_clf.add(Dense(units=1, init='uniform', activation='sigmoid'))
model_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_clf.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = model_clf.predict(X_test)
y_pred = (y_pred > 0.5)

acc = accuracy_score(y_test, y_pred)



'''
HYPERPARAMETER TUNING - GRIDSEARCH
'''
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    model_clf = Sequential()
    model_clf.add(Dense(units=6, init='uniform', activation='relu', input_dim=11))
    model_clf.add(Dropout(rate = 0.1))
    model_clf.add(Dense(units=6, init='uniform', activation='relu'))
    model_clf.add(Dropout(rate = 0.1))
    model_clf.add(Dense(units=1, init='uniform', activation='sigmoid'))
    model_clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model_clf

classifier = KerasClassifier(build_fn=build_classifier)

# Defining parameters for Grid Search
params = {'batch_size' : [25, 32],
          'epochs' : [100, 500],
          'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=params,
                           scoring = 'accuracy',
                           cv = 10)

grid_search_fit = grid_search.fit(X_train, y_train)
best_parameters = grid_search_fit.best_params_
best_accuracy = grid_search_fit.best_score_













