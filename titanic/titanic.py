import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


TRAIN = 'train.csv'
TEST  = 'test.csv'

X = [
    'Sex',
    'Pclass',
    'Fare',
    'AgeFilledByClassMedian',
    'AgeClass',
    'SibSp',
    'Parch',
    'FamilySize',
    'Married'
]
Y = 'Survived'
ID = 'PassengerId'

OUTPUT = [ID, Y]

MODE = 'DEVELOP'


def gender_to_integer(gender):
    if gender.strip().lower() == 'male':
        return 1
    else:
        return 0


def embarked_to_integer(embarked):
    if embarked == 'C':
        return 1
    elif embarked == 'Q':
        return 2
    elif embarked == 'S':
        return 3
    else:
        return 0


def clean_data(data, mode='train'):
    data['Sex'] = data['Sex'].apply(gender_to_integer)
    data['Title'] = data['Name'].apply(extract_name_title)
    data['Embarked'] = data['Embarked'].apply(embarked_to_integer)
    data['AgeFilledByClassMedian'] = fill_ages_with_class_median(data)
    data['AgeClass'] = data['AgeFilledByClassMedian'] * data['Pclass']
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['Married'] = extract_marital_status(data)

    return data.fillna(0)

def fill_ages_with_class_median(data):
    median_age_by_class = data.groupby('Pclass')['Age'].median().to_dict()
    return data.apply(
        lambda row:
            median_age_by_class[row['Pclass']]
            if pd.isnull(row['Age'])
            else row['Age'],
        axis=1
    )

def extract_name_title(name):
    return name.split(',')[1].split('.')[0].strip()

def name_title_to_integer(name_title):
    return

def extract_marital_status(data):
    return data['Title'].apply(
        lambda title:
            0
            if (title == 'Master') or (title == 'Miss') or (title == 'Mlle') or (title == 'Ms')
            else 1
    )

def fit_model(x, y):
    model = GridSearchCV(
        estimator  = LinearSVC(),
        n_jobs = -1,
        param_grid = dict(
            C = np.logspace(-6, -1, 10)
        ),
    )
    #model = RandomForestClassifier(n_estimators = 100)
    model.fit(x, y)
    return model


def predict(model, x):
    return model.predict(x)


def evaluate_model(model, x, true_y):
    predicted_y = model.predict(x)
    return classification_report(true_y, predicted_y)


def main():
    #Read training data
    train = clean_data(
        pd.read_csv('train.csv'),
        mode = 'train'
    )

    #Read test data
    test = clean_data(
        pd.read_csv('test.csv'),
        mode = 'test'
    )

    if MODE == 'DEVELOP':
        #Split data (for model development)
        x_train, x_test, y_train, y_test = train_test_split(
            train[X],
            train[Y],
            test_size = 0.2,
            train_size = 0.8
        )

        model = fit_model(x_train, y_train)

        print(evaluate_model(model, x_test, y_test))

    elif MODE == 'FINAL':
        model = fit_model(train[X], train[Y])
        test[Y] = predict(model, test[X])
        test[OUTPUT].to_csv('predictions.csv', index=False)

    else:
        print("ERROR: Bad MODE specified; choose one of DEVELOP or FINAL")

if __name__ == '__main__':
    main()
