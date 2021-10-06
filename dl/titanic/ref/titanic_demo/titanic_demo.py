
# refer https://www.kaggle.com/c/titanic
import numpy as np # linear algebra
import pandas as pd # data processing, csv file I/O

serial_number  = 0

def serial_number_plus():
    global serial_number
    serial_number += 1
    return serial_number


dicArray = {
    'h': b'\xe2\x94\x9c',
    'l': b'\xE2\x94\x82',
    'm': b'\xE2\x94\x80',
    'e': b'\xE2\x94\x94',
    '+': b'\xE2\x95\x94',
    '%': b'\xE2\x95\x97',
    'p': b'\xE2\x95\x9A',
    'q': b'\xE2\x95\x9D',
    '-': b'\xE2\x95\x90',
    '|': b'\xE2\x95\x91',
    'k': b'\xE2\x95\xA0',
    'f': b'\xE2\x95\xA3',
}

def decode(x):
    return (''.join(dicArray.get(i,i.encode('utf-8')).decode('utf-8') for i in x))

def main():
    print(serial_number_plus(),decode('------------------------------------------------------------------------'))


# refer https://www.kaggle.com/c/titanic   
import os
import sys
for dirname, _, filenames in os.walk('.\data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv(".\data/train.csv")
train_data.head()

test_data = pd.read_csv('.\data/test.csv')
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)


# Use RandomForestClassifier for demo 
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


if __name__ == '__main__':
    main()