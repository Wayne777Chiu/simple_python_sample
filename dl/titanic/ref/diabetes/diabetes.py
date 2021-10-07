#refer https://yanwei-liu.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-4-%E4%BD%BF%E7%94%A8keras%E5%BB%BA%E7%AB%8Bmlp%E5%92%8Clinear-regression%E6%A8%A1%E5%9E%8B-6072f1c38275
#refer https://colab.research.google.com/drive/1RjvgCt_QUPB7CQVTQ7DHL6qzfdfR1EYx#scrollTo=zNuCpqE6w6YW
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
df = pd.read_csv("./diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:8]
Y = dataset[:, 8]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割訓練和測試資料集
X_train, Y_train = X[:690], Y[:690]     # 訓練資料前690筆
X_test, Y_test = X[690:], Y[690:]       # 測試資料後78筆
print(X_train)
print(Y_train)
print("--------------------------")
# 定義模型
model = Sequential()
model.add(Dense(8, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])
# 訓練模型
model.fit(X_train, Y_train, epochs=10, batch_size=10, verbose=0)
# 評估模型
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print(Y_test)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

# 測試資料集的預測值
# Y_pred = model.predict_classes(X_test, batch_size=10, verbose=0)
Y_pred = model.predict(X_test)
# get max value
classes_Y_pred = np.argmax(Y_pred, axis=1)      
# get round value (decimal=0)
classes_Y_pred = np.around(Y_pred).astype(int)  
# print([i for item in classes_Y_pred for i in item])
# get round value (decimal=0)
classes_Y_pred = (model.predict(X_test) > 0.5).astype("int32") 

print(classes_Y_pred[0], classes_Y_pred[1])  #第0筆不會得糖尿病、第1筆會得糖尿病