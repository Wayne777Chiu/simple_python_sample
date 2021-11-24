
train_data,test_data = train_df.values, test_df.values

#set random to noise.
np.random.shuffle(train_data)

train_data,test_data  = np.asarray(train_data).astype('float32'), np.asarray(test_data).astype('float32')

#normalization
x_train_data,x_test_data  = train_data/255, test_data/255
y_train_onehot, y_test_data = np_utils.to_categorical()
model = Sequential()
model.add(Dense(units=255,
                input_dim =28*28,
                kernel_initializer="normal",
                activation="relu"))
model.add(Dense(units=10,
                kernel_initializer="normal",
                activation="sigmoid"))              #output layer 

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]
# print("=====================================================")

history = model.fit(train_data, train_target_data, epochs= 100, validation_data=(valid_feature_data, valid_target_data))

showplt()



# model.fit(train_feature_data,train_target_data, epochs=10, batch_size=10, verbose=0)

# loss, accuracy = model.evaluate(train_feature_data, train_target_data)
# print("Train data accuracy = {:.2f}".format(accuracy))
# loss, accuracy = model.evaluate(test_feature_data, test_target_data)
# print("Test data accuracy = {:.2f}".format(accuracy))

# Test data prediction
Y_pred = model.predict(test_feature_data)
# # print([i for item in Y_pred for i in item])
classes_Y_pred = np.around(Y_pred).astype(int)  
# print([i for item in classes_Y_pred for i in item])
loss, accuracy = model.evaluate(test_feature_data, test_target_data)
print("Test data accuracy = {:.2f}".format(accuracy))

output = pd.DataFrame({'PassengerId': test_data_df.PassengerId, 'Survived': [i for item in classes_Y_pred for i in item]})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# if __name__ == '__main__':
#     main()