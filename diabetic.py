from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
dataset = loadtxt('diabetes.csv', delimiter=',')
#print(dataset)
x = dataset[:,0:8] #row 0 to 7 column
y = dataset[:,8] #8 column alone
#print("Input",x)
#print("output",y)
model = Sequential()
model.add(Dense(12,input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))#1 neuron enough if activates diabetes is there else not there
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model training
model.fit(x, y, epochs=30, batch_size=10)#increasing value of epochs increases accuracy
#evaluation
_, accuracy = model.evaluate(x,y)
print('Accuracy: %.2f' %(accuracy*100))
#Model save
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

