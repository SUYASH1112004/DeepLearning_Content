import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

#Sample Sequence Classification
#Suppose we have 10 sequences, each with 5 timesteps, vocabulary
size=20
x=np.random.randint(20,size=(10,5)) #Input sequence (10 samples , 5 Timesteps)
y=np.random.randint(2,size=(10,1))  #Binary Output Labels

#Build Rnn Model
model =  Sequential()
model.add(Embedding(input_dim=20,output_dim=8,input_length=5))  # Word embedding
model.add(SimpleRNN(16,activation='tanh'))   #RNN Layer
model.add(Dense(1,activation='sigmoid'))    #Output Layer

#Compile Model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Train
model.fit(x,y,epochs=5,verbose=1)

#prediction
print("Sample prediction :",model.predict(x[:1]))