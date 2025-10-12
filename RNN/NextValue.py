import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN , Dense

# 1) Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2) Create a Longer Sequence(0-199)
data = np.arange(0,200,dtype=np.float32)

# 3) Normalize to  [0,1] to help training
max_value=data.max()
data_norm = data/max_value 

# 4) Make Sliding Windows : use 'Window' numbers to predict the next one
window = 5 #Timestep
X,Y = [],[]

for i in range(len(data_norm)-window):
    X.append(data_norm[i:i+window])  # e.g. [0,1,2,3,4] (normalized)
    Y.append(data_norm[i+window])   # Next Number Normalized

X=np.array(X)[...,np.newaxis]   #Shape :(Samples,Timesteps,features=1) (...) means all existing dimensions as they are.
                                #np.newaxis add new axis to dimension
Y=np.array(Y)

print("X Shape :",X.shape)  #(Samples,5,1)

#5] Train / Test split
split=int(0.8*len(X))

X_train,X_test = X[:split],X[split:]
y_train,y_test=Y[:split],Y[split:]

# 6] Build a tiny RNN
model = Sequential([
    SimpleRNN(32,activation='tanh',input_shape=(window,1)),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),loss='mse')

# 7] Train
history=model.fit(X_train,y_train,epochs=30,batch_size=32,verbose=0,validation_data=(X_test,y_test))

# 8] Helper to predict the next number for a given (Unnormalized sequence) Sequence
def pred_next(seq):
    seq=np.asarray(seq,dtype=np.float32)
    assert len(seq)== window ,f"Need {window} Numbers "
    seq_norm = (seq / max_value).reshape(1,window,1)
    pred_norm= model.predict(seq_norm,verbose=0)[0,0]
    return pred_norm * max_value # de normalize back

# 9] Demo Prediction
tests=[
    [95,96,97,98,99],
    [7,8,9,10,11],
    [100,101,102,103,104]
]

for t in tests:
    print(f"Input :{t} -> Predicted next : {round(pred_next(t))}")

