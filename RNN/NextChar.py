import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense
from tensorflow.keras.utils import to_categorical

def char_Prediction():
    #1) Prepare Toy Text Dataset
    text="hellohellohellohello" #Repeating Pattern
    chars=sorted(list(set(text)))   #Unique Characters
    char_to_int= {c:i for i,c in enumerate (chars)}
    int_to_char={i:c for i,c in enumerate(chars)}

    print("Unique Characters :",chars)

    #2) Convert text to integer sequence
    encoded=[char_to_int[c] for c in text]

    #3) Create Input Output pairs (Sequence Length = 3 )
    seq_length=3
    x,y =[],[]

    for i in range(len(encoded)-seq_length):
        x.append(encoded[i:i+seq_length])   #e.g : "hel"
        y.append(encoded[i+seq_length])    #next char eg : "l"
    
    x=np.array(x)
    y=np.array(y)

    #One-hot encoded output
    y=to_categorical(y,num_classes=len(chars))

    #Reshape X for RNN (Samples,TimeSteps,Features)
    x=x.reshape((x.shape[0],x.shape[1],1))


    #4) Build RNN Model
    model=Sequential()
    model.add(SimpleRNN(16,activation='tanh',input_shape=(seq_length,1)))
    model.add(Dense(len(chars),activation='softmax'))

    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy"])

    #5)Train
    model.fit(x,y,epochs=300,verbose=0)

    #6) Test Prediction
    test_input=["h","e","l"]    #We Expect model to predict "l"
    encoded_input=np.array([[char_to_int[c] for c in test_input]]).reshape(1,seq_length,1)

    pred=model.predict(encoded_input,verbose=0)
    predicted_char=int_to_char[np.argmax(pred)]

    print("Input:",test_input,"-> Predicted Next Char :",predicted_char)

def main():
    char_Prediction()

if __name__=="__main__":
    main()

