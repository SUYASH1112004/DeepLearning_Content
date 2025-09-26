import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN, Dense 

def SentimentAnalysis():
    #Tiny dataset (0-Negative ; 1-Positive)
    sentences=[
        "I Love this movie",
        "This film was great",
        "What a fantastic experience",
        "I really enjoyed it",
        "Absolutely wonderful acting",
        "I hate this move",
        "The film was terrible",
        "What a bad experience",
        "I really disliked it",
        "Absolutely horrible acting"
     ]

    print("Input Dataset")
    print(sentences)
    labels=[1,1,1,1,1,0,0,0,0,0]

    #Tokenize text -> convert word to integer
    tokenizer=Tokenizer(num_words=50,oov_token="<oov>")
    tokenizer.fit_on_texts(sentences)
    X=tokenizer.texts_to_sequences(sentences)

    print("word Index :",tokenizer.word_index)  #Show vocabulary

    #Pad sequences -> same length input
    maxlen=5
    x=pad_sequences(X,maxlen=maxlen)
    y=np.array(labels)


    #Build simple RNN model
    model=Sequential()
    model.add(Embedding(input_dim=50,output_dim=8,input_length=maxlen)) #Word Embedding
    model.add(SimpleRNN(8,activation='tanh'))
    model.add(Dense(1,activation='sigmoid'))    #Binary Output
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    #Train the model
    model.fit(x,y,epochs=30,verbose=1)

    #test on new example
    test_Sentences=["I Enjoyed the film","I hated the film"]
    test_seq=tokenizer.texts_to_sequences(test_Sentences)
    test_seq=pad_sequences(test_seq,maxlen=maxlen)
    pred=model.predict(test_seq)

    for s,p in zip (test_Sentences,pred):
        print(f"Sentences : {s} -> Sentiment:","Positive" if p>0.5 else "Negative")

def main():
    SentimentAnalysis()

if __name__ == "__main__":
    main()
    