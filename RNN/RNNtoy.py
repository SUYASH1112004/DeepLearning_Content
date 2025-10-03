#Toy implementation of rnn to understand how it processes a sequence

import numpy as np

#Example Sentence:- "I Love RNN" (Encoded As Numbers)
sequence=[1,2,3]    # 1=I , 2=Love , 3=RNN

#Initialize Weights and Hidden state
wx,wh,b=0.5,0.8,0.1 #Random chosen values
h=0 # initial hidden state

print("Processing sequence step by step :")

for t,x in enumerate (sequence):
    h=np.tanh(wx*x+wh*h+b) #Memory Update
    print(f"Timestep = {t+1} | Input = {x} | Hidden State = {h:.4f}")