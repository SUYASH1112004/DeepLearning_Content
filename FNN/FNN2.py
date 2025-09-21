from sklearn.datasets import make_moons #Synthetic Dataset Generator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Synthetic 2D Classification 
x,y=make_moons(n_samples=1000,noise=0.2,random_state=42)
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.25,random_state=42)


#2 Hidden Layers : 16 & 8 Neurons Relu + Adam Optimizer  
#First hiddden layer contain 16 neuron & Second hidden layer contain 8 neuron
clf=MLPClassifier(hidden_layer_sizes=(16,8),activation='relu',solver='adam',max_iter=500,random_state=42)
clf.fit(xtr,ytr)


print("Test Accuracy :",accuracy_score(yte,clf.predict(xte)))