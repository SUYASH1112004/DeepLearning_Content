import math
import numpy as np
import matplotlib.pyplot as plt


#Sigmoid Activation Function
def activation(z):
    return 1/(1+np.exp(-z))


#Neuron Calculation
def neuron_forward(inputs,weights,bias):
    
    #Displaying Input Weight & Bias
    print("Input (X) :",inputs)
    print("Weights (W) :",weights)
    print("Bias (B) :",bias)

    #Linear Part Of Deep Learning (Z=Summation(w.x) + b)
    z=sum(w*x for w,x in zip(weights,inputs))+bias
    print("Summation (z= summation (x.w) + b ) :",z)

    #Activation Function Output
    sigmoid=activation(z)
    print("Activation Function Output :",sigmoid)

    return z,sigmoid

#Plotting sigmoid graph

def plot_sigmoid():
    z_values=np.linspace(-10,10,200) 
    sigmoid_values=1/(1+np.exp(-z_values))
    plt.figure(figsize=(8,5))
    plt.plot(z_values,sigmoid_values,label="Sigmoid",linewidth=2,color="black")
    plt.axhline(y=0,color="Black",linewidth=0.5)
    plt.axhline(y=1,color="Black",linewidth=0.5)
    plt.axvline(x=0,color="gray",linestyle="--")
    plt.title("Sigmoid Activation function",fontsize=16)
    plt.xlabel("Summation (Z)",fontsize=14)
    plt.ylabel("Activation Output",fontsize=14)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.show()


def main():
    #Example input weights and bias
    inputs=[1.0,2.0,3.0]
    weights=[0.6,0.4,-0.2]
    bias=0.5

    #Runs the neuron forward pass
    z,sigmoid=neuron_forward(inputs,weights,bias)

    #Plot Sigmoid curve
    plot_sigmoid()


if __name__=="__main__":
    main()



