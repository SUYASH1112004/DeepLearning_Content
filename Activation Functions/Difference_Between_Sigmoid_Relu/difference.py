import numpy as np
import matplotlib.pyplot as plt
import math

#-----------------------------------
# Activation Functions
#-----------------------------------
def sigmoid(z):
    return 1/(1+math.exp(-z))

def relu(z):
    return max(0,z)

#-----------------------------------

#-----------------------------------
# Neuron Calculation
#-----------------------------------
def neuron_calculation(inputs,weights,bias,activation_fuc):
    #Display Of Inputs
    print("Inputs (X) :",inputs)
    print("Weights (W) :",weights)
    print("Bias (b) :",bias)

    #Summation
    z=sum(x*w for x,w in zip(inputs,weights))+bias
    print("Summation (Z = W*X + b) :",z)

    #calculaing activation function
    actfun=activation_fuc(z)
    print(f"Activation Function : {activation_fuc.__name__}")
    print(f"Output (Activation function) : {actfun}\n")

    return z,actfun
#---------------------------------

#--------------------------------
# Plot Sigmoid and Relu 
def plot_sigmoid_relu():
    z_value=np.linspace(-10,10,200)
    sigmoid_values=1/(1+np.exp(-z_value))
    relu_value=np.maximum(0,z_value)

    plt.figure(figsize=(8,5))
    plt.plot(z_value,sigmoid_values,label="Relu",linewidth=2,color="green")
    plt.plot(z_value,relu_value,label="Relu",linewidth=2,color="green")
    plt.axhline(y=0,color="black",linewidth=0.5)
    plt.axhline(y=1,color="black",linewidth=0.5)
    plt.axvline(x=0,color="grey",linestyle="--")
    plt.title("Sigmoid vs ReLU Activation Function",fontsize=16)
    plt.xlabel("Summation (z)",fontsize=14)
    plt.ylabel("Activation Output",fontsize=14)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.show()

def main():
    #Example Input
    inputs=[1.0,2.0,3.0]    #Exaample input features
    weights=[0.6,0.4,-0.2]  #Weights for each input
    bias=0.5

    print("=== Sigmoid Neuron ===")
    neuron_calculation(inputs,weights,bias,sigmoid)

    print("=== Relu Neuron ===")
    neuron_calculation(inputs,weights,bias,relu)

    plot_sigmoid_relu()


if __name__ == "__main__":
    main()
    