import matplotlib.pyplot as plt
import numpy as np

#ReLU Activation Function
def relu(z):
    return np.maximum(0,z)

#Neuron Calculation
def neuron_calculation(inputs,weights,bias):
    #Displaying input
    print("Inputs x :",inputs)
    print("Weights W :",weights)
    print("Bias B :",bias)

    #Calculating summation with bias
    z=sum(x*w for x,w in zip(inputs,weights))+bias
    print("Summation (z = w*x + b) :",z)

    #Calculating Activation 
    actfun=relu(z)
    print("Activation Function (RELU ) Output :",actfun)

    return z,actfun

def plot_relu():
    z_value=np.linspace(-10,10,200)
    relu_value=np.maximum(0,z_value)

    plt.figure(figsize=(8,5))
    plt.plot(z_value,relu_value,label="Relu",linewidth=2,color="green")
    plt.axhline(y=0,color="black",linewidth=0.5)
    plt.axvline(x=0,color="grey",linestyle="--")
    plt.title("ReLU Activation Function",fontsize=16)
    plt.xlabel("Summation (z)",fontsize=14)
    plt.ylabel("Activation Output",fontsize=14)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.show()


def main():
    #Example input weights and bias
    inputs=[1.0,2.0,3.0]    #Exaample input features
    weights=[0.6,0.4,-0.2]  #Weights for each input
    bias=0.5

    #Run the neuron forward pass
    z,actfun=neuron_calculation(inputs,weights,bias)

    plot_relu()


if __name__ == "__main__":
    main()
