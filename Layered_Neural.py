import math

#------------------------------------
#  Activation Function
#------------------------------------
def Relu(x):
    return max(0,x)

def sigmoid(x):
    return 1/(1+math.exp(-x))
#------------------------------------

#-----------Forward Pass With Step By Step Printing ------------
def forwardpass(inputs):
    print("---------Input Layer----------")
    print(f"Input X1 : {inputs[0]} ,Input X2 : {inputs[1]}")

    #---------Hidden Layers-----------
    weights_hidden=[
        [0.5,-0.2], #Neuron1 Weight
        [0.8,0.4]   #Neuron2 Weight
        ]
    
    bias_hidden=[0.1,-0.1]  #Bias For Each Hidden Neuron
    
    hidden_output=[]
    print("-----------Hidden Layers -----------------")
    for i in range (len(weights_hidden)):
        print(f"Neuron {i+1}")
        w=weights_hidden[i]
        b=bias_hidden[i]

        #--------Multiplication steps----------
        print("Step 1 :- Multiply inputs by weights ")
        print(f"{w[0]} x {inputs[0]} = {w[0]*inputs[0]:.3f}")
        print(f"{w[1]} x {inputs[1]} = {w[1]*inputs[1]:.3f}")

        #------------Summation Step------------
        
        z=sum(w_j * x_j for w_j,x_j in zip(w,inputs))+b
        print("Step 2 :- Add Result and Bias")
        print(f"Summation :{z:.3f}")

        #Activation
        a=Relu(z)
        print(f"Apply Relu (max(0,{z:.3f})) : {a:.3f}")

        hidden_output.append(a)
    #--------Output Layer----------
    weights_output=[1.0,-1.5]       #Weights from hidden to output
    bias_output=0.2

    print("-------------Output layer------------")
    print("Step 1 : Multiply hidden output by weights")
    print(f"({weights_output[0]} * {hidden_output[0]}) = {weights_output[0] * hidden_output[0]:.3f}")
    print(f"({weights_output[1]} * {hidden_output[1]}) = {weights_output[1] * hidden_output[1]:.3f}")

    z_out=sum(w_o * h for w_o,h in zip(weights_output,hidden_output))+bias_output
    print(f"Step 2 : Add Result & bias {z_out:.3f}")

    y_hat=sigmoid(z_out)
    print(f"Step 3 : Apply Sigmoid 1/(1+exp(-z)) : {y_hat:.3f}")

    print(f"Final Output:{y_hat:.3f} -> {y_hat*100:.2f} % Confidence in positive class")

if __name__ == "__main__":
    inputs=[2.0,3.0]    #Example Input Feature
    forwardpass(inputs)
 