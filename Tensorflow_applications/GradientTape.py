#Gradient computation is required for optimization purpose
import tensorflow as tf

x=tf.Variable(3.0)

with tf.GradientTape() as tape:
    y=x**2 + 2*x+1      #Function y

grade=tape.gradient(y,x)

print("Function : Y = X^2 + 2x + 1")
print("Gradient dy/dx :",grade.numpy())
