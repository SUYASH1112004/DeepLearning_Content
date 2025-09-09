import tensorflow as tf

a=tf.constant(5)
b=tf.constant(3)
add=tf.add(a,b)
Sub=tf.subtract(a,b)
mul=tf.multiply(a,b)
div=tf.divide(a,b)

print("Addition is :",add.numpy())
print("Substraction is :",Sub.numpy())
print("Multiplication is :",mul.numpy())
print("Division is :",div.numpy())

#.numpy() coverts tensor result into numpy values