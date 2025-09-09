import tensorflow as tf

#Define Matrix
A=tf.constant([[1,2],[3,4]],dtype=tf.float32)
B=tf.constant([[5,6],[7,8]],dtype=tf.float32)

#Matrix Multipication
matmul=tf.matmul(A,B)

#Transpose
Transpose=tf.transpose(A)

print("Matrix A :\n",A.numpy())
print("Matrix B :\n",B.numpy())
print("Matrix Multiplication :\n",matmul.numpy())
print("Matrix Transpose of A :\n",Transpose.numpy())
