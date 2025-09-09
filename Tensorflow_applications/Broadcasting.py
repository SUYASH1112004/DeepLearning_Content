import tensorflow as tf

c=tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
d=tf.constant(3.0)

#Broad cast Add
sum=c+d

#Broad cast multiply
mul=c*d

print("Tensor C :\n",c.numpy())
print("BroadCast Addition :\n",sum.numpy())
print("broadcast multiplication :\n",mul.numpy())