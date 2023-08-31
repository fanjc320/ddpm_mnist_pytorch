import numpy as np

a=np.array([1,2,3])
print("a:", a)
b=a[:,None]
print("a None:", b)

c=a[:, None, None]
# 第一个冒号代表切片，把一维a的元素全部切完，然后第二个为None，表示把一维变成二维并且第二个维度上的值为0，第三个元素为None，表示增加一个维度，并且该维度值为0
# 所以a[:,None,None]结果为：
print("a None None:", c)

d=a[:, None, None, None]
print("a None None None:", d)

a=np.array([[3,3,3],[4,4,4],[5,5,5]])
e=a[:,None]
print("a None:", e)