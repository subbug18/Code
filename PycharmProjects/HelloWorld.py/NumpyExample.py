import numpy as np


a=np.array([[2,3,45,6,57,8],[1,2,5,6,5,8]])

a[1][5] =33
print(a[1][5])

print(a.shape)

print(a.ndim)
print(a)

a=a.reshape((3,4)).copy()
print(a**2)

print([1,2,3,4]*2)

print(a[a>4])

c=[1,2, np.NaN, 3,4]

print(c)

print(np.isnan(c))

print(np.mean(c[~np.isnan(c)]))
#print(a)