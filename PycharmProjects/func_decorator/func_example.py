
print("hi")

def hello():
    print("Inside Hi")

    def nestedHello1():
        return print("nested Hello")

        def nestedHello2():
            return print("hello-2")

        nestedHello1()
        nestedHello2()




#%matplotlib
#%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y=np.arange(10)

plt.plot(x,y)
plt.show()

image = np.random.rand(30, 30)
plt.imshow(image, cmap=plt.cm.hot)