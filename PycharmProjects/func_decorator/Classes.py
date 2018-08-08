class A:
    def f(self):
        return self.g()
    def g(self):
        return 'A'

class B(A):
    def g(self):
        return 'B'


a=A()
print(a.f(), a.g())

b=B()
print(b.f(), b.g())


class S(object):
    def __init__(self, a):
        self.a = a
    def __add__(self, x):
            return S(self.a + x)

class T(object):
    def __init__(self, b):
        self.b=b
    def __mul__(self, y):
        return T(self.b*y)
s=S(2)
t=T(3)

u= s + 10
print(u.a)

y = t *10
print(y.b)

#print(10.__mul__(10))