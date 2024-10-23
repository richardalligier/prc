import numpy as np

def pad_rightmost_axis(a,pad_size):
    pad_width = [(0,0)] * len(a.shape)
    pad_width[-1] = (0,pad_size)
    return np.pad(a,pad_width)

def make_same_degrees(a,b):
    da = a.degree()
    db = b.degree()
    if da > db: b = Polynomial(pad_rightmost_axis(b.coeffs,da-db))
    if db > da: a = Polynomial(pad_rightmost_axis(a.coeffs,db-da))
    return a,b
# used to model a batch of polynomials
# righ-most dimension is the coeffs dim, choice made to enable broadcast on it
class Polynomial:
    def __init__(self,coeffs):
        self.coeffs = coeffs
    def eval(self,x): # Horner formula
        res = self.coeffs[...,-1]
        n = self.coeffs.shape[-1]
        for i in range(n-2,-1,-1):
            res = self.coeffs[...,i] + res  * x
        return res
    def __str__(self):
        return str(self.coeffs)
    def degree(self):
        return self.coeffs.shape[-1] - 1
    def shape(self):
        return self.coeffs.shape
    def deriv(self):
        res = self.coeffs[...,1:] * np.arange(1,self.degree()+1)
        return Polynomial(res)
    def __add__(self,other):
        a,b = make_same_degrees(self,other)
        return Polynomial(a.coeffs + b.coeffs)
    def __mul__(self,other):
        if isinstance(other, (int, float, complex)):
            return Polynomial(other*self.coeffs)
        a, b = self, other
        resshape = np.broadcast_shapes(*[x.coeffs.shape[:-1] for x in [a,b]])
        resshape = resshape + (a.coeffs.shape[-1]+b.coeffs.shape[-1]-1,)
        res = np.zeros(resshape,dtype=np.float64)
        for i in range(a.degree()+1):
            for j in range(b.degree()+1):
                res[..., i + j] += a.coeffs[...,i] * b.coeffs[...,j]
        return Polynomial(res)
    def __rmul__(self,other):
        return self.__mul__(other)
    def __sub__(self,other):
        return self + (-1)* other
    def roots2(self):
        assert self.degree() == 2
        a = self.coeffs[...,2]
        b = self.coeffs[...,1]
        c = self.coeffs[...,0]
        delta = b ** 2 - 4 * a * c
        sqrtdelta = np.sqrt(delta)
        x1 = (- b - sqrtdelta)/(2 * a)
        x2 = (- b + sqrtdelta)/(2 * a)
        return np.stack([x1,x2],axis=-1)

def main():
    a = Polynomial(np.arange(9).reshape(3,3))
    b = Polynomial(np.arange(5))
    print(a)
    print(a.roots2())
    print(b)
    print(a+b)
    print(a)
    print(a*b)
    print(a.deriv())
    pass

if __name__ == '__main__':
    main()
