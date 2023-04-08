import math 

class Value:
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None # internal variables used for autograd graph construction
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc


    def __add__(self, other):
        """
        Adds two `Value` objects together.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out


    def __mul__(self, other):
        """
        Multiplies two `Value` objects together.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out


    def __pow__(self, other):
        """
        Raises the `Value` object to the power of `other`.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out


    def relu(self):
        """
        Computes the rectified linear unit (ReLU) function of the `Value` object.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def sigmoid(self):
        """
        Computes the sigmoid function of the `Value` object.
        """
        x = self.data
        t = (1 + math.exp(-x))**-1
        out = Value(t, (self,), 'sigmoid')
        
        def _backward():
            self.grad += t*(1-t) * out.grad 
        out._backward = _backward

        return out  


    def tanh(self):
        """
        Computes the hyperbolic tangent function of the `Value` object.
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    

    def exp(self):
        """
        Computes the exponential function of the `Value` object.
        """
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        
        return out
    
    
    def log(self):
        """
        Computes the natural logarithm of the `Value` object.
        """
        if self.data <= 0:
            raise ValueError("can't log negative or zero value")
        x = self.data
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad = (x**(-1)) * out.grad
        out._backward = _backward
        return out


    def backward(self):
        """
        Computes the gradients of all previous nodes.
        """
        # topological order all of the children in the graph
        cdef list topo = []
        cdef set visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __neg__(self): # -self
        """
        Computes the negative of the `Value` object.
        """
        return self * -1


    def __radd__(self, other): # other + self
        """
        Adds a `Value` object to another value.
        """
        return self + other


    def __sub__(self, other): # self - other
        """
        Subtracts a `Value` object from another `Value` object.
        """
        return self + (-other)


    def __rsub__(self, other): # other - self
        """
        Subtracts a `Value` object from another value.
        """
        return other + (-self)


    def __rmul__(self, other): # other * self
        """
        Multiplies a `Value` object with another value.
        """
        return self * other


    def __truediv__(self, other): # self / other
        """
        Divides a `Value` object by another `Value` object.
        """
        return self * other**-1


    def __rtruediv__(self, other): # other / self
        """
        Divides a value by a `Value` object.
        """
        return other * self**-1


    def __repr__(self):
        """
        Returns a string representation of the `Value` object.
        """
        return f"Value(data={self.data}, grad={self.grad})"