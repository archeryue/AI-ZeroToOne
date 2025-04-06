# auto-grad engine for scalars

class Scalar:
    def __init__(self, val, _prev=(), _op=''):
        self.val = val
        self.grad = 0
        # attributes for auto-grad
        self._backward = lambda: None
        self._prev = _prev
        self._op = _op

    def __repr__(self):
        return f"Scalar(val={self.val}, grad={self.grad})"

    def __add__(self, other):
        next = Scalar(self.val + other.val, (self, other), '+')
        def _backward():
            self.grad += next.grad
            other.grad += next.grad
        next._backward = _backward
        return next

    def __mul__(self, other):
        next = Scalar(self.val * other.val, (self, other), '*')
        def _backward():
            self.grad += next.grad * other.val
            other.grad += next.grad * self.val
        next._backward = _backward
        return next

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Wrong Operand Type"
        next = Scalar(self.val ** other, (self,), f'**{other}')
        def _backward():
            self.grad += next.grad * other * self.val ** (other - 1)
        next._backward = _backward
        return next
    
    def relu(self):
        next = Scalar(max(0, self.val), (self,), 'ReLU')
        def _backward():
            self.grad += next.grad * (self.val > 0)
        next._backward = _backward
        return next

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build_topo(p)
                topo.append(v)
        build_topo(self)
        # back-propagate the gradients
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * Scalar(-1.0)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
