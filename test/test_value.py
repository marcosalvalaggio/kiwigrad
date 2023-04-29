from kiwigrad import Value
import torch
import unittest

class TestValue(unittest.TestCase):

    def test_add_operation(self):
        #kiwigrad
        a = Value(2.)
        b = Value(3.)
        c = a + b
        c.backward()
        #pytorch
        at = torch.tensor([2.], requires_grad=True)
        bt = torch.tensor([3.], requires_grad=True)
        ct = at + bt
        ct.backward()
        #test
        self.assertAlmostEqual(a.grad, at.grad)
        self.assertAlmostEqual(b.grad, bt.grad)

    def test_mul_operation(self):
        #kiwigrad
        a = Value(2.)
        b = Value(3.)
        c = a * b
        c.backward()
        #pytorch
        at = torch.tensor([2.], requires_grad=True)
        bt = torch.tensor([3.], requires_grad=True)
        ct = at * bt
        ct.backward()
        #test 
        self.assertAlmostEqual(a.grad, at.grad)
        self.assertAlmostEqual(b.grad, bt.grad)

    def test_pow_operation(self):
        #kiwigrad
        a = Value(2.)
        b = a ** 2
        b.backward()
        #pytorch
        at = torch.tensor([2.], requires_grad=True)
        bt = at ** 2
        bt.backward()
        #test
        self.assertAlmostEqual(a.grad, at.grad)

    def test_log_operation(self):
        #kiwigrad
        a = Value(2.)
        b = a.log()
        b.backward()
        #pytorch
        at = torch.tensor([2.], requires_grad=True)
        bt = at.log()
        bt.backward()
        #test
        self.assertAlmostEqual(a.grad, at.grad)





# if __name__ == "__main__":
#     unittest.main()