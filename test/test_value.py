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
        at_g = float(at.grad)
        bt_g = float(bt.grad)
        #test
        self.assertAlmostEqual(a.grad, at_g)
        self.assertAlmostEqual(b.grad, bt_g)

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
        at_g = float(at.grad)
        bt_g = float(bt.grad)
        #test 
        self.assertAlmostEqual(a.grad, at_g)
        self.assertAlmostEqual(b.grad, bt_g)

    def test_pow_operation(self):
        #kiwigrad
        a = Value(2.)
        b = a ** 2
        b.backward()
        #pytorch
        at = torch.tensor([2.], requires_grad=True)
        bt = a ** 2
        bt.backward()
        bt_g = float(bt.grad)
        #test
        self.assertAlmostEqual(b.grad, bt_g)




# if __name__ == "__main__":
#     unittest.main()