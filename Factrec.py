import unittest

# Function to calculate factorial
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Test Case for factorial function
class TestFactorial(unittest.TestCase):
    
    def test_factorial_positive(self):
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(3), 6)
    
    def test_factorial_zero(self):
        self.assertEqual(factorial(0), 1)
    
    def test_factorial_one(self):
        self.assertEqual(factorial(1), 1)

    def test_factorial_large(self):
        self.assertEqual(factorial(10), 3628800)

if __name__ == '__main__':
    unittest.main()
