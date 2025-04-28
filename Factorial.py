```python
import unittest

# Calculates the factorial of a given number using recursion
def factorial(n):
if n == 0 or n == 1:
return 1
else:
return n * factorial(n - 1)

# Defines a test case class for the factorial function
class TestFactorial(unittest.TestCase):

# Tests the factorial function with positive numbers
def test_factorial_positive(self):
self.assertEqual(factorial(5), 120)
self.assertEqual(factorial(3), 6)

# Tests the factorial function with zero
def test_factorial_zero(self):
self.assertEqual(factorial(0), 1)

# Tests the factorial function with one
def test_factorial_one(self):
self.assertEqual(factorial(1), 1)

# Tests the factorial function with a large number
def test_factorial_large(self):
self.assertEqual(factorial(10), 3628800)

# Runs the test cases when the script is executed
if __name__ == '__main__':
unittest.main()
```