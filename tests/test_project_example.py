#The test file should be on the form test_.py
import unittest #Library for unittesting
from directory.basicfunction import BasicFunction #Function to test
 
class TestBasicFunction(unittest.TestCase):
    def setUp(self):
        self.func = BasicFunction()
 
    def test_1(self):
        self.assertTrue(True)
 
    def test_2(self):
        self.assertTrue(True)
 
    def test_3(self):
        self.assertEqual(self.func.state, 0)

if __name__ == '__main__':
    unittest.main()