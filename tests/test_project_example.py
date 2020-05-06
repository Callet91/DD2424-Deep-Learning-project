"""Test example module."""
import unittest


class TestBasicFunction(unittest.TestCase):
    """Hello world test case."""

    def test_hello_world(self):
        """Hello world unittest."""
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
