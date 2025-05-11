import unittest
from inferring import count_burpies, reset_counts, number_of_burpies, previous_class

class TestCountBurpies(unittest.TestCase):

    def setUp(self):
        reset_counts()
        

    def test_no_burpies(self):
        self.assertEqual(count_burpies(0), 0)
        self.assertEqual(count_burpies(0), 0)

    def test_single_burpie(self):
        self.assertEqual(count_burpies(1), 0)
        self.assertEqual(count_burpies(0), 0)
        self.assertEqual(count_burpies(1), 1)

    def test_multiple_burpies(self):
        self.assertEqual(count_burpies(1), 0)
        self.assertEqual(count_burpies(0), 0)
        self.assertEqual(count_burpies(1), 1)
        self.assertEqual(count_burpies(0), 1)
        self.assertEqual(count_burpies(1), 2)

    def test_alternating_classes(self):
        self.assertEqual(count_burpies(1), 0)
        self.assertEqual(count_burpies(0), 0)
        self.assertEqual(count_burpies(0), 0)
        self.assertEqual(count_burpies(0), 0)
        self.assertEqual(count_burpies(1), 1)
        self.assertEqual(count_burpies(1), 1)
        self.assertEqual(count_burpies(1), 1)
        self.assertEqual(count_burpies(1), 1)
        self.assertEqual(count_burpies(0), 1)
        self.assertEqual(count_burpies(1), 2)
        self.assertEqual(count_burpies(1), 2)
        self.assertEqual(count_burpies(1), 2)
        self.assertEqual(count_burpies(1), 2)
        self.assertEqual(count_burpies(1), 2)
        self.assertEqual(count_burpies(0), 2)
        self.assertEqual(count_burpies(0), 2)
        self.assertEqual(count_burpies(0), 2)
        self.assertEqual(count_burpies(0), 2)
        self.assertEqual(count_burpies(0), 2)
        self.assertEqual(count_burpies(0), 2)

if __name__ == "__main__":
    unittest.main()