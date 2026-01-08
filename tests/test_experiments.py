import unittest

from experiments import Experiment1, Experiment2, Experiment3


class TestExperiments(unittest.TestCase):
    def test_experiment1_run(self):
        exp = Experiment1()
        res = exp.run()
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("experiment"), "Experiment1")

    def test_experiment2_run(self):
        exp = Experiment2()
        res = exp.run()
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("experiment"), "Experiment2")

    def test_experiment3_run(self):
        exp = Experiment3()
        res = exp.run()
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("experiment"), "Experiment3")


if __name__ == "__main__":
    unittest.main()
