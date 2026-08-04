from __future__ import annotations

import importlib.util
import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_python_package_layout(self) -> None:
        import ioc

        self.assertIsNotNone(ioc.__doc__)
        self.assertIsNotNone(importlib.util.find_spec("ioc.DDIOC.pipeline"))
        self.assertIsNotNone(importlib.util.find_spec("ioc.DDIOC.validate_dnn"))
        self.assertIsNotNone(importlib.util.find_spec("ioc.LIOC.linear_ioc_bilevel"))

    def test_standalone_example_ground_truth(self) -> None:
        from examples.synthetic_ioc_demo import get_ground_truth_hlo

        hlo = get_ground_truth_hlo()
        self.assertEqual(len(hlo.feature_names), 10)
        self.assertAlmostEqual(float(sum(hlo.omega)), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()