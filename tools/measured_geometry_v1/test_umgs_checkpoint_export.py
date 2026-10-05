import unittest
import numpy as np

from .umgs_checkpoint_export import require_locked_support, support_identity


class SupportTest(unittest.TestCase):
    def setUp(self):
        self.values = {key: np.arange(4*width, dtype=np.float32).reshape(4, width)
                       for key, width in zip(('xyz','scaling','rotation','opacity'), (3,3,4,1))}

    def test_exact_support_pass(self):
        self.assertEqual(require_locked_support(self.values, {k:v.copy() for k,v in self.values.items()})['gaussian_count'], 4)

    def test_each_support_property_tamper_rejected(self):
        for key in self.values:
            with self.subTest(key=key):
                other={k:v.copy() for k,v in self.values.items()};other[key][0,0]+=.001
                with self.assertRaises(ValueError):require_locked_support(self.values,other)

    def test_order_count_dtype_and_nonfinite_rejected(self):
        variants=[{k:v[::-1] for k,v in self.values.items()},
                  {k:v[:-1] for k,v in self.values.items()},
                  {k:v.astype(np.float64) for k,v in self.values.items()}]
        bad={k:v.copy() for k,v in self.values.items()};bad['xyz'][0,0]=np.nan;variants.append(bad)
        for other in variants:
            with self.assertRaises(ValueError):require_locked_support(self.values,other)

    def test_missing_support_property_rejected(self):
        other=dict(self.values);other.pop('opacity')
        with self.assertRaises(ValueError):support_identity(other)


if __name__ == '__main__':unittest.main()
