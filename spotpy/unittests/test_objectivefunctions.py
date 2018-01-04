import unittest
import numpy as np
import spotpy.objectivefunctions as of

class TestObjectiveFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_mask_out_missing_items(self):
        observed = np.array(range(1,6), dtype='float')
        modelled = np.array(range(1,6), dtype='float') * 1.1
        a, b = of.remove_missing_observations(observed, modelled)
        self.assertEqual(len(a), len(observed))
        self.assertEqual(len(b), len(modelled))
        for i in range(0,5):
            self.assertEqual(a[i], observed[i])
            self.assertEqual(b[i], modelled[i])
        observed[2] = np.nan
        a, b = of.remove_missing_observations(observed, modelled)
        self.assertEqual(len(a), len(observed)-1)
        self.assertEqual(len(b), len(modelled)-1)
        for i in [0,1]:
            self.assertEqual(a[i], observed[i])
            self.assertEqual(b[i], modelled[i])
        for i in [2,3]:
            self.assertEqual(a[i], observed[i+1])
            self.assertEqual(b[i], modelled[i+1])

    def test_nashsutcliffe(self):
        # > nse(1:5, 1:5)
        # [1] 1
        observed = np.array(range(1,6), dtype='float')
        modelled = np.array(range(1,6), dtype='float')
        self.assertEqual(of.nashsutcliffe(observed, modelled), 1.0)
        # values against a trusted R implementation:
        # > nse(1:5 * 1.1, 1:5)
        # [1] 0.945
        self.assertEqual(of.nashsutcliffe(observed, modelled * 1.1), 0.945)
        # > nse(1:5 * 1.1, c(1, 2, NA, 4, 5))
        # [1] 0.954
        observed[2] = np.nan
        self.assertEqual(of.nashsutcliffe(observed, modelled * 1.1), 0.954)
        # if the modelled and observed values are not of same length
        observed = np.array(range(1,8))
        self.assertEqual(of.nashsutcliffe(observed, modelled), np.nan)

if __name__ == '__main__':
    unittest.main()