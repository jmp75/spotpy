import unittest
import numpy as np
import spotpy.objectivefunctions as of

class TestObjectiveFunctions(unittest.TestCase):

    def setUp(self):
        pass

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