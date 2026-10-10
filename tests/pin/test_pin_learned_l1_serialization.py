from pathlib import Path
import json
import unittest
from unittest.mock import patch
import numpy as np
from pcb.io import write_json


class SerializationTests(unittest.TestCase):
    def test_numpy_junction_and_coordinates_preserved(self):
        raw={'junctions':[(np.int64(10),np.int64(20))],
             'point':{'x':np.float64(10.125),'y':np.float64(20.75)}}
        # Test encoding without depending on a Windows sandbox temp directory.
        with patch.object(Path,'mkdir'),patch.object(Path,'write_text') as output:
            write_json(Path('diagnostics.json'),raw)
            self.assertEqual(json.loads(output.call_args.args[0]),
                {'junctions':[[10,20]],'point':{'x':10.125,'y':20.75}})

    def test_nonfinite_still_rejected(self):
        with patch.object(Path,'mkdir'),patch.object(Path,'write_text'):
            with self.assertRaises(ValueError):
                write_json(Path('invalid.json'),{'x':np.float64(float('nan'))})


if __name__=='__main__': unittest.main()
