import importlib.util
import tempfile
import unittest
from pathlib import Path
from ovr.analysis import agreement
from ovr.common import ROOT, write_json


class AnalysisTests(unittest.TestCase):
    def test_independent_agreement_and_uncertainty(self):
        a = agreement(["1", "1", "0", "0", "uncertain"], ["1", "0", "0", "1", "1"])
        self.assertEqual(a["binary_pairs"], 4)
        self.assertEqual(a["raw_agreement"], .5)
        self.assertEqual(a["cohen_kappa"], 0)

    def test_constant_labels_do_not_invent_kappa(self):
        a = agreement(["0", "0"], ["0", "0"])
        self.assertEqual(a["raw_agreement"], 1)
        self.assertIsNone(a["cohen_kappa"])

    def test_empty_agreement(self):
        self.assertIsNone(agreement([], [])["cohen_kappa"])


class ReleaseGateTests(unittest.TestCase):
    def test_preparation_cannot_pass_final_submission_gate(self):
        spec = importlib.util.spec_from_file_location("packaging_script", ROOT / "scripts/package_project.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        errors = module.submission_errors(ROOT)
        self.assertTrue(any("3040" in e for e in errors))
        self.assertTrue(any("annotations" in e.lower() for e in errors))
        self.assertTrue(any("author" in e.lower() for e in errors))


if __name__ == "__main__":
    unittest.main()
