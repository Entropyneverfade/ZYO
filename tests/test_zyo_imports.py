# 回归测试：检查大小写导入与旧名称兼容，不加载不必要的外部引擎。
import importlib
import unittest


class ZYOImportTests(unittest.TestCase):
    def test_lower_and_uppercase_imports_share_public_symbols(self):
        lower = importlib.import_module("zyo")
        upper = importlib.import_module("ZYO")

        self.assertEqual(lower.__version__, "0.3.4")
        self.assertIs(upper.Model, lower.Model)
        self.assertEqual(lower.BINARY, "B")
        self.assertEqual(lower.Status.OPTIMAL.value, "OPTIMAL")

    def test_case_insensitive_tokens_are_normalized(self):
        constants = importlib.import_module("zyo.constants")

        self.assertEqual(constants.normalize_vtype("binary"), "B")
        self.assertEqual(constants.normalize_vtype("Bin"), "B")
        self.assertEqual(constants.normalize_vtype("i"), "I")
        self.assertEqual(constants.normalize_sense("MAXIMIZE"), "max")
        with self.assertRaises(ValueError):
            constants.normalize_vtype("semi-magical")


if __name__ == "__main__":
    unittest.main()
