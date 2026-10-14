"""自写公开机组组合生成器：确定性、结构和私有输入隔离。"""

import unittest
import json
from pathlib import Path

from examples.generate_synthetic_uc import PROJECT_ROOT, generate_case, serialize_case, validated_output_path
from zyo_power.unit_commitment import validate_uc_input


class PublicSyntheticUCTests(unittest.TestCase):
    def test_reproducible_and_schema_valid(self):
        first = generate_case(20261014, 'balanced')
        self.assertEqual(first, generate_case(20261014, 'balanced'))
        self.assertEqual(validate_uc_input(first), 24)
        self.assertEqual(len(first['units']), 2)
        self.assertTrue(first['strict'])

    def test_seed_and_profile_change_the_series(self):
        original = generate_case(20261014, 'balanced')
        other_seed = generate_case(20261015, 'balanced')
        stressed = generate_case(20261014, 'tight_reserve')
        self.assertNotEqual(original['demand'], other_seed['demand'])
        self.assertNotEqual(original['demand'], stressed['demand'])
        self.assertTrue(all(x >= 0 for key in ('wind', 'solar', 'reserve')
                            for x in stressed[key]))

    def test_bad_seed_or_profile_rejected(self):
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_case(seed)
        with self.assertRaises(ValueError):
            generate_case(1, 'unknown')

    def test_bundled_inputs_reproduce_exact_generator_output(self):
        # 公开 JSON 必须与当前生成器逐字段一致，不能事后手改序列。
        examples = Path(__file__).resolve().parents[1] / 'examples'
        for profile in ('balanced', 'tight_reserve'):
            with self.subTest(profile=profile):
                source = examples / f'synthetic_uc_{profile}_20261014.json'
                actual = json.loads(source.read_text(encoding='utf-8'))
                self.assertEqual(actual, generate_case(20261014, profile))
                self.assertEqual(source.read_bytes(), serialize_case(actual))

    def test_generator_rejects_output_outside_archive(self):
        with self.assertRaises(ValueError):
            validated_output_path(PROJECT_ROOT.parent / 'outside.json')


if __name__ == '__main__':
    unittest.main()
