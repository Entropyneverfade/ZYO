# 变量名唯一性检查的复杂度回归：必须保持 O(1)/变量，否则大模型读取会退化成 O(n^2)。
# 背景：`add_var` 曾用 `name in {v.name for v in self.variables}` 判重，每次调用重建全部名字的
# 集合，于是 MIPLIB 的 neos-4763324-toguru（53593 变量 / 106954 约束 / 266805 非零元的 MPS）
# 读取超过 900 秒且内存涨到 8 GB 以上；修好后同一文件 1.75 秒读完。
import pickle
import time
import unittest

from lzyopt.model import Model


class ModelVariableIndexTests(unittest.TestCase):
    def test_many_variables_are_added_in_linear_time(self):
        count = 50_000
        model = Model('scale')
        started = time.perf_counter()
        for index in range(count):
            model.add_var(f'x{index}', lb=0.0, ub=1.0)
        elapsed = time.perf_counter()-started
        self.assertEqual(len(model.variables), count)
        # 缺陷版本在 40000 个变量上耗时 44.8 秒（1119 us/变量）；修好后约 1.9 us/变量。
        # 20 秒的上界对"线性"有 100 倍余量，对"二次"则必然失败，因此是稳定的判别式。
        self.assertLess(elapsed, 20.0, f'adding {count} variables took {elapsed:.2f}s')
        self.assertLess(elapsed/count, 5e-5, f'{1e6*elapsed/count:.1f} us per variable')

    def test_index_stays_consistent_with_the_variable_list(self):
        model = Model('consistency')
        for index in range(500):
            model.add_var(f'v{index}')
            self.assertEqual(len(model.name_index()), len(model.variables))
        self.assertEqual(model.name_index()['v499'], 499)

    def test_duplicate_and_empty_names_are_still_rejected(self):
        model = Model('dup')
        model.add_var('a')
        with self.assertRaises(ValueError):
            model.add_var('a')
        with self.assertRaises(ValueError):
            model.add_var('')
        self.assertEqual([v.name for v in model.variables], ['a'])

    def test_default_names_are_generated_and_unique(self):
        model = Model('auto')
        for _ in range(5):
            model.add_var()
        self.assertEqual([v.name for v in model.variables],
                         ['x0', 'x1', 'x2', 'x3', 'x4'])

    def test_index_is_rebuilt_lazily_for_models_without_it(self):
        # 旧 pickle 里没有 _variable_names；恢复出的模型必须仍能加变量且判重有效。
        model = Model('legacy')
        model.add_var('p')
        restored = pickle.loads(pickle.dumps(model))
        restored.__dict__.pop('_variable_names', None)
        restored.add_var('q')
        self.assertEqual([v.name for v in restored.variables], ['p', 'q'])
        with self.assertRaises(ValueError):
            restored.add_var('p')

    def test_pickle_roundtrip_keeps_the_index_working(self):
        model = Model('roundtrip')
        model.add_var('a', lb=0.0, ub=2.0, kind='I')
        restored = pickle.loads(pickle.dumps(model))
        restored.add_var('b')
        self.assertEqual(restored.name_index(), {'a': 0, 'b': 1})
        self.assertEqual(restored.variables[0].kind, 'I')


if __name__ == '__main__':
    unittest.main()
