# 原生 MPS 的手算模型与拒绝路径；不借助外部解析器生成预期。
import importlib.util
import unittest
from pathlib import Path

from zyo import Model


SIMPLE = """NAME sample
ROWS
 N cost
 G demand
 L cap
 E balance
COLUMNS
 x cost 2 demand 1
 x cap 1 balance 1
 y cost 3 demand 1
 y balance -1
RHS
 r demand 4 cap 3
 r balance 0 cost -5
ENDATA
"""


class TestMPS(unittest.TestCase):
    def parse(self, text):
        self.assertIsNotNone(importlib.util.find_spec('zyo.mps'), 'native MPS reader missing')
        from zyo.mps import loads_mps
        return loads_mps(text)

    def test_original_math_and_native_optimum(self):
        model = self.parse(SIMPLE)
        self.assertEqual(model.to_dict(), {
            'format': 'lzyopt-1', 'name': 'sample', 'sense': 'min',
            'variables': [{'name': 'x', 'lb': 0., 'ub': None, 'kind': 'C'},
                          {'name': 'y', 'lb': 0., 'ub': None, 'kind': 'C'}],
            'objective': {'terms': {'x': 2., 'y': 3.}, 'constant': 5.},
            'constraints': [
                {'name': 'demand', 'sense': '>=', 'terms': {'x': 1., 'y': 1.}, 'constant': -4.},
                {'name': 'cap', 'sense': '<=', 'terms': {'x': 1.}, 'constant': -3.},
                {'name': 'balance', 'sense': '==', 'terms': {'x': 1., 'y': -1.}, 'constant': 0.}]})
        result = model.solve(solver='native')
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertAlmostEqual(result.objective, 15.)

    def test_max_d_exponent_integer_marker_and_offset(self):
        model = self.parse("""NAME integer
OBJSENSE
 MAX
ROWS
 N obj
 L cap
COLUMNS
 mark 'MARKER' 'INTORG'
 x obj 2D0 cap 1
 end 'MARKER' 'INTEND'
RHS
 r cap 2.5 obj 4
BOUNDS
 UI b x 3
ENDATA
""")
        self.assertEqual(model.variables[0].kind, 'I')
        self.assertEqual((model.variables[0].lb, model.variables[0].ub), (0., 3.))
        result = model.solve(solver='native')
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.objective, 0.)

    def test_all_bounds_and_marker_defaults(self):
        model = self.parse("""NAME bounds
ROWS
 N obj
COLUMNS
 m 'MARKER' 'INTORG'
 a obj 0
 n 'MARKER' 'INTEND'
 b obj 0
 c obj 0
 d obj 0
 e obj 0
 f obj 0
 g obj 0
 h obj 0
BOUNDS
 LO set b -5
 UP set b -2
 FX set c -3
 FR set d
 MI set e
 PL set e
 BV set f
 LI set g -4
 UI set g 8
 PL set h
ENDATA
""")
        self.assertEqual([(v['lb'], v['ub'], v['kind']) for v in model.to_dict()['variables']],
                         [(0., 1., 'I'), (-5., -2., 'C'), (-3., -3., 'C'),
                          (None, None, 'C'), (None, None, 'C'), (0., 1., 'B'),
                          (-4., 8., 'I'), (0., None, 'C')])

    def test_free_negative_dense_solution(self):
        model = self.parse('NAME free\nOBJSENSE MAX\nROWS\n N obj\nCOLUMNS\n x obj 1\nBOUNDS\n MI b x\n UP b x -2\nENDATA\n')
        self.assertEqual(model.solve(solver='native').objective, -2.)

    def test_model_read_and_json_preserved(self):
        from tempfile import TemporaryDirectory
        fixture = Path(__file__).parent / 'fixtures' / 'mps' / 'hand_min.mps'
        self.assertEqual(Model.read(fixture).to_dict(), self.parse(SIMPLE).to_dict())
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'case.MPS'
            path.write_text(SIMPLE, encoding='utf-8-sig')
            self.assertEqual(Model.read(path).to_dict(), self.parse(SIMPLE).to_dict())
            model = self.parse(SIMPLE)
            path = Path(temp) / 'case.json'
            model.write(path)
            self.assertEqual(Model.read(path).to_dict(), model.to_dict())

    def test_rejects_ambiguous_unsupported_or_malformed_input(self):
        cases = [
            '', SIMPLE.replace('ENDATA', ''), SIMPLE + 'ROWS\n',
            SIMPLE.replace('ROWS', 'RANGES'), SIMPLE.replace('ROWS', 'SOS'),
            SIMPLE.replace('ROWS', 'QMATRIX'), SIMPLE.replace(' N cost', ' N cost\n N other'),
            SIMPLE.replace(' G demand', ' G demand\n G demand'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost 2 cost 1'),
            SIMPLE.replace('x cost 2 demand 1', 'x missing 2'),
            SIMPLE.replace('x cost 2 demand 1', 'cost 2 demand 1'),
            SIMPLE.replace('r demand 4 cap 3', 'r demand 4 demand 3'),
            SIMPLE.replace('r balance 0 cost -5', 's balance 0 cost -5'),
            SIMPLE.replace('r balance 0 cost -5', 'r missing 0'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost nan'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost 1e999'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost 1_000'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost 2d-9999'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost 1e-999999999999999999999'),
            SIMPLE.replace('x cost 2 demand 1', 'x cost 1e999999999999999999999'),
            SIMPLE.replace('ENDATA', 'BOUNDS\n UP b x -2\nENDATA'),
            SIMPLE.replace('ENDATA', 'BOUNDS\n LO b x 1\n LO b x 2\nENDATA'),
            SIMPLE.replace('ENDATA', 'BOUNDS\n BV b missing\nENDATA'),
            SIMPLE.replace('ENDATA', 'BOUNDS\n LO a x 0\n UP b x 3\nENDATA'),
            SIMPLE.replace('ENDATA', 'BOUNDS\n SC b x 3\nENDATA'),
            SIMPLE.replace('COLUMNS', "COLUMNS\n m 'MARKER' 'INTORG'"),
            SIMPLE.replace('COLUMNS', "COLUMNS\n m 'MARKER' 'INTEND'"),
            SIMPLE.replace('RHS', 'COLUMNS'),
            SIMPLE.replace('NAME sample', 'NAME sample with spaces'),
            SIMPLE.replace('y balance -1', 'y balance -1\n x demand 0'),
        ]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, r'line \d+'):
                    self.parse(text)


if __name__ == '__main__':
    unittest.main()
