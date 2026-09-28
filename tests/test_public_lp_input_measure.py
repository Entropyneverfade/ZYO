# 公共稀疏入口的教学测量使用手算矩阵，不从被测实现推导期望字节数。
import unittest
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from zyo.status import Status

# 隔离模式 -I 不把导出树根目录加入 sys.path；按发行文件路径加载测量器，求解包仍取净安装版本。
_SCRIPT = Path(__file__).resolve().parents[1]/'tools'/'measure_public_lp_input.py'
_SPEC = importlib.util.spec_from_file_location('zyo_public_input_measure', _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
measure_case = _MODULE.measure_case


class PublicLpInputMeasureTests(unittest.TestCase):
    def test_diagonal_input_has_hand_checked_matrix_storage_and_certified_solution(self):
        # 若漏计CSC列指针、错误构造矩阵或把稠密入口偷换成外部引擎，此测试必须失败。
        row = measure_case(3, solve=True)
        self.assertEqual(row['rows'], 3)
        self.assertEqual(row['columns'], 3)
        self.assertEqual(row['nonzeros'], 3)
        self.assertEqual(row['dense_bytes'], 72)
        self.assertEqual(row['csc_bytes'], 52)  # 3×float64 + 3×int32 + 4×int32
        self.assertTrue(row['matrices_equal'])
        self.assertEqual(row['status'], 'OPTIMAL')
        self.assertEqual(row['objective'], 0.0)
        self.assertTrue(row['certificate_verified'])
        self.assertFalse(row['fallback_used'])

    def test_zero_dimension_is_rejected_before_building_a_model(self):
        with self.assertRaises(ValueError):
            measure_case(0)

    def test_failed_public_solve_does_not_publish_measurement_files(self):
        # 固定教学LP正常可解析；仅故障注入求解返回值以覆盖真实数值失败的报告门。
        failure = SimpleNamespace(status=Status.NUMERICAL_ERROR, objective=None,
                                  metadata={'certificate': None, 'fallback_used': False},
                                  solver_name='native_simplex', solver_version='0.3.6', runtime=0.0)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'rejected'
            with patch.object(_MODULE.Model, 'solve', return_value=failure):
                with self.assertRaises(RuntimeError):
                    _MODULE.main(['--output', str(destination)])
            self.assertFalse(destination.exists(), '失败求解不能留下看似有效的发布数据目录')

    def test_measurement_serialization_has_stable_lf_line_endings(self):
        # 源表进入 Git 后字节不得因 Windows 自动换行转换而改变，便于 QA 哈希跨平台复算。
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'measured'
            # 全套回归会导入对照引擎；新进程才能独立验证本脚本的原生隔离门。
            process = subprocess.run([sys.executable, '-B', str(_SCRIPT), '--output', str(output)],
                                     capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(process.returncode, 0, process.stderr)
            for name in ('matrix_storage.csv', 'measurements.json'):
                data = (output/name).read_bytes()
                self.assertNotIn(b'\r\n', data, name)
                self.assertIn(b'\n', data, name)

    def test_renderer_can_redraw_into_new_directory_without_touching_bundled_figure(self):
        # 用户从发行源码重绘时必须有独立输出目录，不覆盖本版已经发布的图。
        figure = Path(__file__).resolve().parents[1]/'docs'/'figures'/'0.3.6'/'Fig01_MatrixStorage'
        if not figure.exists():
            figure = Path(__file__).resolve().parents[1]/'docs'/'public'/'figures'/'0.3.6'/'Fig01_MatrixStorage'
        shipped = figure/'images'/'Fig01_MatrixStorage.svg'
        original = shipped.read_bytes()
        csv_bytes = (figure/'source_data'/'matrix_storage.csv').read_bytes()
        shipped_qa = json.loads((figure/'qa'/'figure_qa.json').read_text('utf-8'))
        self.assertNotIn(b'\r\n', csv_bytes)
        self.assertEqual(shipped_qa['source_sha256'], hashlib.sha256(csv_bytes).hexdigest())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'redrawn'
            command = [sys.executable, '-B', str(figure/'scripts'/'plot_python.py'),
                       '--output', str(output)]
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                     errors='replace')
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue((output/'images'/'Fig01_MatrixStorage.svg').is_file())
            qa = json.loads((output/'qa'/'figure_qa.json').read_text('utf-8'))
            self.assertEqual(qa['source_sha256'], hashlib.sha256(csv_bytes).hexdigest())
            self.assertNotIn(b'\r\n', (output/'images'/'Fig01_MatrixStorage.svg').read_bytes())
            self.assertNotIn(b'\r\n', (output/'qa'/'figure_qa.json').read_bytes())
            self.assertEqual(shipped.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
