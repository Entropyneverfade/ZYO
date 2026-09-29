# 本图实际后端：Origin 试绘导出带 demo 水印，使用 Python 回退。
# 仅依据同目录实测 CSV 绘制，不重算、平滑或调整求解结果。
"""依据本版本实测 CSV，用 matplotlib 重绘矩阵缓冲区字节比。"""

import csv
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source_data'/'matrix_storage.csv'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='new output directory; shipped figure remains untouched')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    rows = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig', newline='')))
    if len(rows) != 4 or any(row['matrices_equal'] != 'True' for row in rows):
        raise ValueError('The preregistered four-case matrix identity evidence is missing')
    sizes = [int(row['rows']) for row in rows]
    if sizes != [200, 600, 1000, 1420]:
        raise ValueError('The preregistered sizes were changed')
    ratios = [int(row['dense_bytes'])/int(row['csc_bytes']) for row in rows]
    rcParams.update({'font.family': 'Times New Roman', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})
    figure, axis = plt.subplots(figsize=(168/25.4, 95/25.4), dpi=150)
    figure.patch.set_facecolor('white')
    axis.set_facecolor('white')
    axis.plot(sizes, ratios, color='#346699', marker='o', markersize=5,
              linewidth=1.6, label='Dense / CSC')
    axis.set(xlim=(150, 1470), ylim=(0, 800), xlabel='LP dimension (rows = columns)',
             ylabel='Dense / CSC matrix bytes')
    axis.set_xticks(sizes)
    axis.set_yticks([0, 200, 400, 600, 800])
    axis.tick_params(direction='in', width=0.7, length=4, labelsize=8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
    axis.legend(loc='upper left', frameon=False, fontsize=8)
    figure.tight_layout(pad=0.8)
    # 正式用户重绘必须选新目录；打包时仅在源图尚无成品的前提下允许本图根目录。
    if output.exists():
        if output != ROOT or (output/'images').exists() or (output/'qa').exists():
            raise FileExistsError(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
    images = output/'images'
    images.mkdir(exist_ok=False)
    for extension in ('pdf', 'svg', 'png'):
        target = images/f'Fig01_MatrixStorage.{extension}'
        if target.exists():
            raise FileExistsError(target)
        figure.savefig(target, dpi=600, facecolor='white')
        if extension == 'svg':
            # Matplotlib路径数据的行尾空格只影响格式；清除后便于Git差异检查。
            normalized = '\n'.join(line.rstrip() for line in target.read_text('utf-8').splitlines())+'\n'
            with target.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write(normalized)
    plt.close(figure)
    qa = output/'qa'
    qa.mkdir(exist_ok=False)
    report = {'renderer': 'Python / matplotlib',
              'fallback_reason': 'The attempted Origin export had a demo watermark; the published renderer is Python/matplotlib.',
              'data_check': 'PASS', 'export_check': 'PASS', 'visual_review': 'PENDING',
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'sizes': sizes, 'dense_to_csc_ratios': ratios,
              'claim': 'Matrix buffer bytes only: NumPy dense nbytes divided by CSC data+indices+indptr bytes. Not solver time or process RSS.'}
    with (qa/'figure_qa.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'renderer': report['renderer'], 'ratios': ratios}, ensure_ascii=False))


if __name__ == '__main__':
    main()
