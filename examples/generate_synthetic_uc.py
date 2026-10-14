"""独立生成公开教学用机组组合输入；不读取任何第三方或私有算例。"""

import argparse
import json
import math
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validated_output_path(path):
    """只向本项目归档写入，拒绝越界路径及覆盖既有证据。"""
    output = Path(path).resolve()
    if not output.is_relative_to(PROJECT_ROOT) or output.suffix.lower() != '.json':
        raise ValueError('输出必须是 ZYO 项目内的 JSON 文件')
    if output.exists():
        raise FileExistsError('拒绝覆盖已有算例文件')
    return output


def generate_case(seed=20261014, profile='balanced'):
    """只由公开公式、固定常数与种子生成 24 个小时级 MW 时段。"""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed 必须为非负整数')
    if profile not in ('balanced', 'tight_reserve'):
        raise ValueError('profile 只允许 balanced 或 tight_reserve')
    rng = random.Random(seed)
    demand, wind, solar, reserve = [], [], [], []
    for hour in range(24):
        # 自写昼夜谐波与伪随机扰动；数值不来自真实负荷/气象/论文资料。
        daily = math.cos(2*math.pi*(hour-19)/24)
        load = 75+14*daily+3*(rng.random()-0.5)
        breeze = 17+5*math.cos(2*math.pi*(hour+3)/24)+4*(rng.random()-0.5)
        sunlight = 24*max(0, math.sin(math.pi*(hour-6)/12))
        if profile == 'tight_reserve':
            load += 5
            breeze *= 0.8
        demand.append(round(load, 3))
        wind.append(round(max(0, breeze), 3))
        solar.append(round(sunlight, 3))
        reserve.append(round(5+0.06*load+(3 if profile == 'tight_reserve' else 0), 3))
    return {
        'name': f'ZYO-public-synthetic-UC-{profile}-seed-{seed}',
        'source': (
            f'ZYO self-authored formula v1; Python random.Random seed={seed}; '
            f'profile={profile}; 24 hourly MW periods; standalone generated teaching input'),
        'demand': demand, 'wind': wind, 'solar': solar, 'reserve': reserve,
        'strict': True, 'ens_cost': 10000,
        'units': [
            {'name': 'Synthetic-A', 'p_min': 12, 'p_max': 80,
             'marginal_cost': 27, 'no_load_cost': 8,
             'startup_cost': 90, 'shutdown_cost': 15,
             'ramp_up': 24, 'ramp_down': 24,
             'startup_ramp': 45, 'shutdown_ramp': 45,
             'min_up': 3, 'min_down': 2,
             'initial_on': 1, 'initial_duration': 4, 'initial_power': 55},
            {'name': 'Synthetic-B', 'p_min': 8, 'p_max': 50,
             'marginal_cost': 46, 'no_load_cost': 5,
             'startup_cost': 55, 'shutdown_cost': 8,
             'ramp_up': 30, 'ramp_down': 30,
             'startup_ramp': 30, 'shutdown_ramp': 30,
             'min_up': 2, 'min_down': 2,
             'initial_on': 0, 'initial_duration': 3, 'initial_power': 0}
        ]
    }


def serialize_case(data):
    """固定 UTF-8 与 CRLF 字节，跨平台输出可与随附 JSON 原件核对。"""
    return (json.dumps(data, ensure_ascii=False, indent=2).replace('\n', '\r\n')+'\r\n').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--profile', choices=('balanced', 'tight_reserve'),
                        default='balanced')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = validated_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_case(args.seed, args.profile)
    output.write_bytes(serialize_case(data))
    print(json.dumps({'output': str(output), 'profile': args.profile,
                      'seed': args.seed, 'hours': 24}, ensure_ascii=False))


if __name__ == '__main__':
    main()
