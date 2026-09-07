# 教学例及明确标注的合成时序；随机种子可复现，但预热轮次不是独立气象样本。
"""Deterministic teaching data and explicitly synthetic development cases."""
import numpy as np
from .data import Battery, StorageCase


def teaching_case():
    net = np.repeat([5., -2., 3., -4.], 6)
    return StorageCase('teaching_24h', np.maximum(-net, 0), np.maximum(net, 0),
                       [Battery('battery', 40, 5, 5)],
                       source='user-specified deterministic teaching net surplus; not weather observations')


def tiny_case():
    # eta=1: store one MWh, supply one of the two MWh demand; thermal cost=4.
    return StorageCase('analytical_2h', [0, 2], [1, 0],
                       [Battery('battery', 2, 1, 1, 1, 1)],
                       thermal_capacity=2, thermal_cost=4,
                       source='constructed analytical development instance; known optimum 4')


def synthetic_case(hours=168, seed=20260907):
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
        raise ValueError('hours must be a positive integer')
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    hour = t % 24
    load = 10 + 2*np.cos(2*np.pi*(hour-19)/24) + .4*rng.uniform(-1, 1, hours)
    wind = np.maximum(0, 4 + 3*np.cos(2*np.pi*t/97) + .7*rng.uniform(-1, 1, hours))
    solar = 10*np.maximum(0, np.sin(np.pi*(hour-6)/12))
    cost = np.where((hour >= 17) & (hour < 23), 90., 35.)
    case = StorageCase(f'synthetic_{hours}h', load, wind+solar,
                       [Battery('battery', 32, 5, 5, throughput_cost=.02)],
                       thermal_capacity=8, thermal_cost=cost, retention=.9995,
                       charging_source='any_supply',
                       source=f'synthetic formula v1; numpy.default_rng seed={seed}; development set, not real weather')
    return case, {'wind': wind, 'solar': solar}
