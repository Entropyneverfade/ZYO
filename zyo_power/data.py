# 固定容量时序数据契约：功率 MW、能量 MWh、步长小时，校验形状与物理参数。
"""Fixed-capacity storage data in MW, MWh and hours."""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Battery:
    name: str
    energy_max: float
    charge_max: float
    discharge_max: float
    eta_charge: float = .95
    eta_discharge: float = .95
    energy_min: float = 0.
    throughput_cost: float = 0.

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError('Battery name must be nonempty')
        for key in asdict(self):
            if key != 'name':
                value = float(getattr(self, key))
                if not math.isfinite(value):
                    raise ValueError(f'{key} must be finite')
                object.__setattr__(self, key, value)
        if not 0 <= self.energy_min <= self.energy_max:
            raise ValueError('Require 0 <= energy_min <= energy_max')
        if min(self.charge_max, self.discharge_max, self.throughput_cost) < 0:
            raise ValueError('Power limits and throughput cost must be nonnegative')
        if not (0 < self.eta_charge <= 1 and 0 < self.eta_discharge <= 1):
            raise ValueError('Efficiencies must be in (0,1]')


def _array(value, shape, name):
    try:
        array = np.array(np.broadcast_to(np.asarray(value, dtype=float), shape), copy=True)
    except ValueError as exc:
        raise ValueError(f'{name} must broadcast to {shape}') from exc
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StorageCase:
    name: str
    load: object
    renewable: object
    batteries: object
    thermal_capacity: object = 0.
    thermal_cost: object = 0.
    dt: object = 1.
    retention: object = 1.
    shed_penalty: float = 10000.
    charging_source: str = 'renewable_only'
    source: str = 'synthetic'

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError('Case name must be nonempty')
        load = np.asarray(self.load, dtype=float)
        if load.ndim != 1 or len(load) == 0:
            raise ValueError('load must be a nonempty time vector')
        batteries = tuple(self.batteries)
        if not batteries or not all(isinstance(b, Battery) for b in batteries):
            raise ValueError('At least one Battery is required')
        if len({b.name for b in batteries}) != len(batteries):
            raise ValueError('Battery names must be unique')
        object.__setattr__(self, 'batteries', batteries)
        for key in ['load', 'renewable', 'thermal_capacity', 'thermal_cost', 'dt']:
            array = _array(getattr(self, key), (len(load),), key)
            if np.any(array < 0) or (key == 'dt' and np.any(array <= 0)):
                raise ValueError(f'{key} contains invalid negative/zero values')
            object.__setattr__(self, key, array)
        retention = np.asarray(self.retention, dtype=float)
        if retention.ndim == 1 and retention.shape == (len(load),):
            retention = retention[:, None]
        retention = _array(retention, (len(load), len(batteries)), 'retention')
        if np.any((retention < 0) | (retention > 1)):
            raise ValueError('Retention per interval must be in [0,1]')
        object.__setattr__(self, 'retention', retention)
        if not math.isfinite(self.shed_penalty) or self.shed_penalty < 0:
            raise ValueError('ENS penalty must be finite and nonnegative')
        if self.charging_source not in ['renewable_only', 'any_supply']:
            raise ValueError('Unknown charging_source')

    @property
    def periods(self):
        return len(self.load)

    def window(self, start, length):
        if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length <= 0:
            raise ValueError('Window start/length must be nonnegative/positive integers')
        indices = (start + np.arange(length)) % self.periods
        return StorageCase(self.name, self.load[indices], self.renewable[indices], self.batteries,
                           thermal_capacity=self.thermal_capacity[indices], thermal_cost=self.thermal_cost[indices],
                           dt=self.dt[indices], retention=self.retention[indices], shed_penalty=self.shed_penalty,
                           charging_source=self.charging_source, source=self.source)

    def to_dict(self):
        return dict(name=self.name, batteries=[asdict(b) for b in self.batteries],
                    **{k:getattr(self,k).tolist() for k in ['load','renewable','thermal_capacity','thermal_cost','dt','retention']},
                    shed_penalty=self.shed_penalty, charging_source=self.charging_source, source=self.source)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['batteries'] = [Battery(**b) for b in data['batteries']]
        return cls(**data)

    def initial(self, energy, *, atol_energy=1e-7, rtol_energy=1e-9):
        """Validate and copy, NEVER clip a propagated numerical boundary state."""
        if not np.isfinite([atol_energy,rtol_energy]).all() or min(atol_energy,rtol_energy)<0:
            raise ValueError('Initial-state tolerances must be finite and nonnegative')
        energy = np.asarray(energy, dtype=float)
        if energy.shape != (len(self.batteries),) or not np.isfinite(energy).all():
            raise ValueError('Initial energy must give one finite MWh value per battery')
        if any(not b.energy_min-(atol_energy+rtol_energy*b.energy_max) <= e <=
               b.energy_max+(atol_energy+rtol_energy*b.energy_max) for e,b in zip(energy,self.batteries)):
            raise ValueError('Initial energy outside bounds')
        return energy.copy()
