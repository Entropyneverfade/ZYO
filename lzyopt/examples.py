# 人工构造的调度与规划示例，不是论文数据或随机充裕度模型。
"""Synthetic demonstrations, not the user's RTS/IEEE case data."""
from .model import Model, quicksum


def dispatch():
    model = Model("synthetic_single_period_dispatch")
    thermal_a = model.add_var("thermal_a_MW", ub=70)
    thermal_b = model.add_var("thermal_b_MW", ub=80)
    wind = model.add_var("wind_MW", ub=35)
    shortage = model.add_var("shortage_MW")
    model.add_constr(thermal_a + thermal_b + wind + shortage == 125, "power_balance")
    model.set_objective(25 * thermal_a + 50 * thermal_b + 10000 * shortage)
    return model


def capacity_planning():
    """Three-stage illustrative investment MILP with FIXED capacity credits.

    This is a deterministic proxy constraint, NOT a reliable-load function,
    EENS certification, endogenous ELCC, or a chronological storage model.
    Costs and data are arbitrary normalized teaching inputs.
    """
    model = Model("synthetic_three_stage_capacity_proxy")
    demand = [85, 120, 160]
    existing_firm = [40, 35, 25]
    thermal, wind, solar, battery = [], [], [], []
    for s in range(3):
        thermal.append(model.add_var(f"thermal_blocks_s{s+1}", ub=3, kind="I"))
        wind.append(model.add_var(f"wind_MW_s{s+1}", ub=100))
        solar.append(model.add_var(f"solar_MW_s{s+1}", ub=100))
        battery.append(model.add_var(f"battery_MW_s{s+1}", ub=30))
        # Thermal block: 50 MW, fixed proxy credit 0.9. All builds persist.
        proxy = quicksum(45 * thermal[t] + 0.18 * wind[t] + 0.12 * solar[t] + 0.7 * battery[t]
                         for t in range(s + 1))
        model.add_constr(existing_firm[s] + proxy >= demand[s], f"stage_{s+1}_capacity_proxy")
    model.set_objective(quicksum((0.92 ** s) * (36 * thermal[s] + 0.3 * wind[s]
                                              + 0.18 * solar[s] + 0.9 * battery[s])
                                for s in range(3)))
    return model


EXAMPLES = {"dispatch": dispatch, "capacity": capacity_planning}
