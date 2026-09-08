# 自有受控 free MPS 读取器：保留原始数学，不为求解后端伪造有限界。
"""Strict whitespace-delimited linear MPS subset, without external parsers.

One NAME, one N objective, contiguous columns, one RHS/bound set. Headers
start in column one; data records are indented. Fixed-field continuations,
space-containing names, ranges, SOS and nonlinear/multi-objective data fail.
INTORG defaults to [0, 1]; other columns default to [0, +inf]. Negative UP/UI
requires an explicit lower bound to avoid differing MPS dialect conventions.
Explicit bound sides cannot overlap (e.g. FX followed by LO is rejected).
"""
import math
from pathlib import Path
import re

from lzyopt.model import Constraint, LinearExpression


_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z')
_SECTIONS = ('NAME', 'OBJSENSE', 'ROWS', 'COLUMNS', 'RHS', 'BOUNDS', 'ENDATA')


def loads_mps(text):
    """Read the documented free subset; every format error carries a line number."""
    from .model import Model

    line_no = 1

    def fail(message):
        raise ValueError(f'MPS line {line_no}: {message}')

    def number(token):
        if not _NUMBER.fullmatch(token):
            fail(f'invalid number {token!r}')
        token = token.replace('D', 'e').replace('d', 'e')
        try:
            value = float(token)
        except (ValueError, OverflowError):
            fail(f'invalid or overflowing number {token!r}')
        # 拒绝溢出及非零数下溢，避免输入时改变系数或可行域。
        # 只检查尾数数字；极长指数可能超过 Decimal 指数范围，不能用于错误判定。
        mantissa = token.lower().split('e', 1)[0]
        nonzero = any(char.isdecimal() and int(char) != 0 for char in mantissa)
        if not math.isfinite(value) or (value == 0.0 and nonzero):
            fail(f'nonfinite or underflowing number {token!r}')
        return value

    section = None
    seen = set()
    name = None
    sense = 'min'
    sense_seen = False
    rows, columns, rhs, bound_records = {}, {}, {}, {}
    objective = None
    integer = False
    previous_column = None
    rhs_set = bound_set = None
    lines = text.lstrip('\ufeff').splitlines()
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip() or raw.startswith('*'):
            continue
        fields = raw.split()
        if section == 'ENDATA':
            fail('trailing content after ENDATA')
        if not raw[0].isspace():
            header = fields[0]
            if header not in _SECTIONS:
                fail(f'unsupported section or unindented data {header!r}')
            if header in seen or (section and _SECTIONS.index(header) <= _SECTIONS.index(section)):
                fail('duplicate or out-of-order section')
            if not seen and header != 'NAME':
                fail('NAME must be first')
            if section == 'OBJSENSE' and not sense_seen:
                fail('missing objective sense')
            if section == 'COLUMNS' and integer:
                fail('unclosed INTORG marker')
            if header == 'NAME':
                if len(fields) != 2:
                    fail('NAME requires one whitespace-free name')
                name = fields[1]
            elif header == 'OBJSENSE':
                if len(fields) not in (1, 2):
                    fail('invalid OBJSENSE')
                if len(fields) == 2:
                    if fields[1] not in ('MIN', 'MAX'):
                        fail('OBJSENSE must be MIN or MAX')
                    sense, sense_seen = fields[1].lower(), True
            elif len(fields) != 1:
                fail('section header has extra fields')
            if header in ('COLUMNS', 'RHS', 'BOUNDS', 'ENDATA') and 'ROWS' not in seen:
                fail('missing ROWS')
            if header in ('RHS', 'BOUNDS', 'ENDATA') and 'COLUMNS' not in seen:
                fail('missing COLUMNS')
            if header == 'COLUMNS' and objective is None:
                fail('exactly one N objective row required')
            seen.add(header)
            section = header
            continue
        if section == 'OBJSENSE':
            if sense_seen or len(fields) != 1 or fields[0] not in ('MIN', 'MAX'):
                fail('one MIN or MAX record required')
            sense, sense_seen = fields[0].lower(), True
        elif section == 'ROWS':
            if len(fields) != 2 or fields[0] not in ('N', 'E', 'L', 'G'):
                fail('expected row type N/E/L/G and name')
            kind, row = fields
            if row in rows:
                fail(f'duplicate row {row!r}')
            if kind == 'N':
                if objective is not None:
                    fail('multiple N/objective rows unsupported')
                objective = row
            rows[row] = kind
        elif section == 'COLUMNS':
            if len(fields) == 3 and fields[1] == "'MARKER'":
                marker = fields[2]
                if marker == "'INTORG'" and not integer:
                    integer = True
                elif marker == "'INTEND'" and integer:
                    integer = False
                else:
                    fail('invalid or unbalanced integer marker')
                previous_column = None
                continue
            if len(fields) not in (3, 5):
                fail('column requires a name and one or two row/value pairs; no fixed continuation')
            col = fields[0]
            if col in columns and previous_column != col:
                fail(f'noncontiguous column {col!r}')
            if col not in columns:
                columns[col] = {'integer': integer, 'terms': {}}
            previous_column = col
            for row, token in zip(fields[1::2], fields[2::2]):
                if row not in rows:
                    fail(f'unknown row {row!r}')
                if row in columns[col]['terms']:
                    fail(f'duplicate coefficient {col!r}/{row!r}')
                columns[col]['terms'][row] = number(token)
        elif section == 'RHS':
            if len(fields) not in (3, 5):
                fail('RHS requires set name and one or two row/value pairs')
            if rhs_set is not None and rhs_set != fields[0]:
                fail('multiple RHS sets unsupported')
            rhs_set = fields[0]
            for row, token in zip(fields[1::2], fields[2::2]):
                if row not in rows or row in rhs:
                    fail(f'unknown or duplicate RHS row {row!r}')
                rhs[row] = number(token)
        elif section == 'BOUNDS':
            if len(fields) < 3:
                fail('bound requires type, set name and column')
            kind, group, col = fields[:3]
            valued = kind in ('LO', 'UP', 'FX', 'LI', 'UI')
            if kind not in ('LO', 'UP', 'FX', 'FR', 'MI', 'PL', 'BV', 'LI', 'UI'):
                fail(f'unsupported bound type {kind!r}')
            if len(fields) != (4 if valued else 3):
                fail('incorrect bound field count')
            if col not in columns:
                fail(f'unknown bound column {col!r}')
            if bound_set is not None and bound_set != group:
                fail('multiple bound sets unsupported')
            bound_set = group
            bounds = bound_records.setdefault(col, {})
            sides = ('lb', 'ub') if kind in ('FX', 'FR', 'BV') else (('lb',) if kind in ('LO', 'LI', 'MI') else ('ub',))
            if any(side in bounds for side in sides):
                fail(f'duplicate or overlapping bounds for {col!r}')
            value = number(fields[3]) if valued else None
            for side in sides:
                bounds[side] = (kind, value, line_no)
        else:
            fail('data outside a supported section')

    line_no = len(lines) + 1
    if section != 'ENDATA':
        fail('missing ENDATA')
    model = Model(name)
    row_terms = {row: {} for row in rows}
    for col, data in columns.items():
        kind = 'I' if data['integer'] else 'C'
        lb, ub = 0., 1. if data['integer'] else math.inf
        bounds = bound_records.get(col, {})
        for side, (code, value, bound_line) in bounds.items():
            if code in ('LI', 'UI'):
                kind = 'I'
            if code == 'BV':
                kind = 'B'
            if side == 'lb':
                lb = -math.inf if code in ('FR', 'MI') else (0. if code == 'BV' else value)
            else:
                ub = math.inf if code in ('FR', 'PL') else (1. if code == 'BV' else value)
                if code in ('UP', 'UI') and ub < 0 and 'lb' not in bounds:
                    line_no = bound_line
                    fail('negative upper bound requires explicit lower bound (LO/LI/MI/FR)')
        variable = model.add_var(col, lb=lb, ub=ub, kind=kind)
        for row, value in data['terms'].items():
            row_terms[row][variable.index] = value
    # MPS 目标行 RHS 是移到左边的常数，必须取负；其他行同样存表达式与零比较。
    model.set_objective(LinearExpression(model, row_terms[objective], -rhs.get(objective, 0.)), sense)
    for row, kind in rows.items():
        if kind != 'N':
            expression = LinearExpression(model, row_terms[row], -rhs.get(row, 0.))
            model.add_constr(Constraint(expression, {'E': '==', 'L': '<=', 'G': '>='}[kind]), row)
    return model


def read_mps(path):
    """Read UTF-8 (optional BOM) text from a local file."""
    return loads_mps(Path(path).read_text(encoding='utf-8-sig'))
