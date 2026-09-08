# 静态实验图和原始绘图数据；失败计数显式列出，缺失测量绝不补零。
"""Reusable UC PNG/SVG figures; imports Matplotlib only when rendering."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


def save_json(path, data):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)


def export_trace(case, trace, path):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        fields = ['hour','demand','wind','solar','ens'] + [f'{u["name"]}_{k}' for u in case['units'] for k in ('power','on','start','stop')]
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for t,demand in enumerate(case['demand']):
            row = dict(hour=t+1,demand=demand,**{k:trace[k][t] for k in ('wind','solar','ens')})
            row.update({f'{u["name"]}_{k}':trace[k][g][t] for g,u in enumerate(case['units']) for k in ('power','on','start','stop')})
            writer.writerow(row)


def comparison_data(records):
    data = []
    # 保持预定引擎次序，不根据成功或速度事后排名。
    for engine in dict.fromkeys(r['engine'] for r in records):
        rows = [r for r in records if r['engine']==engine]
        def median(key, divisor=1):
            values = [r.get(key) for r in rows]
            values = [v/divisor for v in values if type(v) in (int,float) and math.isfinite(v) and v>=0]
            return statistics.median(values) if values else None
        data.append(dict(engine=engine,attempts=len(rows),accepted=sum(bool(r['accepted_optimal']) for r in rows),
                         statuses=dict(Counter(r['status'] for r in rows)),median_call_seconds=median('solve_call_seconds'),
                         median_rss_mib=median('peak_rss_bytes',1024**2)))
    return data


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC') if f in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':10,'svg.fonttype':'none','axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                         'axes.facecolor':'white','legend.frameon':False})
    return plt,font


def save_figure(fig, stem, font):
    stem = Path(stem)
    for suffix in ('.png','.svg','.render.json'):
        if stem.with_suffix(suffix).exists(): raise FileExistsError(stem.with_suffix(suffix))
    # 轴范围固定后将连续轴边界也标出，保存相同画布的位图与可编辑矢量文字。
    for ax in fig.axes:
        if ax.get_visible():
            ax.set_xticks(ax.get_xlim(),minor=True)
            ax.set_yticks(ax.get_ylim(),minor=True)
    fig.savefig(stem.with_suffix('.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(stem.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    save_json(stem.with_suffix('.render.json'),dict(font=font,backend='Matplotlib Agg',png_dpi=180,
              svg_text='editable',size_inches=fig.get_size_inches().tolist(),white_background=True))


def plot_dispatch(case, trace, stem, engine, status, metadata=None):
    from zyo_power.unit_commitment import validate_uc
    audit = validate_uc(case,trace)
    if not audit['physical_pass']:
        raise ValueError('Refuse dispatch figure for a physically invalid candidate')
    stem = Path(stem)
    # 图契约：上图展示真实逐时平衡，下图展示启停状态；不替失败结果生成假轨迹。
    save_json(stem.with_suffix('.data.json'),dict(case=case,trace=trace,engine=engine,status=status,audit=audit,metadata=metadata,
        contract='quantitative grid; actual candidate supply plus commitment; synthetic development; PNG/SVG editable text'))
    export_trace(case,trace,stem.with_suffix('.data.csv'))
    plt,font = plotting()
    import numpy as np
    from matplotlib.colors import ListedColormap
    hours = np.arange(1,len(case['demand'])+1)
    fig,(ax,heat) = plt.subplots(2,1,figsize=(11,6.3),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    thermal = np.sum(trace['power'],axis=0)
    ax.stackplot(hours,thermal,trace['wind'],trace['solar'],trace['ens'],
                 labels=['火电 Thermal','风电 Wind','光伏 Solar','失负荷 ENS'],colors=['#597eaa','#6aa89e','#dfbb61','#c78783'],step='mid')
    ax.plot(hours,case['demand'],color='#333333',linewidth=1.6,drawstyle='steps-mid',label='需求 Demand')
    ax.set(ylabel='功率 Power (MW)',xlim=(.5,len(hours)+.5),ylim=(0,max(case['demand'])*1.15))
    ax.set_title(f'机组组合运行 / UC dispatch — {engine} · {status}',loc='left')
    ax.legend(ncol=5,loc='upper left',fontsize=9); ax.grid(axis='y',alpha=.15)
    heat.imshow(trace['on'],aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#eeeeee','#597eaa']),
                extent=(.5,len(hours)+.5,len(case['units'])-.5,-.5),interpolation='nearest')
    heat.set_yticks(range(len(case['units'])),[u['name'] for u in case['units']])
    heat.set_xticks(hours); heat.set_xlabel('小时 Hour | 灰色=停机 off；蓝色=开机 on')
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    fig.text(.02,.015,f'Synthetic teaching/development · {len(hours)} × 1 h · physical pass · cost={audit["cost"]:.6g} · ENS={audit["ens_mwh"]:.3g} MWh',fontsize=9)
    save_figure(fig,stem,font); plt.close(fig)


def plot_comparison(records, stem, manifest):
    stem = Path(stem)
    records = sorted(records,key=lambda r:manifest['engines'].index(r['engine']))
    data = comparison_data(records)
    save_json(stem.with_suffix('.data.json'),dict(summary=data,records=records,manifest=manifest,
              contract='quantitative grid; all attempted cold calls; failures retained; no ranking; descriptive n=3'))
    with stem.with_suffix('.data.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    plt,font = plotting()
    fig,axes = plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1,1.5]},layout='constrained')
    for ax,key,label in zip(axes[:2],('median_call_seconds','median_rss_mib'),('冷求解调用 Cold call (s)','进程树峰值 Peak RSS (MiB)')):
        for i,row in enumerate(data):
            value = row[key]
            if value is not None:
                ax.barh(i,value,color='#597eaa',height=.55)
                ax.text(value,i,f' {value:.3g}',va='center',fontsize=9)
            else: ax.text(0,i,'missing',va='center')
        maximum = max((r[key] or 0 for r in data),default=1)
        ax.set_xlim(0,max(maximum*1.25,1)); ax.set_yticks(range(len(data)),[r['engine'] for r in data])
        ax.set_xlabel(label); ax.set_ylim(len(data)-.5,-.5); ax.grid(axis='x',alpha=.15)
    axes[2].set_axis_off()
    for i,row in enumerate(data):
        statuses = ', '.join(f'{k}: {v}' for k,v in row['statuses'].items())
        axes[2].text(0,1-(i+.5)/len(data),f'{row["engine"]}: accepted {row["accepted"]}/{row["attempts"]}\n{statuses}',va='center',fontsize=10)
    fig.suptitle('机组组合隔离比较 / UC isolated comparison',x=.02,ha='left')
    fig.get_layout_engine().set(rect=(0,.14,1,.80))
    scale=manifest['scale']
    fig.text(.02,.015,f'Synthetic development · {scale["periods"]} h · {scale["variables"]} vars / {scale["binary_variables"]} binary / {scale["constraints"]} rows\n'
             '3 fresh processes/engine · 1 thread · call 30 s / supervisor 60 s / 4 GiB · medians include failed calls when measured\n'
             'No warmup; process startup is separate in raw wall_seconds; missing times remain missing. No global ranking.',fontsize=9)
    save_figure(fig,stem,font); plt.close(fig)
