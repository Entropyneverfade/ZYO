# 从已保存时序生成供需、SOC 和收敛图；数值验收仍以原始数据及独立检查为准。
"""Static local figures; raw chart data are retained in adjacent CSV/JSON."""
import numpy as np


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#282828','text.color':'#282828',
                         'axes.edgecolor':'#666666','grid.color':'#dddddd'})
    return plt


def plot_dispatch(case,result,path):
    plt=_pyplot()
    trace=result['trace']
    elapsed=np.r_[0,np.cumsum(case.dt)]
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    fig.suptitle(f"{case.name}: {result['mode']} / {result['engine']}\n"
                 f"{case.periods} intervals; fixed capacity; deterministic data, not probability samples",fontsize=13)
    ax=axes[0]
    for series,label,color,style in [(case.load,'Load','#333333','-'),
             (case.renewable,'Available renewable','#2471A3','-'),
             (trace['thermal'],'Thermal','#D9822B','--')]:
        ax.stairs(series,elapsed,label=label,color=color,linestyle=style)
    ax.set_ylabel('Supply / load (MW)')
    ax.legend(loc='upper right',ncol=3,fontsize=9)
    ax=axes[1]
    for series,label,color,style in [(-np.sum(trace['charge'],axis=1),'Charge (-)','#2471A3','-'),
             (np.sum(trace['discharge'],axis=1),'Discharge (+)','#D9822B','--'),
             (trace['shed'],'Unserved load','#333333',':')]:
        ax.stairs(series,elapsed,label=label,color=color,linestyle=style)
    ax.axhline(0,color='#777777',linewidth=.7)
    ax.set_ylabel('Storage / shortage (MW)')
    ax.legend(loc='upper right',ncol=3,fontsize=9)
    ax=axes[2]
    for s,battery in enumerate(case.batteries):
        ax.plot(elapsed,trace['energy'][:,s],color='#2471A3',linestyle=['-','--',':','-.'][s%4],label=battery.name)
        ax.axhline(battery.energy_max,color='#777777',linestyle=':',linewidth=.8)
        ax.axhline(battery.energy_min,color='#777777',linestyle=':',linewidth=.8)
    ax.set_ylabel('Stored energy (MWh)')
    ax.set_xlabel('Elapsed time (hours); energy includes initial and final timepoints')
    ax.legend(loc='upper right',fontsize=9)
    for ax in axes:
        ax.grid(axis='y',alpha=.6)
        ax.set_xlim(elapsed[0],elapsed[-1])
    fig.savefig(path,dpi=140)
    plt.close(fig)


def plot_convergence(case,study,path):
    plt=_pyplot()
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True,layout='constrained')
    fig.suptitle(f'{case.name}: boundary residual by virtual cycle\nDiscrete iterations; not independent weather years',fontsize=12)
    maximum=max([float(v) for run in study['runs'] for h in run['history'] for v in h['residual_mwh']]+[1e-6])
    for ax,label,run in zip(axes,['Low initial energy','Mid initial energy','High initial energy'],study['runs']):
        for s,battery in enumerate(case.batteries):
            x=[h['cycle'] for h in run['history']]
            y=[h['residual_mwh'][s] for h in run['history']]
            ax.scatter(x,y,facecolors='none',edgecolors='#2471A3',marker=['o','s','^'][s%3],label=battery.name)
            ax.axhline(run['energy_tolerance_mwh'][s],color='#D9822B',linestyle='--',linewidth=1)
        ax.set_title(label,fontsize=10)
        ax.set_xlabel('Virtual cycle number')
        ax.grid(axis='y',alpha=.5)
        ax.set_ylim(-.02*maximum,1.15*maximum)
        if x:
            ax.set_xticks(x)
        ax.text(.98,.98,run['status'],transform=ax.transAxes,va='top',ha='right',fontsize=7)
    axes[0].set_ylabel('|End - start energy| (MWh)\nDashed line: configured closure tolerance')
    fig.savefig(path,dpi=140)
    plt.close(fig)
