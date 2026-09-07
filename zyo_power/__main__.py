# 储能应用命令行入口；旧全套实验含外部比较，正式自主入口另行提供。
"""python -m zyo_power --help"""
import argparse
import os


def main():
    parser=argparse.ArgumentParser(description='ZYO storage periodic-state experiments; MW/MWh/h; deterministic ENS')
    parser.add_argument('--suite',choices=['teaching','small','week','year-rule','year-rolling','all'],default='teaching')
    parser.add_argument('--output',required=True,help='New local directory; existing paths are never overwritten')
    parser.add_argument('--config',help='StorageCase JSON; overrides --suite')
    parser.add_argument('--mode',choices=['rule','full','rolling'],default='rule')
    parser.add_argument('--engine',choices=['native','highs','gurobi','copt'],default='native')
    parser.add_argument('--lookahead',type=int,default=48)
    parser.add_argument('--terminal-value',type=float,default=25.)
    parser.add_argument('--max-cycles',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20260907)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--strict',action='store_true',help='For --config only: forbid load shedding')
    args=parser.parse_args()
    if args.strict and not args.config:
        parser.error('--strict requires --config; predefined comparative suites use explicit diagnostic mode')
    from .experiment import run_suite,run_config
    if args.config:
        run_config(args.config,args.output,mode=args.mode,engine=args.engine,plots=not args.no_plots,
                   lookahead=args.lookahead,terminal_value=args.terminal_value,max_cycles=args.max_cycles,strict=args.strict)
    else:
        run_suite(args.output,suite=args.suite,plots=not args.no_plots,max_cycles=args.max_cycles,seed=args.seed)
    print('Artifacts:',os.path.abspath(args.output))


if __name__=='__main__':
    main()
