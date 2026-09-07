# 历史命令行入口；仅保留兼容，不作为新开发的推荐入口。
# Legacy CLI entry point; prefer the public zyo command for new work.
from .cli import main

raise SystemExit(main())
