"""pytest設定ファイル

テストモジュールがソースコードをimportできるようにパスを設定
"""

import sys
from pathlib import Path

# プロジェクトルートの04_srcをPythonパスに追加
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "04_src"
sys.path.insert(0, str(src_path))
