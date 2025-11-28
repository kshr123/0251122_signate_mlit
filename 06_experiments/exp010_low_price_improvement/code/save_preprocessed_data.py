"""
既存のrun_dirに前処理済みデータを保存する一時スクリプト
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "04_src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl
import yaml
from sklearn.model_selection import KFold
from data.loader import DataLoader
from preprocessing import preprocess_for_training

run_dir = Path(__file__).parent.parent / "outputs" / "run_20251128_090118"

print("Loading data...")
data_config_path = project_root / "03_configs" / "data.yaml"
with open(data_config_path, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

data_config["data"]["train_path"] = str(project_root / data_config["data"]["train_path"])
data_config["data"]["test_path"] = str(project_root / data_config["data"]["test_path"])
data_config["data"]["sample_submit_path"] = str(project_root / data_config["data"]["sample_submit_path"])

loader = DataLoader(config=data_config, add_address_columns=False)
train = loader.load_train()
test = loader.load_test()

cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv_splits = list(cv.split(train))

print("Preprocessing...")
X_train, X_test, y_train, pipeline = preprocess_for_training(train, test, cv_splits=cv_splits)

print("Saving parquet...")
X_train.write_parquet(run_dir / "X_train.parquet")
pl.DataFrame({"target": y_train.to_numpy()}).write_parquet(run_dir / "y_train.parquet")
print(f"Saved to {run_dir}")
