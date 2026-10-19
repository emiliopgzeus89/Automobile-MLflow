from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fuentes"))
from datos_prep import load_uci_automobile, clean_automobile

raw_path = ROOT / "datos" / "datos_ini" / "automobile_original.csv"
clean_path = ROOT / "datos" / "datos_limp" / "automobile_limpio.csv"
raw_path.parent.mkdir(parents=True, exist_ok=True)
clean_path.parent.mkdir(parents=True, exist_ok=True)

if not raw_path.exists():
    df_raw = load_uci_automobile()
    df_raw.to_csv(raw_path, index=False)
else:
    df_raw = pd.read_csv(raw_path, na_values=["?"])

df_clean = clean_automobile(df_raw)
df_clean.to_csv(clean_path, index=False)
print(f"Original: {df_raw.shape} -> {raw_path}")
print(f"Limpio:   {df_clean.shape} -> {clean_path}")
