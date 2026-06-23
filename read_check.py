import pandas as pd

path_ = "/mnt/labworlds/Provenza/NBUPipeline/AA002/2026-05-21/rawlfp/raw/AA002_2026-05-21-2026-05-22_timedomain_LFP.parq"

df = pd.read_parquet(path_)
