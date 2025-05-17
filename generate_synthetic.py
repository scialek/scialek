"""Generate 365‑day synthetic Lifestyle DAG dataset"""
import numpy as np, pandas as pd, math, random

N = 365
rng = np.random.default_rng(42)

rain = rng.binomial(1, 0.30, size=N)
z = np.zeros(N)
tfs = np.zeros(N)
sleep = np.zeros(N)
a_today = np.zeros(N)
a_tom = np.zeros(N)
tfs[0] = 40

for t in range(N):
    z[t] = max(0, rng.normal(55 - 25*rain[t], 12))
    if t:
        tfs[t] = 0.85*tfs[t-1] + 0.20*z[t] + 6*rain[t] + rng.normal(0,2)
    sleep[t] = np.clip(82 + 0.12*z[t] - 0.40*tfs[t] + rng.normal(0,1.5), 50, 100)
    a_today[t] = np.clip(4 + 0.03*z[t] - 1.2*rain[t] + rng.normal(0,0.6), 1, 7)
    if t:
        a_tom[t] = a_today[t-1]

df = pd.DataFrame({
    "precipitationFlag": rain,
    "zone2Minutes": np.round(z).astype(int),
    "trainingFatigueScore": np.round(tfs).astype(int),
    "sleepQuality": np.round(sleep).astype(int),
    "affectToday": np.round(a_today).astype(int),
    "affectTomorrow": np.round(a_tom).astype(int),
})
df.to_csv("LifestyleDAG.csv", index=False)
print("Saved LifestyleDAG.csv with", len(df), "rows")
