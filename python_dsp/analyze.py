"""Usage: python analyze.py log.csv
Log the ESP32 serial output to log.csv first (Arduino IDE serial monitor copy, PuTTY log, or:
  python -m serial.tools.miniterm COM3 230400 > log.csv ).
Produces: time-domain plot, PSD comparison, tremor-band attenuation (dB), loop timing.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

FS = 100.0
df = pd.read_csv(sys.argv[1], on_bad_lines="skip").dropna()
df = df.apply(pd.to_numeric, errors="coerce").dropna()
t = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0

raw = df["ax"].values - df["ax"].mean()
filt = df["fx"].values - df["fx"].mean()

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(t, raw, label="raw", alpha=0.7)
ax[0].plot(t, filt, label="filtered")
ax[0].set_title("Time domain (ax)"); ax[0].set_xlabel("s"); ax[0].set_ylabel("g"); ax[0].legend()

fr, pr = welch(raw, FS, nperseg=256)
ff, pf = welch(filt, FS, nperseg=256)
ax[1].semilogy(fr, pr, label="raw"); ax[1].semilogy(ff, pf, label="filtered")
ax[1].axvspan(3, 12, alpha=0.1, color="red", label="tremor band")
ax[1].set_title("Welch PSD"); ax[1].set_xlabel("Hz"); ax[1].legend()

ax[2].plot(t, df["f0_hz"]); ax[2].set_title("Tracked tremor frequency f0")
ax[2].set_xlabel("s"); ax[2].set_ylabel("Hz")
plt.tight_layout(); plt.savefig("results.png", dpi=150)

band = (fr >= 3) & (fr <= 12)
att = 10 * np.log10(pr[band].sum() / pf[band].sum())
print(f"Tremor-band (3-12 Hz) power reduction: {att:.2f} dB")
print(f"Mean tracked f0 (when active): {df.loc[df['f0_hz'] > 0, 'f0_hz'].mean():.2f} Hz")
print(f"Loop time: mean {df['loop_us'].mean():.0f} us, max {df['loop_us'].max():.0f} us "
      f"(budget {1e6 / FS:.0f} us)")