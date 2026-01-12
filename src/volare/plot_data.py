import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.style.use('science')

# data = np.genfromtxt('questdb-audchf.csv',delimiter=',')
# print(data)
# sys.exit()

df = pd.read_csv('../candle_data/questdb-audchf.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.set_index("timestamp").sort_index()
df.index.to_series().diff().value_counts()

ts = df.index[:2000]
rt = np.log(df['close'] / df['open'])[:2000]

# N = int(sys.argv[1])
# vt = np.sqrt(1/N * sum(rt[**2)

# plt.figure()
# plt.plot(ts,rt)
# plt.show()

# print(df)
sys.exit()

tstamp = df['timestamp'][:200]
open_value = df['open'][:200]
close_value = df['close'][:200]
plt.figure()
plt.plot(tstamp,open_value-close_value)
# plt.plot(tstamp,close_value)
plt.show()

sys.exit()

tstamp = data[:,0]
symbol = data[:,1]
open_value = data[:,2]
high = data[:,3]
low = data[:,4]
close_value = data[:,5]

plt.figure()
plt.plot(tstamp,open_value)
plt.plot(tstamp,close_value)
plt.show()
