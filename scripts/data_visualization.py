import pandas as pd

import pandas as pd

sensor = pd.read_csv(
    './Set1/sensordata/File_name_2022-09-09T13_42_22.323924.csv',
    header=None
)

sensor.columns = [
    "Accelerometer",
    "Acoustic",
    "Force_X",
    "Force_Y",
    "Force_Z",
    "Timestamp"
]

print(sensor.head())
print(sensor.shape)

# print(sensor.describe())
# print("\nMin values:\n", sensor.min())
# print("\nMax values:\n", sensor.max())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(sensor["Accelerometer"])
plt.title("Accelerometer Signal")
plt.show()