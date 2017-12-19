#! /usr/bin/python

import csv
import matplotlib.pyplot as plt
import numpy as np

with open('ref.csv', 'r') as f:
    reader = csv.reader(f)
    hist_data = list(reader)

hist_data = np.array([int(i) for i in hist_data[0]])

with open('output.csv', 'r') as f:
    reader = csv.reader(f)
    output_data = list(reader)

output_data = np.array([int(i) for i in output_data[0]])

x_range = np.array(range(0, 256))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_range, hist_data, x_range, output_data)
ax.set_ylabel('Frequency')
ax.set_xlabel('Histogram bin (pixel value)')
ax.set_xlim([0,256])
plt.show()
