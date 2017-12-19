#! /usr/bin/python

import csv
import matplotlib.pyplot as plt

with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    hist_data = list(reader)

hist_data = [int(i) for i in hist_data[0]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(hist_data)
ax.set_ylabel('Frequency')
ax.set_xlabel('Histogram bin (pixel value)')
ax.set_xlim([0,256])
plt.show()
