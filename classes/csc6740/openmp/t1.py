import numpy as np
import matplotlib.pyplot as plt

l1 = [1.4978969505783388, 2.7306709265175724, 5.17686250757117, 5.947807933194155, 6.519450800915332]
l2 = [1.5494847133842224, 3.035716137058732, 5.97937512762916, 7.912714498040805, 8.660455486542443]
l3 = [1.6337678744713184, 3.011627122076345, 5.817106163825418, 8.056474730830871, 8.69257933318721]
l4 = [1.5173095839988155, 2.755319761938081, 5.597588507784671, 7.674225055025678, 8.7122542469937]
l5 = [1.5586483846689982, 2.909443842810307, 5.78035143769968, 7.769297199652977, 8.858257189443588]

fig, ax = plt.subplots()
N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars
rects1 = ax.bar(ind - (width*2), l1, width, color='.8')
rects2 = ax.bar(ind-width, l2, width, color='.6')
rects3 = ax.bar(ind, l3, width, color='.4')
rects4 = ax.bar(ind+(width*1), l4, width, color='.2')
rects5 = ax.bar(ind+(width*2), l5, width, color='.0')

ax.set_ylabel('Speedup')
ax.set_title('Speedup by Core Count (wilma)')
ax.set_xlabel('Threads Used')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('2', '4', '8', '16', '24'))
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
          ('1,000', '2,000', '3,000', '4,000', '5,000', '10,000'), loc='upper left')
plt.show()
