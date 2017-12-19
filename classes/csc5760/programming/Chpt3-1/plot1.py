#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Simple script to generate charts based on scale testing """

import json
import numpy as np
import matplotlib.pyplot as plt

MATRIX_SIZES = [1024, 4096, 8192, 16384]
THREAD_COUNTS = [4, 8, 16, 32, 64]

# def get_average_value(data, threads, size):
#     """ combine any values with the same settings into an average """
#     values = []
#     for result in data:
#         if (result['matrix_size'] == size) and (result['threads'] == threads):
#             values.append(result['duration'])
#     if len(values) > 0:
#         return sum(values)/float(len(values))
#     else:
#         return 0


# def get_size_values(data, size):
#     """ Get all of the values for a given matrix size """
#     values = []
#     for thread_count in THREAD_COUNTS:
#         values.append(get_average_value(data, thread_count, size))
#     return values

# def get_speedup(data, size):
#     speedup = []
#     values = get_size_values(data['results'], size)
#     for i in range (1, len(values)):
#         speedup.append(values[0]/values[i])
#     return speedup

# def get_efficiency(data, size):
#     efficiency = []
#     values = get_size_values(data['results'], size)
#     for i in range (1, len(values)):
#         efficiency.append((values[0]/values[i])/THREAD_COUNTS[i])
#     return efficiency

def build_machine_plots():
    """ generate the plots for a machine """
    #print 'Building plots for {0}'.format(data['name'])

    # count = 0
    # fig, axarr = plt.subplots(nrows=len(MATRIX_SIZES), ncols=1, sharex=True)

    # for size in MATRIX_SIZES:
    #     axarr[count].plot(get_size_values(data['results'], size))
    #     #print get_size_values(data['results'], size)
    #     count += 1

    # plt.suptitle('Scaling Plots for {0}'.format(data['name']))
    # fig.text(0.04, 0.5, 'Duration (seconds)', va='center', rotation='vertical')
    # fig.savefig(data['name'] + '.png')


    # calculate the speedup
    fig, ax = plt.subplots()
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*1), [3.263, 4.291, 4.754, 1.897, 0.852], width, color='.8')
    rects2 = ax.bar(ind-(width*0), [2.795, 3.990, 4.191, 3.077, 2.134], width, color='.6')
    rects3 = ax.bar(ind+(width*1), [2.723, 4.238, 4.669, 3.444, 2.435], width, color='.4')
    rects4 = ax.bar(ind+(width*2), [2.735, 4.113, 4.575, 4.447, 2.975], width, color='.2')

    ax.set_ylabel('Speedup')
    ax.set_title('Speedup by Processor Count')
    ax.set_xlabel('Processors Used')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('4', '8', '16', '32', '64'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
              ('1024', '4096', '8192', '16384'), loc='upper right')
    fig.savefig('speedup.png')
    #plt.show()

    # calculate the efficiency
    fig, ax = plt.subplots()
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*1), [81.58, 53.64, 29.71, 5.93, 1.33], width, color='.8')
    rects2 = ax.bar(ind-(width*0), [69.86, 49.88, 26.19, 9.62, 3.33], width, color='.6')
    rects3 = ax.bar(ind+(width*1), [68.08, 52.98, 29.18, 10.76, 3.81], width, color='.4')
    rects4 = ax.bar(ind+(width*2), [68.38, 51.41, 28.59, 13.90, 4.65], width, color='.2')

    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency by Processor Count')
    ax.set_xlabel('Processors Used')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('4', '8', '16', '32', '64'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
              ('1024', '4096', '8192', '16384'), loc='upper right')
    fig.savefig('efficiency.png')




def main():
    """ main entry point for the program """
    build_machine_plots()


if __name__ == '__main__':
    main()