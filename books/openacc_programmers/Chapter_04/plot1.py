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
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*0), [0.472487104406797, 4.57151837019741, 2.72882002843844], width, color='.8')
    rects2 = ax.bar(ind+(width*1), [0.642112345634295, 11.2960920557659, 4.0734294422213], width, color='.6')

    threshold = 1.0
    ax.plot([0, 2.5], [threshold, threshold], "k--")
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup Compared To Serial')
    ax.set_xlabel('Approach')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Naive', 'Optimized', 'OpenMP'))
    ax.legend((rects1[0], rects2[0]),
              ('Lap1', 'Lap2'), loc='upper right')
    fig.savefig('speedup.png')
    #plt.show()

    # calculate the efficiency
    fig, ax = plt.subplots()
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*0), [11.9111286, 25.2094258, 2.6055082, 4.3649374], width, color='.8')
    rects2 = ax.bar(ind+(width*1), [14.9974238, 23.356386, 1.3276648, 3.6817684], width, color='.6')

    ax.set_ylabel('Time Elapsed')
    ax.set_title('Time Elapsed by Approach')
    ax.set_xlabel('Approach')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Serial', 'Naive', 'Optimized', 'OpenMP'))
    ax.legend((rects1[0], rects2[0]),
              ('Lap1', 'Lap2'), loc='upper right')
    fig.savefig('times.png')




def main():
    """ main entry point for the program """
    build_machine_plots()


if __name__ == '__main__':
    main()