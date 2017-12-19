#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Simple script to generate charts based on scale testing """

import json
import numpy as np
import matplotlib.pyplot as plt

MATRIX_SIZES = [1000, 2000, 3000, 4000, 5000, 10000]
THREAD_COUNTS = [1, 2, 4, 8, 16, 24]

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
    N = 7
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*2.5), [1.96, 3.70, 6.45, 9.76, 12.31, 13.22, 12.85], width, color='1')
    rects2 = ax.bar(ind-(width*1.5), [1.99, 3.92, 7.55, 13.79, 22.86, 32.65, 39.51], width, color='.8')
    rects3 = ax.bar(ind-(width*0.5), [1.99, 3.98, 7.88, 15.38, 29.09, 51.61, 82.05], width, color='.6')
    rects4 = ax.bar(ind+(width*0.5), [1.99, 3.99, 7.97, 15.84, 31.22, 60.38, 112.28], width, color='.4')
    rects5 = ax.bar(ind+(width*1.5), [1.99, 3.99, 7.99, 15.96, 31.80, 63.05, 123.67], width, color='.2')
    rects6 = ax.bar(ind+(width*2.5), [1.99, 3.99, 7.99, 15.99, 31.95, 63.76, 126.89], width, color='.0')

    ax.set_ylabel('Speedup')
    ax.set_title('Speedup by Processor Count')
    ax.set_xlabel('Processors Used')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('2', '4', '8', '16', '32', '64', '128'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),
              ('10', '20', '40', '80', '160', '320'), loc='upper left')
    fig.savefig('speedup.png')
    #plt.show()

    # calculate the efficiency
    fig, ax = plt.subplots()
    N = 7
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*2.5), [98.04, 92.59, 80.65, 60.98, 38.46, 20.66, 10.04], width, color='1')
    rects2 = ax.bar(ind-(width*1.5), [99.50, 98.04, 94.34, 86.21, 71.43, 51.02, 30.86], width, color='.8')
    rects3 = ax.bar(ind-(width*0.5), [99.88, 99.50, 98.52, 96.15, 90.91, 80.65, 64.10], width, color='.6')
    rects4 = ax.bar(ind+(width*0.5), [99.97, 99.88, 99.63, 99.01, 97.56, 94.34, 87.72], width, color='.4')
    rects5 = ax.bar(ind+(width*1.5), [99.99, 99.97, 99.91, 99.75, 99.38, 98.52, 96.62], width, color='.2')
    rects6 = ax.bar(ind+(width*2.5), [100.00, 99.99, 99.98, 99.94, 99.84, 99.63, 99.13], width, color='.0')

    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency by Processor Count')
    ax.set_xlabel('Processors Used')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('2', '4', '8', '16', '32', '64', '128'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),
              ('10', '20', '40', '80', '160', '320'), loc='lower left')
    fig.savefig('efficiency.png')




def main():
    """ main entry point for the program """
    build_machine_plots()


if __name__ == '__main__':
    main()