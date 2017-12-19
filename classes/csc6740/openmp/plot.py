#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Simple script to generate charts based on scale testing """

import json
import numpy as np
import matplotlib.pyplot as plt

MATRIX_SIZES = [1000, 2000, 3000, 4000, 5000, 10000]
THREAD_COUNTS = [1, 2, 4, 8, 16, 24]

def get_average_value(data, threads, size):
    """ combine any values with the same settings into an average """
    values = []
    for result in data:
        if (result['matrix_size'] == size) and (result['threads'] == threads):
            values.append(result['duration'])
    if len(values) > 0:
        return sum(values)/float(len(values))
    else:
        return 0


def get_size_values(data, size):
    """ Get all of the values for a given matrix size """
    values = []
    for thread_count in THREAD_COUNTS:
        values.append(get_average_value(data, thread_count, size))
    return values

def get_speedup(data, size):
    speedup = []
    values = get_size_values(data['results'], size)
    for i in range (1, len(values)):
        speedup.append(values[0]/values[i])
    return speedup

def get_efficiency(data, size):
    efficiency = []
    values = get_size_values(data['results'], size)
    for i in range (1, len(values)):
        efficiency.append((values[0]/values[i])/THREAD_COUNTS[i])
    return efficiency

def build_machine_plots(data):
    """ generate the plots for a machine """
    print 'Building plots for {0}'.format(data['name'])

    count = 0
    fig, axarr = plt.subplots(nrows=len(MATRIX_SIZES), ncols=1, sharex=True)

    for size in MATRIX_SIZES:
        axarr[count].plot(get_size_values(data['results'], size))
        #print get_size_values(data['results'], size)
        count += 1

    plt.suptitle('Scaling Plots for {0}'.format(data['name']))
    fig.text(0.04, 0.5, 'Duration (seconds)', va='center', rotation='vertical')
    fig.savefig(data['name'] + '.png')



    # calculate the speedup
    fig, ax = plt.subplots()
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*2.5), get_speedup(data, 1000), width, color='1')
    rects2 = ax.bar(ind-(width*1.5), get_speedup(data, 2000), width, color='.8')
    rects3 = ax.bar(ind-(width*0.5), get_speedup(data, 3000), width, color='.6')
    rects4 = ax.bar(ind+(width*0.5), get_speedup(data, 4000), width, color='.4')
    rects5 = ax.bar(ind+(width*1.5), get_speedup(data, 5000), width, color='.2')
    rects6 = ax.bar(ind+(width*2.5), get_speedup(data, 10000), width, color='.0')

    ax.set_ylabel('Speedup')
    ax.set_title('Speedup by Thread Count ({0})'.format(data['name']))
    ax.set_xlabel('Threads Used')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('2', '4', '8', '16', '24'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),
              ('1,000', '2,000', '3,000', '4,000', '5,000', '10,000'), loc='upper left')
    fig.savefig(data['name'] + '_speedup.png')
    #plt.show()


    # calculate the efficiency
    fig, ax = plt.subplots()
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.14       # the width of the bars
    rects1 = ax.bar(ind-(width*2.5), get_efficiency(data, 1000), width, color='1')
    rects2 = ax.bar(ind-(width*1.5), get_efficiency(data, 2000), width, color='.8')
    rects3 = ax.bar(ind-(width*0.5), get_efficiency(data, 3000), width, color='.6')
    rects4 = ax.bar(ind+(width*0.5), get_efficiency(data, 4000), width, color='.4')
    rects5 = ax.bar(ind+(width*1.5), get_efficiency(data, 5000), width, color='.2')
    rects6 = ax.bar(ind+(width*2.5), get_efficiency(data, 10000), width, color='.0')

    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency by Thread Count ({0})'.format(data['name']))
    ax.set_xlabel('Threads Used')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('2', '4', '8', '16', '24'))
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),
              ('1,000', '2,000', '3,000', '4,000', '5,000', '10,000'), loc='upper right')
    fig.savefig(data['name'] + '_efficiency.png')




def main():
    """ main entry point for the program """
    results = None

    with open('prog1_results.json') as json_data:
        results = json.load(json_data)

    if results is not None:
        if 'sources' in results:
            for source in results['sources']:
                build_machine_plots(source)


if __name__ == '__main__':
    main()