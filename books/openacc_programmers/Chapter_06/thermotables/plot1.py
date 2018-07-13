#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Simple script to generate charts based on scale testing """

import json
import numpy as np
import matplotlib.pyplot as plt

MATRIX_SIZES = [1024, 4096, 8192, 16384]
THREAD_COUNTS = [4, 8, 16, 32, 64]

def build_machine_plots():
    """ generate the plots for a machine """
    

    num_points   = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    cpu_1core    = [0.001, 0.001, 0.08, 0.87, 8.76, 87.64, 881.31]
    cpu_8core    = [0.01,   0.01, 0.02, 0.17, 1.72, 16.68, 172.65]
    cpu_gpu      = [0.90,   0.90, 0.88, 0.93, 1.38, 05.31, 44.36]
    cpu_gpu_data = [0.21,   0.21, 0.21, 0.23, 0.51, 02.89, 26.35]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e1, 1e7)
    ax.set_ylim(1e-03, 1e3)
    plt.grid(True,which="both",ls="-")
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Execution Time(s)')

    data01 = ax.plot(num_points, cpu_1core, "ko--")
    data02 = ax.plot(num_points, cpu_8core, "kv-.")
    data03 = ax.plot(num_points, cpu_gpu, "ks:")
    data04 = ax.plot(num_points, cpu_gpu_data, "kd-")

    ax.legend((data01[0], data02[0], data03[0], data04[0]),
              ('CPU 1 Core', 'CPU 8 Cores', 'CPU + GPU', 'CPU + GPU + Data Locality'), loc='upper left')

    fig.savefig('scale.png')


def main():
    """ main entry point for the program """
    build_machine_plots()


if __name__ == '__main__':
    main()