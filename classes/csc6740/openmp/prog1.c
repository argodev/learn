/*
Copyright (c) 2017 Rob Gillen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <error.h>
#include <argp.h>
#include <inttypes.h>
#include <time.h>
#include <omp.h>

const char *argp_program_version = "prog1 1.0";
const char *argp_program_bug_address = "<regillen42@students.tntech.edu>";

/* Program documentation. */
static char doc[] = "CSC6740 Assignment 1 -- Matrix Multiplication with OpenMP";

/* The options we understand. */
static struct argp_option options[] = {
    {"size", 's', "SIZE", 0, "Size of a matrix side (e.g. M x M)"},
    {"threads", 't', "THREADS", 0, "Number of threads to utilize"}
};

// hold our command line arguments
struct arguments {
    int matrix_size;
    int thread_count;
};

// my parser function
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
    case 's':
        arguments->matrix_size = arg ? atoi (arg) : 10;
        break;
    case 't':
        arguments->thread_count = arg ? atoi (arg) : 10;
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

// my argp parser
static struct argp argp = { options, parse_opt, 0, doc };

double build_good_random() {
    return ((float)rand()/(float)(RAND_MAX)) * 1024;
}

int main(int argc, char **argv) {
    struct arguments arguments;

    arguments.matrix_size = 0;
    arguments.thread_count = 0;

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    printf("\n");
    printf("***********************************************\n");
    printf("* Matrix Math with OpenMP\n");
    printf("* \n");
    printf("* Using %i threads\n", arguments.thread_count);
    printf("* Matrix size: %i x %i\n", arguments.matrix_size, arguments.matrix_size);
    printf("*\n\n");

    if ((arguments.matrix_size <= 0) || (arguments.thread_count <= 0)) {
        fprintf(stderr, "\n[ERROR] You must specify a matrix size and thread count greater than zero\n");
        exit(1);
    }

    // generate the matrices (e.g. 10x10 should be double[100], and in row-major format)
    double total_size = arguments.matrix_size*arguments.matrix_size;
    double *m1 = malloc (sizeof(double)*total_size);
    double *m2 = malloc (sizeof(double)*total_size);
    double *a1 = malloc (sizeof(double)*total_size);
    double *a2 = malloc (sizeof(double)*total_size);

    // double m1[9];
    // double m2[9];
    // double a1[9];
    // double a2[9];
    // m1[0] = 1;
    // m1[1] = 2;
    // m1[2] = 3;
    // m1[3] = 4;
    // m1[4] = 5;
    // m1[5] = 6;
    // m1[6] = 7;
    // m1[7] = 8;
    // m1[8] = 9;

    // m2[0] = 9;
    // m2[1] = 8;
    // m2[2] = 7;
    // m2[3] = 6;
    // m2[4] = 5;
    // m2[5] = 4;
    // m2[6] = 3;
    // m2[7] = 2;
    // m2[8] = 1;


    for (int i = 0; i < arguments.matrix_size*arguments.matrix_size; i++) {
        m1[i] = build_good_random();
        m2[i] = build_good_random();
    }

    // print the matrix (row major)
    // printf("MATRIX #1\n");
    // for (int i = 0; i < arguments.matrix_size; i++) {
    //     for (int j = 0; j < arguments.matrix_size; j++) {
    //         int index = (arguments.matrix_size * i) + j;
    //         printf("%f\t", m1[index]);
    //     }
    //     printf("\n");
    // }

    // printf("\nMATRIX #2\n");
    // for (int i = 0; i < arguments.matrix_size; i++) {
    //     for (int j = 0; j < arguments.matrix_size; j++) {
    //         int index = (arguments.matrix_size * i) + j;
    //         printf("%f\t", m2[index]);
    //     }
    //     printf("\n");
    // }
    int side_len = arguments.matrix_size;

    // ok... let's do it the hard way...
    // printf("Running Serial Calculations\n");
    // // double counter = 0;
    // for (int i = 0; i < side_len; i++) {
    //     for (int j = 0; j < side_len; j++) {
    //         double answer = 0;
    //         for (int k = 0; k < side_len; k++) {
    //            answer += m1[(side_len*i) + k] * m2[(side_len*k) + j];
    //         }
    //         a1[(side_len*i) + j] = answer;
    //     }
    // }

    printf("Running Parallel Calculations\n");
    // set the number of threads
    omp_set_num_threads(arguments.thread_count);
    // arguments.thread_count
    // set up the parallel branch
    #pragma omp parallel
    {
      // calculate the range (into i) for this particular thread
      int ID = omp_get_thread_num();
      int split = side_len / arguments.thread_count;
      int min = split * ID;
      int max = min + split;
    //   printf("MIN for %d: %d\n", ID, min);
    //   printf("MAX for %d: %d\n", ID, max);

      // do it just like before, but only our part (data parallel)
      for (int i = min; i < max; i++) {
        for (int j = 0; j < side_len; j++) {
            double answer = 0;
            for (int k = 0; k < side_len; k++) {
               answer += m1[(side_len*i) + k] * m2[(side_len*k) + j];
            }
            a1[(side_len*i) + j] = answer;
        }
    }
    }



    // printf("\nRESULTS\n");
    // for (int i = 0; i < arguments.matrix_size; i++) {
    //     for (int j = 0; j < arguments.matrix_size; j++) {
    //         int index = (arguments.matrix_size * i) + j;
    //         printf("%f\t", a1[index]);
    //     }
    //     printf("\n");
    // }

    // printf("\n%0.0f total operations\n", counter);
    printf("\n");
    free(m1);
    free(m2);
    free(a1);
    free(a2);
    exit (0);
}

