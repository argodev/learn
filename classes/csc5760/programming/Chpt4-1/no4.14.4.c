#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <argp.h>
#include <inttypes.h>
#include <time.h>
#include "timer.h"

const char *argp_program_version = "no4.14.4 1.0";
const char *argp_program_bug_address = "<regillen42@students.tntech.edu>";

/* Program documentation. */
static char doc[] = "CSC5760 Prog 1 -- Problem 4.14.4";

/* The options we expect. */
static struct argp_option options[] = {
    {"iterations", 'i', "ITERATIONS", 0, "Number of times to create and delete threads"},
    {"threads", 't', "THREADS", 0, "Number of threads to utilize"}
};

// hold our command line arguments
struct arguments {
    int iterations;
    int thread_count;
};
  
// my parser function
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
    case 'i':
        arguments->iterations = arg ? atoi (arg) : 1;
        break;
    case 't':
        arguments->thread_count = arg ? atoi (arg) : 4;
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

// my argp parser
static struct argp argp = { options, parse_opt, 0, doc };

/* Global variables */
int     thread_count;


/* Parallel function */
void *Pth_mat_vect(void* rank);



/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
      struct arguments arguments;
      double start, finish;
      arguments.iterations = 0;
      arguments.thread_count = 0;
      double running_total = 0;

      argp_parse(&argp, argc, argv, 0, 0, &arguments);
      
      printf("\n");
      printf("***********************************************\n");
      printf("* Thread Creation/Removal\n");
      printf("* \n");
      printf("* Using %i threads\n", arguments.thread_count);
      printf("*\n\n");

      if (arguments.thread_count <= 0) {
            fprintf(stderr, "\n[ERROR] You must specify a thread count greater than zero\n");
            printf("\n");
            exit(1);
      }
      
      long       thread;
      pthread_t* thread_handles;

      thread_count = arguments.thread_count;
      thread_handles = malloc(thread_count*sizeof(pthread_t));
      
      printf("\nCreating Some Threads...\n");
      for (int i = 0; i < arguments.iterations; i++) {      
        GET_TIME(start);      
        for (thread = 0; thread < thread_count; thread++) {
            pthread_create(&thread_handles[thread], NULL,
                  Pth_mat_vect, (void*) thread);
        }

        for (thread = 0; thread < thread_count; thread++) {
            pthread_join(thread_handles[thread], NULL);
        }

        GET_TIME(finish);
        double total_time = finish - start;
        double average_time = total_time / thread_count;
        running_total += average_time;
      }

      double final_average = running_total / arguments.iterations;
      printf("Average Creation Time/thread given %d threads: %e seconds\n", arguments.iterations, final_average);
      free(thread_handles);
      
      return 0;
}  /* main */



/*------------------------------------------------------------------
 * Function:       Pth_mat_vect
 * Purpose:        Multiply an mxn matrix by an nx1 column vector
 * In arg:         rank
 * Global in vars: A, x, m, n, thread_count
 * Global out var: y
 */
void *Pth_mat_vect(void* rank) {
   long my_rank = (long) rank;

   return NULL;
}  /* Pth_mat_vect */

 