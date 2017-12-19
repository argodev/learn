#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <argp.h>
#include <inttypes.h>
#include <time.h>
#include "timer.h"

const char *argp_program_version = "no4.13.14 1.0";
const char *argp_program_bug_address = "<regillen42@students.tntech.edu>";

/* Program documentation. */
static char doc[] = "CSC5760 Prog 1 -- Problem 4.13.14";

/* The options we expect. */
static struct argp_option options[] = {
      {"rows", 'm', "ROWS", 0, "Number of rows in the matrix (e.g. M x n)"},
      {"cols", 'n', "COLS", 0, "Number of columns in the matrix (e.g. m x N)"},
      {"threads", 't', "THREADS", 0, "Number of threads to utilize"}
};

// hold our command line arguments
struct arguments {
    int num_rows;
    int num_cols;
    int thread_count;
};
  
// my parser function
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
    case 'm':
        arguments->num_rows = arg ? atoi (arg) : 10;
        break;
    case 'n':
        arguments->num_cols = arg ? atoi (arg) : 10;
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
int     m, n;
double* A;
double** B;
double* x;
double* y;
double* shared_times;
double* indiv_times;

/* Serial functions */
void Usage(char* prog_name);
void Read_matrix(char* prompt, double A[], int m, int n);
void Read_vector(char* prompt, double x[], int n);
void Print_matrix(char* title, double A[], int m, int n);
void Print_vector(char* title, double y[], double m);
double build_random();

/* Parallel function */
void *Pth_mat_vect(void* rank);
void *Pth_mat_vect_2d(void* rank);
// void *Pth_mat_vect_priv(void* rank);



/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
      struct arguments arguments;
      double start, finish;
      
      arguments.num_rows = 0;
      arguments.num_cols = 0;
      arguments.thread_count = 0;
      
      argp_parse(&argp, argc, argv, 0, 0, &arguments);
      
      printf("\n");
      printf("***********************************************\n");
      printf("* Matrix-Vector with PThreads\n");
      printf("* \n");
      printf("* Using %i threads\n", arguments.thread_count);
      printf("* Matrix size: %i x %i\n", arguments.num_rows, arguments.num_cols);
      printf("*\n\n");

      if ((arguments.num_rows <= 0) || (arguments.thread_count <= 0)) {
            fprintf(stderr, "\n[ERROR] You must specify a matrix size and thread count greater than zero\n");
            printf("\n");
            exit(1);
      }
      
      m = arguments.num_rows;
      n = arguments.num_cols;

      // generate the matrix in row-major format and the vector
      double total_size = m*n;
      A = malloc(sizeof(double)*total_size);
      x = malloc(sizeof(double)*n);

      // set up the two-dimensional array
      const size_t row_pointers_bytes = m * sizeof *B;
      const size_t row_elements_bytes = n * sizeof **B;
      B = malloc(row_pointers_bytes + n * row_elements_bytes);

      // now we need to initialize it so that the row's pointer points to the right place
      size_t i;
      double* const data = B + m;
      for(i = 0; i < m; i++)
        B[i] = data + i * n;


      // let's fill our matrix
      int two_d_row = 0;
      int two_d_col = 0;
      for (int i = 0; i < total_size; i++) {
        double random_value = build_random();
        A[i] = random_value;
        B[two_d_row][two_d_col] = random_value;

        // increment our odd 2d stuff
        two_d_col++;

        if (two_d_col >= n) {
            two_d_row++;
            two_d_col = two_d_col - n;
        }
      }

      // let's fill our vector
      for (int i = 0; i < arguments.num_cols; i++) {
            x[i] = build_random();
      }


      long       thread;
      pthread_t* thread_handles;

      thread_count = arguments.thread_count;
      thread_handles = malloc(thread_count*sizeof(pthread_t));
      shared_times = malloc(sizeof(double)*thread_count);
      indiv_times = malloc(sizeof(double)*thread_count);

      y = malloc(m*sizeof(double));
   
      if ((m < 50) && (n < 50)) {
            Print_matrix("Generated Matrix:", A, m, n);
            Print_vector("\nGenerated Vector:", x, n);
      }
      printf("\nRunning Parallel Calculations\n");
      
      GET_TIME(start);      
      for (thread = 0; thread < thread_count; thread++) {
            pthread_create(&thread_handles[thread], NULL,
                  Pth_mat_vect, (void*) thread);
      }

      for (thread = 0; thread < thread_count; thread++) {
            pthread_join(thread_handles[thread], NULL);
      }

      GET_TIME(finish);
      printf("Elapsed time = %e seconds\n", finish - start);

      // loop through and calculate time
      double shared_time = 0;
      for (int i = 0; i < thread_count; i++) {
            shared_time += shared_times[i];
      }
      printf("Shared Calcuation Time = %e seconds\n", shared_time);
      

      if (m < 50) {
            Print_vector("\nThe product is", y, m);
            printf("\n");
      }

    printf("\nRunning 2D Parallel Calculations\n");
    GET_TIME(start);      
      
    // handle the independent version
    for (thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL,
              Pth_mat_vect_2d, (void*) thread);
    }

      for (thread = 0; thread < thread_count; thread++) {
            pthread_join(thread_handles[thread], NULL);
      }
      GET_TIME(finish);
      printf("Elapsed time = %e seconds\n", finish - start);

      // loop through and calculate time
      double indiv_time = 0;
      for (int i = 0; i < thread_count; i++) {
            indiv_time += indiv_times[i];
      }
    printf("2D Calcuation Time = %e seconds\n", indiv_time);



      free(A);
      free(B);
      free(x);
      free(y);
      free(thread_handles);
      free(shared_times);
      free(indiv_times);
      
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
   int i, j;
   int local_m = m/thread_count; 
   int my_first_row = my_rank*local_m;
   int my_last_row = (my_rank+1)*local_m - 1;
   double start, finish;

   GET_TIME(start);      
   
   for (i = my_first_row; i <= my_last_row; i++) {
      y[i] = 0.0;
      for (j = 0; j < n; j++)
          y[i] += A[i*n+j]*x[j];
   }

   GET_TIME(finish);
   shared_times[my_rank] = finish - start;

   return NULL;
}  /* Pth_mat_vect */

void *Pth_mat_vect_2d(void* rank) {
    long my_rank = (long) rank;
    int i, j;
    int local_m = m/thread_count; 
    int my_first_row = my_rank*local_m;
    int my_last_row = (my_rank+1)*local_m - 1;
    double start, finish;
 
    GET_TIME(start);      
    
    for (i = my_first_row; i <= my_last_row; i++) {
       y[i] = 0.0;
       for (j = 0; j < n; j++)
           y[i] += B[i][j]*x[j];
    }
 
    GET_TIME(finish);
    indiv_times[my_rank] = finish - start;
 
    return NULL;
 }  /* Pth_mat_vect */
 

/*------------------------------------------------------------------
 * Function:    Print_matrix
 * Purpose:     Print the matrix
 * In args:     title, A, m, n
 */
void Print_matrix( char* title, double A[], int m, int n) {
   int   i, j;

   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%4.1f ", A[i*n + j]);
      printf("\n");
   }
}  /* Print_matrix */


/*------------------------------------------------------------------
 * Function:    Print_vector
 * Purpose:     Print a vector
 * In args:     title, y, m
 */
void Print_vector(char* title, double y[], double m) {
   int   i;

   printf("%s\n", title);
   for (i = 0; i < m; i++)
      printf("%4.1f ", y[i]);
   printf("\n");
}  /* Print_vector */



double build_random() {
      return ((float)rand()/(float)(RAND_MAX)) * 1024;
  }