#include <stdio.h>
#include <inttypes.h>
#include <sodium.h>


const unsigned int M_ROWS = 10;
const unsigned int N_COLS_ROWS = 15;
const unsigned int P_COLS = 12;


int foo(uint rows, uint cols) {

  if (sodium_init() == -1) {
    return 1;
  }

  // add some of a given length
  int max_vals;
  max_vals = rows * cols;

  uint32_t valueArray[max_vals];

  for (int i = 0; i < max_vals; i++) {
    valueArray[i] = randombytes_uniform(100);
  }

  printf("%ld\n", sizeof(valueArray));

  for (int j = 0; j < max_vals; j++) {
    printf("%u ", valueArray[j]);
  }

  printf("\n");

}

int main() {
  foo(M_ROWS, N_COLS_ROWS);
  foo(N_COLS_ROWS, P_COLS);
}
