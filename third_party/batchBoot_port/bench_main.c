/* Runs batch_bootstrap_process of BatchBoot (full case, test/batchBoot_main.c) for the parameter sets given on the
   command line: 2, 4, 6, 8 (Boot2 ... Boot8). Each prints the time of one batch (clock(), one thread) and the
   amortized time, then checks the decryption of every coefficient. */
#include "batchBoot.h"
#include <stdlib.h>
int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    int c = atoi(argv[i]);
    printf("== Boot%d\n", c); fflush(stdout);
    batch_bootstrap_process(c); fflush(stdout);
  }
  return 0;
}
