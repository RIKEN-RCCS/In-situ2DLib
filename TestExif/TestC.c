#include "pi2d_c.h"
#include "stdlib.h"

int main(int argc, char** argv) {
  float* data;
  float pi = 3.14159;
  int M = 80, N = 60;
  int i, j, iret;

  data = malloc(sizeof(float)*M*N);
  for ( j = 0; j < N; j++ )
    for ( i = 0; i < M; i++ )
      data[M*j + i] = (i * 2.0 * pi / (M-1)) * (j * 2.0 * pi / (N-1));

  iret = pi2d_init();
  iret = pi2d_setattrib("arraySize=[60,80]", 17);
  iret = pi2d_draws(0, data, "default", 7, 10, 0);
  iret = pi2d_output(0, 0, 0, 0);
  iret = pi2d_finalize();

  free(data);
  return 0;
}
