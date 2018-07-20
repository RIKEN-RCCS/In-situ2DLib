#include "pi2d_c.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

float* READ_P(int dims[3]) {
  FILE* fp;
  int sz;
  int dimSz, dimSzXY;
  float* p;
  
  fp = fopen("../data/cavP.d", "rb");
  if ( ! fp ) {
    fprintf(stderr, "READ_P: can not open ../cavP.d\n");
    return NULL;
  }

  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_P: invalid size record read\n");
    fclose(fp);
    return NULL;
  }
  if ( fread(dims, 4, 3, fp) != 3 ) {
    fprintf(stderr, "READ_P: invalid dims record read\n");
    fclose(fp);
    return NULL;
  }
  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_P: invalid size record read\n");
    fclose(fp);
    return NULL;
  }

  dimSz = dims[0] * dims[1] * dims[2];
  if ( dimSz < 1 ) {
    fprintf(stderr, "READ_P: invalid dims: %d\n", dimSz);
    fclose(fp);
    return NULL;
  }
  dimSzXY = dims[0]*dims[1];
  p = (float*)malloc(dimSz*sizeof(float));
  if ( ! p ) {
    fprintf(stderr, "READ_P: can not allocate memory: p[%d]\n", dimSz);
    fclose(fp);
    return NULL;
  }

  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_P: invalid size record read\n");
    fclose(fp);
    free(p);
    return NULL;
  }
  for ( int i = 0; i < dims[2]; i++ ) {
    if ( fread(&p[dimSzXY*i], 4, dimSzXY, fp) != dimSzXY ) {
      fprintf(stderr, "READ_P: data read failed\n");
      fclose(fp);
      free(p);
      return NULL;
    }
  }

  fclose(fp);
  return p;
}

float* READ_V(int dims[3]) {
  FILE* fp;
  int sz;
  int dimSz, dimSzXY;
  float* v;

  fp = fopen("../data/cavV.d", "rb");
  if ( ! fp ) {
    fprintf(stderr, "READ_V: can not open ../cavP.d\n");
    return NULL;
  }

  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_V: invalid size record read\n");
    fclose(fp);
    return NULL;
  }
  if ( fread(dims, 4, 3, fp) != 3 ) {
    fprintf(stderr, "READ_V: invalid dims record read\n");
    fclose(fp);
    return NULL;
  }
  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_V: invalid size record read\n");
    fclose(fp);
    return NULL;
  }

  dimSz = dims[0] * dims[1] * dims[2];
  if ( dimSz < 1 ) {
    fprintf(stderr, "READ_V: invalid dims: %d\n", dimSz);
    fclose(fp);
    return NULL;
  }
  dimSzXY = dims[0]*dims[1];
  v = (float*)malloc(dimSz * 3 * sizeof(float));
  if ( ! v ) {
    fprintf(stderr, "READ_V: can not allocate memory: v[%d]\n", dimSz*3);
    fclose(fp);
    return NULL;
  }

  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_V: invalid size record read\n");
    fclose(fp);
    free(v);
    return NULL;
  }
  for ( int i = 0; i < dims[2]; i++ ) {
    if ( fread(&v[3*dimSzXY*i], 4, 3*dimSzXY, fp) != 3*dimSzXY ) {
      fprintf(stderr, "READ_V: data read failed\n");
      fclose(fp);
      free(v);
      return NULL;
    }
  }

  fclose(fp);
  return v;
}

  
int main(int argc, char** argv) {
  int p_dims[3], v_dims[3];
  float *p, *v;
  char attrbuf[64];
  int vidx[] = {0, 1};
  int iret;

  p = READ_P(p_dims);
  if ( ! p ) exit(1);
  v = READ_V(v_dims);
  if ( ! v ) exit(1);

  iret = pi2d_init();
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_init failed\n");
    exit(1);
  }
  
  sprintf(attrbuf, "imageSize=500,500");
  iret = pi2d_setattrib(attrbuf, strlen(attrbuf));
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_setattrib imageSize failed\n");
    exit(1);
  }

  sprintf(attrbuf, "arraySize=%d,%d", p_dims[0], p_dims[1]);
  iret = pi2d_setattrib(attrbuf, strlen(attrbuf));
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_setattrib arraySize failed\n");
    exit(1);
  }

  iret = pi2d_draws(0, p, "", 0, 20, 1);
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_draws failed\n");
    exit(1);
  }

  sprintf(attrbuf, "arraySize=%d,%d", v_dims[0], v_dims[1]);
  iret = pi2d_setattrib(attrbuf, strlen(attrbuf));
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_setattrib arraySize failed\n");
    exit(1);
  }

  sprintf(attrbuf, "vectorMag=5");
  iret = pi2d_setattrib(attrbuf, strlen(attrbuf));
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_setattrib vectorMag failed\n");
    exit(1);
  }
  
  iret = pi2d_drawv(v, 3, vidx, "", 0, -2, 0);
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_drawv failed\n");
    exit(1);
  }

  iret = pi2d_output(0, 0, 0, 0);
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_output failed\n");
    exit(1);
  }

  iret = pi2d_finalize();
  if ( iret != 0 ) {
    fprintf(stderr, "pi2d_finalize failed\n");
    exit(1);
  }

  free(p);
  free(v);
  exit(0);
}
