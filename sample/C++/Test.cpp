#include "Pi2D.h"
#include <cstdio>
#include <cstdlib>

float* READ_P(int dims[3]) {
  FILE* fp = fopen("../data/cavP.d", "rb");
  if ( ! fp ) {
    fprintf(stderr, "READ_P: can not open ../cavP.d\n");
    return NULL;
  }
  int sz;
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

  int dimSz = dims[0] * dims[1] * dims[2];
  if ( dimSz < 1 ) {
    fprintf(stderr, "READ_P: invalid dims: %d\n", dimSz);
    fclose(fp);
    return NULL;
  }
  int dimSzXY = dims[0]*dims[1];
  float* p = new float[dimSz];
  if ( ! p ) {
    fprintf(stderr, "READ_P: can not allocate memory: p[%d]\n", dimSz);
    fclose(fp);
    return NULL;
  }

  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_P: invalid size record read\n");
    fclose(fp);
    delete [] p;
    return NULL;
  }
  for ( int i = 0; i < dims[2]; i++ ) {
    if ( fread(&p[dimSzXY*i], 4, dimSzXY, fp) != dimSzXY ) {
      fprintf(stderr, "READ_P: data read failed\n");
      fclose(fp);
      delete [] p;
      return NULL;
    }
  }

  fclose(fp);
  return p;
}

float* READ_V(int dims[3]) {
  FILE* fp = fopen("../data/cavV.d", "rb");
  if ( ! fp ) {
    fprintf(stderr, "READ_V: can not open ../cavP.d\n");
    return NULL;
  }
  int sz;
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

  int dimSz = dims[0] * dims[1] * dims[2];
  if ( dimSz < 1 ) {
    fprintf(stderr, "READ_V: invalid dims: %d\n", dimSz);
    fclose(fp);
    return NULL;
  }
  int dimSzXY = dims[0]*dims[1];
  float* v = new float[dimSz * 3];
  if ( ! v ) {
    fprintf(stderr, "READ_V: can not allocate memory: v[%d]\n", dimSz*3);
    fclose(fp);
    return NULL;
  }

  if ( fread(&sz, 4, 1, fp) != 1 ) {
    fprintf(stderr, "READ_V: invalid size record read\n");
    fclose(fp);
    delete [] v;
    return NULL;
  }
  for ( int i = 0; i < dims[2]; i++ ) {
    if ( fread(&v[3*dimSzXY*i], 4, 3*dimSzXY, fp) != 3*dimSzXY ) {
      fprintf(stderr, "READ_V: data read failed\n");
      fclose(fp);
      delete [] v;
      return NULL;
    }
  }

  fclose(fp);
  return v;
}


int main(int argc, char** argv) {
  int p_dims[3], v_dims[3];
  float* p = READ_P(p_dims);
  if ( ! p ) exit(1);
  float* v = READ_V(v_dims);
  if ( ! v ) exit(1);

  Pi2D pi2d;
  char attrBuff[64];

  sprintf(attrBuff, "imageSize=500,500");
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set imageSize failed\n");
    exit(1);
  }

  sprintf(attrBuff, "arraySize=%d,%d", p_dims[0], p_dims[1]);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set P arraySize failed\n");
    exit(1);
  }

  if ( ! pi2d.DrawS(ColorContour, p, "", 20, true) ) {
    fprintf(stderr, "DrawS failed\n");
    exit(1);
  }

  sprintf(attrBuff, "arraySize=%d,%d", v_dims[0], v_dims[1]);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set V arraySize failed\n");
    exit(1);
  }

  sprintf(attrBuff, "vectorMag=5");
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "pi2d_setattrib vectorMag failed\n");
    exit(1);
  }

  int vidx[] = {0, 1};
  if ( ! pi2d.DrawV(v, 3, vidx, "", -2, false) ) {
    fprintf(stderr, "DrawV failed\n");
    exit(1);
  }

  if ( ! pi2d.Output(0, 0, 0, 0) ) {
    fprintf(stderr, "Output failed\n");
    exit(1);
  }

  delete [] p;
  delete [] v;
  exit(0);
}
