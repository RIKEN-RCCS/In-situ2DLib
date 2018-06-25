#ifndef _PI2D_DEFS_H_
#define _PI2D_DEFS_H_

#ifdef _REAL_DBL
typedef double Real;
#else
typedef float Real;
#endif

enum CVType {
  ColorContour = 0,
  ContourLine = 1,
};

#endif // _PI2D_DEFS_H_
