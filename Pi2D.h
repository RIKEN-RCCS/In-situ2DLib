#ifndef _PI2D_H_
#define _PI2D_H_

#include <stdio.h>
#include <iostream>
#include <string>
#include <map>
#include "LUT.h"

#ifdef _REAL_DBL
  typedef double Real;
#else
  typedef float Real;
#endif

enum CVType {
  ColorContour = 0,
  ContourLine = 1,
};

class Pi2D
{
public:
  static size_t s_id;

  Pi2D();
  ~Pi2D();
  bool SetAttribute(const std::string);
  bool SetCoord(const Real*, const int, const int[2]);
  bool SetLUT(const std::string, const LUT*);
  bool DrawS(const CVType, const Real*, const std::string,
             const int);
  bool DrawV(const Real*, const int, const int[2], 
             const std::string, const int);
  bool Save(const int, const int, const int);
  bool ImportAttrib(const std::string);
  bool ExportAttrib(const std::string);

  int m_imageSz[2];
  int m_arraySz[2];
  Real m_viewPoint[4];
  std::string m_outputPtn;
  Real* m_coord;
  std::map<std::string, LUT> m_lutList;
  Real m_lineWidth;
  Real m_vectorMag;
  size_t m_id;

private:
  int m_arrSz[2];

};

# endif // _PI2D_H_
