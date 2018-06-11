#ifndef _PI2D_H_
#define _PI2D_H_

#include <stdio.h>
#include <iostream>
#include <string>
#include <map>
#include "LUT.h"

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
  bool SetAttrib(const std::string);
  bool SetCoord(const float*, const int, const int[2]);
  bool SetLUT(const std::string, const LUT*);
  bool DrawS(const CVType, const float*, const std::string,
             const int);
  bool DrawV(const float*, const int, const int[2], 
             const std::string, const int);
  bool Save(const int, const int, const int);
  bool ImportAttrib(const std::string);
  bool ExportAttrib(const std::string);

  int m_imageSz[2];
  int m_arraySz[2];
  float m_viewPoint[4];
  std::string m_outputPtn;
  float* m_coord;
  std::map<std::string, LUT> m_lutList;
  float m_lineWidth;
  float m_vectorMag;
  size_t m_id;

private:

};

# endif // _PI2D_H_
