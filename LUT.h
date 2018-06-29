#ifndef _LUT_H_
#define _LUT_H_

#include <stdio.h>
#include <iostream>
#include <map>

#include "Pi2Ddefs.h"


struct Color {
  float red;
  float green;
  float blue;
  Color(const float r=1.f, const float g=1.f, const float b=1.f)
    : red(r), green(g), blue(b) {}
};


class LUT {
public:
  LUT();
  ~LUT();

  Color ColorByValue(const float);

  std::map<float, Color> colorList;
  Real cbSize[2];
  Real cbPos[2];
  bool cbHoriz;
  size_t cbNumTic;
};

# endif // _LUT_H_

