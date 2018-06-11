#ifndef _LUT_H_
#define _LUT_H_

#include <stdio.h>
#include <iostream>
//#include <string>
#include <map>

typedef struct Color {
  float red;
  float green;
  float blue;
} color_s;

class LUT
{
public:
  LUT();
  ~LUT();

  color_s ColorByValue(const float);

  std::map<float, color_s> m_colorList;
};

# endif // _LUT_H_

