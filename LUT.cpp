#include "LUT.h"

using namespace std;

LUT::LUT() : cbHoriz(false), cbNumTic(2)
{
  color_s clr(1.0, 1.0, 1.0);
  colorList[0.0] = clr;
  cbSize[0] = 0.05; cbSize[1] = 0.5;
  cbPos[0] = 0.0; cbPos[1] = 0.0;
  printf("LUT: setring\n");
}

LUT::~LUT()
{
}

color_s LUT::ColorByValue(const float val)
{
  color_s clr;

  map<float, color_s>::iterator itr;
  itr = colorList.begin();
  if ( val < (*itr).first )
    return (*itr).second;
  for ( ; itr != colorList.end(); itr++ ) {
    if ( val <= (*itr).first )
    //if ( val == (*itr).first )
      return (*itr).second;
    clr = (*itr).second;
  }

  return clr;
}
