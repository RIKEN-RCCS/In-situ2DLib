#include "LUT.h"

using namespace std;

LUT::LUT() : cbHoriz(false), cbNumTic(2)
{
  Color clr(1.0, 1.0, 1.0);
  colorList[0.0] = clr;
  cbSize[0] = 0.05; cbSize[1] = 0.5;
  cbPos[0] = 0.0; cbPos[1] = 0.0;
  printf("LUT: setring\n");
}

LUT::~LUT()
{
}

Color LUT::ColorByValue(const float val)
{
  Color clr;

  map<float, Color>::iterator itr;
  itr = colorList.begin();
  if ( val < (*itr).first )
    return (*itr).second;
  float bfval = (*itr).first;
  Color bfclr = (*itr).second;
  for ( ; itr != colorList.end(); itr++ ) {
    if ( val <= (*itr).first ) {
      float df = (*itr).first - val;
      float vdf = (*itr).first - bfval;
      float a = df/vdf;
      clr.red = bfclr.red + ((*itr).second.red - bfclr.red) * a;
      clr.green = bfclr.green + ((*itr).second.green - bfclr.green) * a;
      clr.blue = bfclr.blue + ((*itr).second.blue - bfclr.blue) * a;
      return clr;
    }
    clr = (*itr).second;
    bfclr = (*itr).second;
  }

  return clr;
}
