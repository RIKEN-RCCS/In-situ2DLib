#include "LUT.h"

using namespace std;

LUT::LUT() : cbHoriz(false), cbNumTic(2)
{
  color_s clr(1.0, 1.0, 1.0);
  colorList[0.0] = clr;
  cbSize[0] = 0.05; cbSize[1] = 0.5;
  cbPos[0] = 0.0; cbPos[1] = 0.0;
  //color_s bgClr(1.0, 1.0, 1.0);
  //cbTicColor = bgClr;
  cbTicColor[0]= 1.0;
  cbTicColor[1]= 1.0;
  cbTicColor[2]= 1.0;
}

LUT::~LUT()
{
}

color_s LUT::ColorByValue(const float val)
{
  color_s clr;

  map<float, color_s>::iterator itr;
  itr = colorList.begin();
  if ( colorList.size() == 1 )
    return (*itr).second;
  if ( val <= (*itr).first )
    return (*itr).second;
  float bfval = (*itr).first;
  color_s bfclr = (*itr).second;
  itr++;
  for ( ; itr != colorList.end(); itr++ ) {
    if ( val <= (*itr).first ) {
      float df = val - bfval;
      float vdf = (*itr).first - bfval;
      float a = df/vdf;
      clr.red = bfclr.red + ((*itr).second.red - bfclr.red) * a;
      clr.green = bfclr.green + ((*itr).second.green - bfclr.green) * a;
      clr.blue = bfclr.blue + ((*itr).second.blue - bfclr.blue) * a;
      return clr;
    }
    clr = (*itr).second;
    bfval = (*itr).first;
    bfclr = (*itr).second;
  }

  return clr;
}
