#include "LUT.h"

using namespace std;

LUT::LUT()
{
  color_s clr = {1.0, 1.0, 1.0};
  m_colorList[0.0] = clr;
}

LUT::~LUT()
{
}

color_s LUT::ColorByValue(const float val)
{
  color_s clr;

  map<float, color_s>::iterator itr;
  itr = m_colorList.begin();
  if ( val < (*itr).first )
    return (*itr).second;
  for ( ; itr != m_colorList.end(); itr++ ) {
    if ( val <= (*itr).first )
    //if ( val == (*itr).first )
      return (*itr).second;
    clr = (*itr).second;
  }

  return clr;
}
