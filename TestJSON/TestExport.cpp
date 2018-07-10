#include "Pi2D.h"

int main(int argc, char** argv) {
  Pi2D p;
  p.m_imageSz[0] = p.m_imageSz[1] = 1000;
  p.m_arraySz[0] = p.m_arraySz[1] = 512;
  p.m_viewPort[2] = p.m_viewPort[3] = 500.0;
  p.m_outputPtn = "Output_%R3_%C3_%S6.png";
  p.m_lineWidth = 1.5;
  p.m_vectorMag = 0.1;
  p.m_vectorHeadRatio[0] = p.m_vectorHeadRatio[1] = 2.5;

  LUT lut;
  lut.colorList[0.0] = color_s(0.01, 0.01, 0.01);
  lut.colorList[100.0] = color_s(1.0, 1.0, 1.0);
  lut.cbPos[0] = 0.1; lut.cbPos[1] = 0.1;
  p.m_lutList["gray"] = lut;

  lut.colorList[0.0] = color_s(0.0, 0.0, 1.0);
  lut.colorList[50.0] = color_s(0.0, 1.0, 0.0);
  lut.colorList[100.0] = color_s(1.0, 0.0, 0.0);
  lut.cbHoriz = true;
  p.m_lutList["rgb"] = lut;

  p.ExportAttrib("X.json");

  return 0;
}
