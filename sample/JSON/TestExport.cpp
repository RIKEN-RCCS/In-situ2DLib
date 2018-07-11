#include "Pi2D.h"

int main(int argc, char** argv) {
  Pi2D pi2d;
  char attrBuff[128];

  // set attributes
  sprintf(attrBuff, "imageSize=%d,%d", 1024, 512);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set imageSize failed\n");
    exit(1);
  }

  sprintf(attrBuff, "arraySize=%d,%d", 256, 256);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set arraySize failed\n");
    exit(1);
  }

  sprintf(attrBuff, "viewport=%g,%g,%g,%g", 0.0, 0.0, 500.0, 500.0);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set viewport failed\n");
    exit(1);
  }
  
  sprintf(attrBuff, "outfilePat=Output_%%R3_%%C3_%%S6.png");
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set outfilePat failed\n");
    exit(1);
  }

  sprintf(attrBuff, "lineWidth=%g", 1.5);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set lineWidth failed\n");
    exit(1);
  }
  
  sprintf(attrBuff, "vectorMag=%g", 0.1);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set vectorMag failed\n");
    exit(1);
  }

  sprintf(attrBuff, "vectorHeadRatio=%g,%g", 2.5, 2.5);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set vectorHeadRatio failed\n");
    exit(1);
  }

  sprintf(attrBuff, "bgColor=%g,%g,%g", 0.2, 0.2, 0.2);
  if ( ! pi2d.SetAttrib(attrBuff) ) {
    fprintf(stderr, "set bgColor failed\n");
    exit(1);
  }

  // set LUTs
  LUT lut;
  lut.colorList[0.0] = color_s(0.01, 0.01, 0.01);
  lut.colorList[100.0] = color_s(1.0, 1.0, 1.0);
  lut.cbPos[0] = 0.1; lut.cbPos[1] = 0.1;
  pi2d.m_lutList["gray"] = lut;

  lut.colorList[0.0] = color_s(0.0, 0.0, 1.0);
  lut.colorList[50.0] = color_s(0.0, 1.0, 0.0);
  lut.colorList[100.0] = color_s(1.0, 0.0, 0.0);
  lut.cbHoriz = true;
  pi2d.m_lutList["rgb"] = lut;

  // export attributes to JSON file
  if ( ! pi2d.ExportAttrib("X.json") ) {
    fprintf(stderr, "export attributes failed\n");
    exit(1);
  }

  exit(0);
}
