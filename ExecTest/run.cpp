#include <stdlib.h>
#include "../Pi2D.h"

int main()
{
  Pi2D* pi0 = new Pi2D();

  bool res = true;

  if ( ! pi0->SetAttrib("imageSize=640, 480") )
    printf("error: etAttribute\n");
  if ( ! pi0->SetAttrib("arraySize=40, 20") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("viewport=-0.5, 40.5, -1.0, 20.2") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("outfilePat=./out_%S0%R1%C2%P3.png") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("lineWidth=1.5") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("vectorMag=2.1") )
    printf("error: SetAttrib\n");

  if ( res ) {
    printf("imageSize: %d,%d\n", pi0->m_imageSz[0], pi0->m_imageSz[1]);
    printf("arraySize: %d,%d\n", pi0->m_arraySz[0], pi0->m_arraySz[1]);
    printf("viewport: %f,%f,%f,%f\n", pi0->m_viewPort[0],
           pi0->m_viewPort[1], pi0->m_viewPort[2], pi0->m_viewPort[3]);
    printf("outfilePat: %s\n", pi0->m_outputPtn.c_str());
    printf("lineWidth: %f\n", pi0->m_lineWidth);
    printf("vectorMag: %f\n", pi0->m_vectorMag);
  }

  //printf("dbg0\n");
  LUT* lut1 = new LUT();
  color_s clr0(0.0, 0.0, 1.0);
  lut1->colorList[1.0] = clr0;
  //printf("dbg1\n");
  if ( ! pi0->SetLUT("LUT_B", lut1) )
    printf("error: SetLUT\n");
  //printf("dbg2\n");

  int i, j;
  int n;
  Real* c_arr = new Real[40 * 20 * 2];
  for (i = 0; i < 20; i++){
    for (j = 0; j < 40; j++){
      n = (40*i + j) * 2;
      c_arr[n] = j;
      c_arr[n+1] = i;
    }
  }
  int vid[2] = {0, 1};
  if ( ! pi0->SetCoord(c_arr, 2, vid) )
    printf("error: SetCoord\n");

  Real* z_arr = new Real[20 * 40];
  for (i = 0; i < 20; i++){
    for (j = 0; j < 40; j++){
      z_arr[40*i + j] = i + j;
    }
  } 

  if ( ! pi0->DrawS(ColorContour, z_arr, "", 20) )
    printf("error: draw contour(f)\n");

  Real* v_arr = new Real[40 * 20 * 2];
  for (i = 0; i < 20; i++){
    for (j = 0; j < 40; j++){
      n = (40*i + j) * 2;
      v_arr[n] = j;
      v_arr[n+1] = i;
    }
  }

  if ( ! pi0->DrawV(v_arr) )
    printf("error: draw vector\n");

  if ( ! pi0->Output(1, 22, 3, 4) )
    printf("error: save\n");

  delete[] c_arr;
  delete[] z_arr;
  delete lut1;
  //delete pi0;
}
