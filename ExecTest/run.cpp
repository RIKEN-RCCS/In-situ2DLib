#include <stdlib.h>
#include "Pi2D.h"

int main()
{
  // オブジェクト1の作成
  Pi2D* pi0 = new Pi2D();
  // オブジェクト2の作成
  Pi2D* pi1 = new Pi2D();

  bool res = false;

  // オブジェクト1の属性の設定
  if ( ! pi0->SetAttrib("imageSize=640, 480") )
    printf("error: etAttribute\n");
  if ( ! pi0->SetAttrib("arraySize=40, 20") )
    printf("error: SetAttrib\n");
  //if ( ! pi0->SetAttrib("viewport=-0.5, 40.5, -1.0, 20.2") )
  //  printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("outfilePat=./out_%S0%R1%C2%P3.png") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("lineWidth=1.5") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("vectorMag=0.5") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("vectorHeadRatio=5.0, 5.0") )
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

  // オブジェクト2の属性の設定
  if ( ! pi1->SetAttrib("imageSize=640, 640") )
    printf("error: etAttribute\n");
  if ( ! pi1->SetAttrib("arraySize=40, 20") )
    printf("error: SetAttrib\n");
  if ( ! pi1->SetAttrib("outfilePat=./out1_%S5.png") )
    printf("error: SetAttrib\n");
  if ( ! pi1->SetAttrib("bgColor=0.7, 0.4, 0.7") )
    printf("error: SetAttrib\n");

  // オブジェクト1のLUTの設定
  LUT* lut0 = new LUT();
  color_s clr0(0.0, 0.0, 1.0);
  lut0->colorList[50.0] = clr0;
  lut0->cbPos[0] = 0.1;
  lut0->cbPos[1] = 0.1;
  lut0->cbSize[0] = 0.8;
  lut0->cbSize[1] = 0.1;
  lut0->cbHoriz = true;
  lut0->cbTicColor[0] = 0.0;
  if ( ! pi0->SetLUT("LUT_B", lut0) )
    printf("error: SetLUT\n");

  LUT* lut1 = new LUT();
  color_s clr2(0.0, 1.0, 0.0);
  lut1->colorList[10.5] = clr2;
  color_s clr1(1.0, 1.0, 0.0);
  lut1->colorList[20.0] = clr1;
  lut1->cbPos[0] = 0.1;
  lut1->cbPos[1] = 0.3;
  lut1->cbSize[0] = 0.8;
  lut1->cbSize[1] = 0.1;
  lut1->cbHoriz = true;
  lut1->cbNumTic = 3;
  if ( ! pi0->SetLUT("LUT_Y", lut1) )
    printf("error: SetLUT\n");

  // オブジェクト2のLUTの設定
  LUT* lut2 = new LUT();
  color_s clr3(0.0, 1.0, 1.0);
  lut2->colorList[0.0] = clr3;
  if ( ! pi1->SetLUT("LUT_S", lut2) )
    printf("error: SetLUT\n");

  // coordデータの作成
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

  // 等高線データの作成
  Real* z_arr = new Real[20 * 40];
  for (i = 0; i < 20; i++){
    for (j = 0; j < 40; j++){
      z_arr[40*i + j] = i + j;
    }
  } 

  // オブジェクト1のcoordデータの設定
  int vid[2] = {0, 1};
  if ( ! pi0->SetCoord(c_arr, 2, vid) )
    printf("error: SetCoord\n");

  // オブジェクト1の等高線図(塗潰し)の描画
  if ( ! pi0->DrawS(ColorContour, z_arr, "LUT_B", 20, true) )
    printf("error: draw contour(f)\n");

  // オブジェクト2のcoordデータの設定
  if ( ! pi1->SetCoord(c_arr, 2, vid) )
    printf("error: SetCoord\n");

  // オブジェクト2の等高線図(線描画)の描画
  if ( ! pi1->DrawS(ContourLine, z_arr, "", 10, false) )
    printf("error: draw contour(f)\n");
  //if ( ! pi1->DrawS(ContourLine, z_arr, "LUT_S") )
  //  printf("error: draw contour(f)\n");

  // ベクトルデータの作成
  Real* v_arr = new Real[40 * 20 * 2];
  for (i = 0; i < 20; i++){
    for (j = 0; j < 40; j++){
      n = (40*i + j) * 2;
      v_arr[n] = j;
      v_arr[n+1] = i;
    }
  }

  // オブジェクト1の属性の設定
  if ( ! pi0->SetAttrib("lineWidth=2.0") )
    printf("error: SetAttrib\n");
  if ( ! pi0->SetAttrib("vectorMag=0.5") )
    printf("error: SetAttrib\n");

  // オブジェクト1のベクトル図の描画
  if ( ! pi0->DrawV(v_arr, 2, vid, "LUT_Y", -1, true) )
    printf("error: draw vector\n");

  // オブジェクト1の画像の出力
  if ( ! pi0->Output(1, 22, 3, 4) )
    printf("error: save\n");
  // オブジェクト2の画像の出力
  if ( ! pi1->Output(8, 0, 0, 0) )
    printf("error: save\n");

  // オブジェクト1を使った等高線図の出力の繰り返し
  if ( ! pi0->DrawS(ColorContour, z_arr, "LUT_B", 20, true) )
    printf("error: draw contour(f)\n");
  if ( ! pi0->Output(2, 22, 3, 4) )
    printf("error: save\n");

  delete[] c_arr;
  delete[] z_arr;
}
