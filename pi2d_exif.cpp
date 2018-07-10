// pi2d_exif
#include "Pi2D.h"

#ifndef PI2D_RET_TYPE
#define PI2D_RET_TYPE int
#define PI2D_SUCCEED 0
#define PI2D_FAILED  !PI2D_SUCCEED
#endif

static Pi2D* s_pi2dexif = NULL;

extern "C" {
  /*! 初期化
   */
  PI2D_RET_TYPE pi2d_init() {
    if ( s_pi2dexif ) return PI2D_SUCCEED;
    s_pi2dexif = new Pi2D();
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_init_() {
    return pi2d_init();
  }

  /*! 使用終了
   */
  PI2D_RET_TYPE pi2d_finalize() {
    if ( s_pi2dexif ) {
      delete s_pi2dexif;
      s_pi2dexif = NULL;
    }
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_finalize_() {
    return pi2d_finalize();
  }

  /*! 属性設定
   */
  PI2D_RET_TYPE pi2d_setattrib(char* arg, int arglen) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! arg || arglen < 1 ) return PI2D_FAILED;
    std::string strArg(arglen, '\0');
    for ( int i = 0; i < arglen; i++ ) strArg[i] = arg[i];
    if ( ! s_pi2dexif->SetAttrib(strArg) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_setattrib_(char* arg, int* arglen) {
    if ( ! arglen ) return PI2D_FAILED;
    return pi2d_setattrib(arg, *arglen);
  }

  /*! 座標値設定
   */
  PI2D_RET_TYPE pi2d_setcord(Real* arr, int veclen, int vecidxt[2]) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! s_pi2dexif->SetCoord(arr, veclen, vecidxt) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_setcord_(Real* arr, int* veclen, int vecidxt[2]) {
    if ( ! veclen ) return PI2D_FAILED;
    return pi2d_setcord(arr, *veclen, vecidxt);
  }

  /*! LUT 作成
   */
  PI2D_RET_TYPE pi2d_createlut(char* name, int namelen) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! name || namelen < 1 ) return PI2D_FAILED;
    std::string strName(namelen, '\0');
    for ( int i = 0; i < namelen; i++ ) strName[i] = name[i];
    LUT lut;
    if ( ! s_pi2dexif->SetLUT(strName, &lut) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_createlut_(char* name, int* namelen) {
    if ( ! namelen ) return PI2D_FAILED;
    return pi2d_createlut(name, *namelen);
  }

  /*! LUTの色設定
   */
  PI2D_RET_TYPE pi2d_setlutcolor(char* name, int namelen, int ncolor,
				 Real* valuelist, Real* colorlist) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! name || namelen < 1 ) return PI2D_FAILED;
    if ( ncolor < 1 || ! valuelist || ! colorlist ) return PI2D_FAILED;
    std::string strName(namelen, '\0');
    for ( int i = 0; i < namelen; i++ ) strName[i] = name[i];
    LUT lut;
    std::map<std::string, LUT>::iterator it
      = s_pi2dexif->m_lutList.find(strName);
    if ( it != s_pi2dexif->m_lutList.end() )
      lut = it->second;
    lut.colorList.clear();
    for ( int i = 0; i < ncolor; i++ ) {
      lut.colorList[valuelist[i]]
	= color_s(colorlist[i*3], colorlist[i*3+1], colorlist[i*3+2]);
    } // end of for(i)
    if ( ! s_pi2dexif->SetLUT(strName, &lut) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_setlutcolor_(char* name, int* namelen, int* ncolor,
				  Real* valuelist, Real* colorlist) {
    if ( ! namelen || ! ncolor ) return PI2D_FAILED;
    return pi2d_setlutcolor(name, *namelen, *ncolor, valuelist, colorlist);
  }

  /*! LUTの属性設定
   */
  PI2D_RET_TYPE pi2d_setlutattrib(char* name, int namelen,
				  char* arg, int arglen) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! name || namelen < 1 ) return PI2D_FAILED;
    if ( ! arg || arglen < 1 ) return PI2D_FAILED;
    std::string strName(arglen, '\0');
    for ( int i = 0; i < namelen; i++ ) strName[i] = name[i];
    std::string strArg(arglen, '\0');
    for ( int i = 0; i < arglen; i++ ) strArg[i] = arg[i];
    LUT lut;
    std::map<std::string, LUT>::iterator it
      = s_pi2dexif->m_lutList.find(strName);
    if ( it != s_pi2dexif->m_lutList.end() ) lut = it->second;
    
    size_t p = strArg.find("=");
    if ( p == std::string::npos ) return PI2D_FAILED;
    std::string kwd = strArg.substr(0, p);
    std::string val = strArg.substr(p+1);
    if ( kwd.size() < 1 || val.size() < 1 ) return PI2D_FAILED;
    if ( kwd == "cbSize" || kwd == "cbPos" ) {
      p = val.find(",");
      if ( p == std::string::npos ) return PI2D_FAILED;
      std::string val1 = val.substr(0, p);
      std::string val2 = val.substr(p+1);
      if ( val1.size() < 1 || val2.size() < 1 ) return PI2D_FAILED;
      if ( kwd == "cbSize" ) {
	lut.cbSize[0] = (Real)atof(val1.c_str());
	lut.cbSize[1] = (Real)atof(val2.c_str());
      } else {
 	lut.cbPos[0] = (Real)atof(val1.c_str());
	lut.cbPos[1] = (Real)atof(val2.c_str());
      }
    }
    else if ( kwd == "cbHoriz" ) {
      if ( val == "true" || val == "True" || val == "TRUE" ||
	   val == "yes" || val == "Yes" || val == "YES" || val == "1" )
	lut.cbHoriz = true;
      else if ( val == "false" || val == "False" || val == "FALSE" ||
		val == "no" || val == "No" || val == "NO" || val == "0" )
	lut.cbHoriz = false;
      else
	return PI2D_FAILED;
    }
    else if ( kwd == "cbNumTic" ) {
      int nt = atoi(val.c_str());
      if ( nt < 0 ) nt = 0;
      lut.cbNumTic = nt;
    }
    else return PI2D_FAILED;

    if ( ! s_pi2dexif->SetLUT(strName, &lut) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_setlutattrib_(char* name, int* namelen,
				   char* arg, int* arglen) {
    if ( ! namelen || ! arglen ) return PI2D_FAILED;
    return pi2d_setlutattrib(name, *namelen, arg, *arglen);
  }

  /*! スカラーデータの可視化描画
   */
  PI2D_RET_TYPE pi2d_draws(int vt, Real* data, char* lutname, int lutnamelen,
			   int nlevels, int cbshow) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! lutname || lutnamelen < 1 ) return PI2D_FAILED;
    std::string strLutName(lutnamelen, '\0');
    for ( int i = 0; i < lutnamelen; i++ ) strLutName[i] = lutname[i];
    CVType evt;
    if ( vt == 0 ) evt = ColorContour;
    else if ( vt == 1 ) evt = ContourLine;
    else return PI2D_FAILED;
    if ( ! s_pi2dexif->DrawS(evt, data, strLutName, nlevels, (cbshow==1)) )
      return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_draws_(int* vt, Real* data, char* lutname, int* lutnamelen,
			    int* nlevels, int* cbshow) {
    if ( ! vt || ! lutnamelen || ! nlevels || ! cbshow ) return PI2D_FAILED;
    return pi2d_draws(*vt, data, lutname, *lutnamelen, *nlevels, *cbshow);
  }
  
  /*! ベクトルデータの可視化描画
   */
  PI2D_RET_TYPE pi2d_drawv(Real* data, int veclen, int vecidx[2],
			   char* lutname, int lutnamelen,
			   int colidx, int cbshow) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! lutname || lutnamelen < 1 ) return PI2D_FAILED;
    std::string strLutName(lutnamelen, '\0');
    for ( int i = 0; i < lutnamelen; i++ ) strLutName[i] = lutname[i];
    if ( ! s_pi2dexif->DrawV(data, veclen, vecidx, strLutName,
			     colidx, (cbshow==1)) )
      return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_drawv_(Real* data, int* veclen, int vecidx[2],
			    char* lutname, int* lutnamelen,
			    int* colidx, int* cbshow) {
    if ( ! veclen || ! lutnamelen || ! colidx || ! cbshow ) return PI2D_FAILED;
    return pi2d_drawv(data, *veclen, vecidx, lutname, *lutnamelen,
		      *colidx, *cbshow);
  }

  /*! 可視化画像出力
   */
  PI2D_RET_TYPE pi2d_output(int step, int row, int col, int proc) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! s_pi2dexif->Output(step, row, col, proc) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_output_(int* step, int* row, int* col, int* proc) {
    if ( ! step || ! row || ! col || ! proc ) return PI2D_FAILED;
    return pi2d_output(*step, *row, *col, *proc);
  }

  /*! ファイルからの属性値読込み
   */
  PI2D_RET_TYPE pi2d_importattrib(char* path, int pathlen) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! path || pathlen < 1 ) return PI2D_FAILED;
    std::string strPath(pathlen, '\0');
    for ( int i = 0; i < pathlen; i++ ) strPath[i] = path[i];
    if ( ! s_pi2dexif->ImportAttrib(strPath) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_importattrib_(char* path, int* pathlen) {
    if ( ! pathlen ) return PI2D_FAILED;
    return pi2d_importattrib(path, *pathlen);
  }

  /*! ファイルへの属性値出力
   */
  PI2D_RET_TYPE pi2d_exportattrib(char* path, int pathlen) {
    if ( ! s_pi2dexif ) return PI2D_FAILED;
    if ( ! path || pathlen < 1 ) return PI2D_FAILED;
    std::string strPath(pathlen, '\0');
    for ( int i = 0; i < pathlen; i++ ) strPath[i] = path[i];
    if ( ! s_pi2dexif->ExportAttrib(strPath) ) return PI2D_FAILED;
    return PI2D_SUCCEED;
  }
  PI2D_RET_TYPE pi2d_exportattrib_(char* path, int* pathlen) {
    if ( ! pathlen ) return PI2D_FAILED;
    return pi2d_exportattrib(path, *pathlen);
  }
				 
} // end of extern "C"
