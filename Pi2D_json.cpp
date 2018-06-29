//
// class Pi2D
//
#include "Pi2D.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

#include "picojson.h"

using namespace std;

// STATIC util
static bool ParseLUT(Pi2D* pi2d, picojson::object& lutObjs) {
  if ( ! pi2d ) return false;

  // clear lutList of pi2d
  pi2d->m_lutList.clear();

  // parse lut series
  picojson::object::iterator lit;
  for ( lit = lutObjs.begin(); lit != lutObjs.end(); lit ++ ) {
    string lutName = lit->first;
    picojson::object lutO = lit->second.get<picojson::object>();
    LUT xlut;

    picojson::object::iterator lait;
    for ( lait = lutO.begin(); lait != lutO.end(); lait++ ) {
      string kwd = lait->first;
      if ( kwd == "cbSize" ) {
	picojson::array xarr = lait->second.get<picojson::array>();
	if ( xarr.size() < 2 ) continue;
	xlut.cbSize[0] = (int)xarr[0].get<double>();
	xlut.cbSize[1] = (int)xarr[1].get<double>();
      }
      else if ( kwd == "cbPos" ) {
	picojson::array xarr = lait->second.get<picojson::array>();
	if ( xarr.size() < 2 ) continue;
	xlut.cbPos[0] = (int)xarr[0].get<double>();
	xlut.cbPos[1] = (int)xarr[1].get<double>();
      }
      else if ( kwd == "cbHoriz" ) {
	xlut.cbHoriz = lait->second.get<bool>();
      }
      else if ( kwd == "cbNumTic" ) {
	xlut.cbNumTic = (size_t)lait->second.get<double>();
      }
      else if ( kwd == "colorList" ) {
	picojson::array xarr = lait->second.get<picojson::array>();
	for ( size_t i = 0; i < xarr.size(); i++ ) {
	  picojson::array yarr = xarr[i].get<picojson::array>();
	  if ( yarr.size() < 2 ) continue;
	  Real val = (Real)yarr[0].get<double>();
	  picojson::array zarr = yarr[1].get<picojson::array>();
	  if ( zarr.size() < 3 ) continue;
	  color_s rgb;
	  rgb.red = (float)zarr[0].get<double>();
	  rgb.green = (float)zarr[1].get<double>();
	  rgb.blue = (float)zarr[2].get<double>();
	  xlut.colorList[val] = rgb;
	} // end of for(i)
      }
    } // end of for(lait)

    pi2d->m_lutList[lutName] = xlut;
  } // end of for(lit)
  return true;
}


bool Pi2D::ImportAttrib(const string path)
{
  if ( path.empty() )
    return false;

  // ファイルオープン
  ifstream fs;
  fs.open(path.c_str());
  if ( ! fs ) return false;

  // picojsonへ読み込み、ファイルクローズ
  picojson::value val;
  try {
    fs >> val;
  } catch (const exception& e) {
    fs.close();
    return false;
  }
  fs.close();

  // トップレベルオブジェクトを取得
  picojson::object root, Pi2DAttr;
  try {
    root = val.get<picojson::object>();
    Pi2DAttr = root["Pi2DAttr"].get<picojson::object>();
  } catch (const exception& e) {
    return false;
  }

  // 属性のパース
  picojson::object::iterator it;
  for ( it = Pi2DAttr.begin(); it != Pi2DAttr.end(); it++ ) {
    string kwd = it->first;
    if ( kwd == "imageSize" ) {
      picojson::array xarr = it->second.get<picojson::array>();
      if ( xarr.size() < 2 ) continue;
      m_imageSz[0] = (int)xarr[0].get<double>();
      m_imageSz[1] = (int)xarr[1].get<double>();
    }
    else if ( kwd == "arraySize" ) {
      picojson::array xarr = it->second.get<picojson::array>();
      if ( xarr.size() < 2 ) continue;
      m_arraySz[0] = (int)xarr[0].get<double>();
      m_arraySz[1] = (int)xarr[1].get<double>();
    }
    else if ( kwd == "viewport" ) {
      picojson::array xarr = it->second.get<picojson::array>();
      if ( xarr.size() < 4 ) continue;
      for ( int i = 0; i < 4; i++ )
	m_viewPort[i] = (Real)xarr[i].get<double>();
    }
    else if ( kwd == "outfilePat" ) {
      m_outputPtn = it->second.get<string>();
    }
    else if ( kwd == "lineWidth" ) {
      m_lineWidth = (Real)it->second.get<double>();
    }
    else if ( kwd == "vectorMag" ) {
      m_vectorMag = (Real)it->second.get<double>();
    }
    else if ( kwd == "vectorHeadRatio" ) {
      picojson::array xarr = it->second.get<picojson::array>();
      if ( xarr.size() < 2 ) continue;
      m_vectorHeadRatio[0] = (Real)xarr[0].get<double>();
      m_vectorHeadRatio[1] = (Real)xarr[1].get<double>();
    }
    else if ( kwd == "LUT" ) {
      picojson::object lutObjs = it->second.get<picojson::object>();
      if ( ! ParseLUT(this, lutObjs) ) {
	continue;
      }
    }
    else continue; // unknown keyword, ignore.
  } // end of for(it)
  
  return true;
}

bool Pi2D::ExportAttrib(const string path)
{
  if ( path.empty() )
    return false;

  // ファイルオープン
  ofstream fs;
  fs.open(path.c_str());
  if ( ! fs ) return false;

  // JSONオブジェクト
  picojson::object root;
  picojson::object Pi2DAttr;

  // Pi2DAttr作成
  if ( m_imageSz[0] != 600 || m_imageSz[1] != 400 ) {
    picojson::array xarr;
    xarr.push_back(picojson::value((double)m_imageSz[0]));
    xarr.push_back(picojson::value((double)m_imageSz[1]));
    Pi2DAttr.insert(make_pair("imageSize", picojson::value(xarr)));
  }
  if ( m_arraySz[0] != -1 || m_arraySz[1] != -1 ) {
    picojson::array xarr;
    xarr.push_back(picojson::value((double)m_arraySz[0]));
    xarr.push_back(picojson::value((double)m_arraySz[1]));
    Pi2DAttr.insert(make_pair("arraySize", picojson::value(xarr)));
  }
  if ( m_viewPort[0] != 0 || m_viewPort[1] != 0 ||
       m_viewPort[2] != 0 || m_viewPort[3] != 0 ) {
    picojson::array xarr;
    xarr.push_back(picojson::value(m_viewPort[0]));
    xarr.push_back(picojson::value(m_viewPort[1]));
    xarr.push_back(picojson::value(m_viewPort[2]));
    xarr.push_back(picojson::value(m_viewPort[3]));
    Pi2DAttr.insert(make_pair("viewport", picojson::value(xarr)));
  }
  if ( m_outputPtn != "./outimage_%S6.png" ) {
    Pi2DAttr.insert(make_pair("outfilePat",
			      picojson::value(m_outputPtn.c_str())));
  }
  if ( m_lineWidth != 1.0 ) {
    Pi2DAttr.insert(make_pair("lineWidth", picojson::value(m_lineWidth)));
  }
  if ( m_vectorMag != 1.0 ) {
    Pi2DAttr.insert(make_pair("vectorMag", picojson::value(m_vectorMag)));
  }
  if ( m_vectorHeadRatio[0] != -1.0 || m_vectorHeadRatio[1] != -1.0 ) {
    picojson::array xarr;
    xarr.push_back(picojson::value(m_vectorHeadRatio[0]));
    xarr.push_back(picojson::value(m_vectorHeadRatio[1]));
    Pi2DAttr.insert(make_pair("vectorHeadRatio", picojson::value(xarr)));
  }

  // LUTList作成
  if ( ! m_lutList.empty() ) {
    picojson::object LUTList;
    map<string, LUT>::iterator lit;
    for ( lit = m_lutList.begin(); lit != m_lutList.end(); lit++ ) {
      LUT& lut = lit->second;
      picojson::object lutObj;
      picojson::array colorList;
      map<float, color_s>::iterator cit = lut.colorList.begin();
      for ( ; cit != lut.colorList.end(); cit++ ) {
	picojson::array carr, rgbarr;
	rgbarr.push_back(picojson::value(cit->second.red));
	rgbarr.push_back(picojson::value(cit->second.green));
	rgbarr.push_back(picojson::value(cit->second.blue));
	carr.push_back(picojson::value(cit->first));
	carr.push_back(picojson::value(rgbarr));
	colorList.push_back(picojson::value(carr));
      } // end of for(cit)
      lutObj.insert(make_pair("colorList", picojson::value(colorList)));
      if ( lut.cbSize[0] != 0.05 || lut.cbSize[1] != 0.5 ) {
	picojson::array xarr;
	xarr.push_back(picojson::value(lut.cbSize[0]));
	xarr.push_back(picojson::value(lut.cbSize[1]));
	lutObj.insert(make_pair("cbSize", picojson::value(xarr)));
      }
      if ( lut.cbPos[0] != 0.0 || lut.cbPos[1] != 0.0 ) {
	picojson::array xarr;
	xarr.push_back(picojson::value(lut.cbPos[0]));
	xarr.push_back(picojson::value(lut.cbPos[1]));
	lutObj.insert(make_pair("cbPos", picojson::value(xarr)));
      }
      if ( lut.cbHoriz != false ) {
	lutObj.insert(make_pair("cbHoriz", picojson::value(lut.cbHoriz)));
      }
      if ( lut.cbNumTic != 2 ) {
	lutObj.insert(make_pair("cbNumTic",
				picojson::value((double)lut.cbNumTic)));
      }
      LUTList.insert(make_pair(lit->first.c_str(), picojson::value(lutObj)));
    } // end of for(lit)
    Pi2DAttr.insert(make_pair("LUT", picojson::value(LUTList)));
  }

  // トップレベルオブジェクト作成とファイル出力
  root.insert(make_pair("Pi2DAttr", picojson::value(Pi2DAttr)));
  fs << picojson::value(root).serialize() << endl;
  fs.close();

  return true;
}
