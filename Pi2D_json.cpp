//
// class Pi2D
//
#include "Pi2D.h"

#include <iostream> // std::cout
#include <fstream>  // std::ifstream
#include <cstdio>

#include "picojson.h"

using namespace std;


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
  fs >> val;
  fs.close();

  // トップレベルオブジェクトを取得
  picojson::object o, o0;
  try {
    o = val.get<picojson::object>();
    o0 = o["Pi2DAttr"].get<picojson::object>();
  } catch (const exception& e) {
    return false;
  }


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
