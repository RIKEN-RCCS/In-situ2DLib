#include "Pi2D.h"
#include "numpy/arrayobject.h"
#include <complex>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <istream>

#if PY_MAJOR_VERSION >= 3
int init_numpy() {import_array();}
#else
void init_numpy() {import_array();}
#endif

#ifdef _REAL_DBL
int NPY_REAL = NPY_DOUBLE;
#else
int NPY_REAL = NPY_FLOAT;
#endif

using namespace std;

size_t Pi2D::s_id = 0;
bool Pi2D::s_debugprint = true;


static string _trim(const string& str, const char* trimCharList=" \t\r\n")
{
  string result;
  string::size_type left = str.find_first_not_of(trimCharList);
  if ( left != string::npos ) {
    string::size_type right = str.find_last_not_of(trimCharList);
    result = str.substr(left, right - left + 1);
  }
  return result;
}

vector<string> _split(const string& str, const char delimChar=',')
{
  vector<string> v;
  stringstream ss(str);
  string buf;
  while ( getline(ss, buf, delimChar) ) {
    v.push_back(buf);
  }
  return v;
}


Pi2D::Pi2D()
{
  m_imageSz[0] = 600;
  m_imageSz[1] = 400;
  m_arraySz[0] = -1;
  m_arraySz[1] = -1;
  for ( int i = 0; i < 4; i++ )
    m_viewPort[i] = 0.0;
  m_outputPtn = "./outimage_%S6.png";
  m_coord = NULL;
  m_lutList.clear();
  m_lineWidth = 1.0;
  m_vectorMag = 1.0;
  m_vectorHeadRatio[0] = -1.0;
  m_vectorHeadRatio[1] = -1.0;
  m_bgColor[0] = 0.0;
  m_bgColor[1] = 0.0;
  m_bgColor[2] = 0.0;

  m_registLut.clear();

  m_veclen = 2;
  m_vecid[0] = 0;
  m_vecid[1] = 1;
  m_veclen_v = 2;
  m_vecid_v[0] = 0;
  m_vecid_v[1] = 1;
 
  pModule = NULL;
  pFuncDrawS = NULL;
  pFuncDrawV = NULL;
  pFuncDrawCB = NULL;
  pFuncOut = NULL;

  if ( s_id == 0 ) {
    Py_Initialize();
    init_numpy();
    if ( s_debugprint && PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
    }

    string pi2dDir = ".";
    char* pi2dDirEnv = getenv("PI2D_DIR");
    if ( pi2dDirEnv && strlen(pi2dDirEnv) > 0 )
      pi2dDir = pi2dDirEnv;
    
    string cmd;
    cmd += "from __future__ import print_function\n";
    cmd += "import sys; sys.path.append('" + pi2dDir + "')\n";
    PyRun_SimpleString(cmd.c_str());
    if ( s_debugprint && PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
    }
    cmd = "import matplotlib\n";
    cmd += "matplotlib.use('Agg')\n";
    cmd += "import numpy as np\n";
    cmd += "import matplotlib.cm as cm\n";
    cmd += "import matplotlib.pyplot as plt\n";
    cmd += "from matplotlib.colors import LinearSegmentedColormap\n";
    PyRun_SimpleString(cmd.c_str());
    if ( s_debugprint && PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
    }
  }

  m_id = s_id;
  s_id++;

  pModule = PyImport_ImportModule("Pi2D");
  if ( ! pModule || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    return;
  }
  pFuncDrawS = PyObject_GetAttrString(pModule, "DrawS");
  if ( ! pFuncDrawS || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    return;
  }
  pFuncDrawV = PyObject_GetAttrString(pModule, "DrawV");
  if ( ! pFuncDrawV || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    return;
  }
  pFuncDrawCB = PyObject_GetAttrString(pModule, "DrawCB");
  if ( ! pFuncDrawCB || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    return;
  }
  pFuncOut = PyObject_GetAttrString(pModule, "Output");
  if ( ! pFuncOut || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    return;
  }
}

Pi2D::~Pi2D()
{
  if ( m_coord )
    delete[] m_coord;

  m_lutList.clear();
  m_registLut.clear();

  // python  decref
  if ( pFuncDrawS )
    Py_DECREF(pFuncDrawS);
  if ( pFuncDrawV )
    Py_DECREF(pFuncDrawV);
  if ( pFuncDrawCB )
    Py_DECREF(pFuncDrawCB);
  if ( pFuncOut )
    Py_DECREF(pFuncOut);
  if ( pModule )
    Py_DECREF(pModule);
}

bool Pi2D::SetAttrib(const string xarg)
{
  string arg = _trim(xarg);
  
  // check empty argument
  if ( arg.empty() ) {
    return false;
  }

  // separete into attribute and value
  size_t p = arg.find("=");
  if  ( p == string::npos ) {
    return false;
  }
  string attr = arg.substr(0, p);
  string vals = arg.substr(p+1);

  if ( attr == "imageSize" ) {
    int w, h;
    vector<string> vs = _split(vals);
    if ( vs.size() < 2 ) return false;
    w = atoi(vs[0].c_str());
    h = atoi(vs[1].c_str());
    if ( w < 0 || h < 0 ) return false;
    m_imageSz[0] = w;
    m_imageSz[1] = h;
  } else if ( attr == "arraySize" ) {
    int w, h;
    vector<string> vs = _split(vals);
    if ( vs.size() < 2 ) return false;
    w = atoi(vs[0].c_str());
    h = atoi(vs[1].c_str());
    if ( w < 0 || h < 0 ) return false;
    m_arraySz[0] = w;
    m_arraySz[1] = h;
  } else if ( attr == "viewport" ) {
    double x0, x1, y0, y1;
    vector<string> vs = _split(vals);
    if ( vs.size() < 4 ) return false;
    x0 = atof(vs[0].c_str());
    x1 = atof(vs[1].c_str());
    y0 = atof(vs[2].c_str());
    y1 = atof(vs[3].c_str());
    m_viewPort[0] = (Real)x0;
    m_viewPort[1] = (Real)x1;
    m_viewPort[2] = (Real)y0;
    m_viewPort[3] = (Real)y1;
  } else if ( attr == "outfilePat" ) {
    if ( vals.empty() ) return false;
    m_outputPtn = vals;
  } else if ( attr == "lineWidth" ) {
    double w;
    int n = sscanf(vals.c_str(), "%lf", &w);
    if ( n != 1 ) return false;
    if ( w < 0.0 ) return false;
    m_lineWidth = (Real)w;
  } else if ( attr == "vectorMag" ) {
    double v;
    int n = sscanf(vals.c_str(), "%lf", &v);
    if ( n != 1 ) return false;
    if ( v < 0.0 ) return false;
    m_vectorMag = (Real)v;
  } else if ( attr == "vectorHeadRatio" ) {
    double r0, r1;
    vector<string> vs = _split(vals);
    if ( vs.size() < 2 ) return false;
    r0 = atof(vs[0].c_str());
    r1 = atof(vs[1].c_str());
    if ( r0 != -1 && r0 < 0.0 ) return false;
    if ( r1 != -1 && r1 < 0.0 ) return false;
    m_vectorHeadRatio[0] = (Real)r0;
    m_vectorHeadRatio[1] = (Real)r1;
  } else if ( attr == "bgColor" ) {
    double r, g, b;
    vector<string> vs = _split(vals);
    if ( vs.size() < 3 ) return false;
    r = atof(vs[0].c_str());
    g = atof(vs[1].c_str());
    b = atof(vs[2].c_str());
    m_bgColor[0] = (Real)r;
    m_bgColor[1] = (Real)g;
    m_bgColor[2] = (Real)b;
  } else {
    return false;
  }

  return true;
}

bool Pi2D::SetCoord(const Real* arr, const int veclen,
                    const int* vecid)
{
  // set null
  if ( ! arr ) {
    free(m_coord);
    m_coord = NULL;
    return true;
  }

  if ( ! vecid ) {
    m_vecid[0] = 0;
    m_vecid[1] = 1;
  }

  if ( veclen < 2 )
    return false;
  if ( vecid[0] < 0  || vecid[0] >= veclen )
    return false;
  if ( vecid[1] < 0  || vecid[1] >= veclen )
    return false;
  m_veclen = veclen;
  m_vecid[0] = vecid[0];
  m_vecid[1] = vecid[1];

  int sz = m_arraySz[0] * m_arraySz[1] * m_veclen;
  m_coord = (Real*)realloc(m_coord, sz * sizeof(Real));

  if ( ! m_coord )
    return false;

  for ( int i = 0; i < sz; i++ )
    m_coord[i] = arr[i];

  return true;
}

bool Pi2D::SetLUT(const std::string name, const LUT* lut)
{
  map<string, LUT>::iterator itr;
  for ( itr = m_lutList.begin(); itr != m_lutList.end(); itr++ ) {
    if ( (*itr).first == name ) {
      if ( ! lut ) {
         m_lutList.erase(itr);
         return true; 
      }

      m_lutList[name] = *lut;
      return true;
    }
  }

  m_lutList[name] = *lut;

  return true;
}

bool Pi2D::DrawS(const CVType vt, const Real* data,
                 const string lutname, const int nlevels,
                 bool cbShow)
{
  int sz;

  bool ret = true;

  long int dims2[1] = {2};
  long int dims4[1] = {4};

  if ( lutname != "" ) {
    bool exist = false;
    map<std::string, LUT>::iterator itr = m_lutList.begin();
    while( itr != m_lutList.end() ) {
      if ( lutname == (*itr).first ) {
        exist = true;
        break;
      }
      ++itr;
    }
    if ( ! exist )
      return false;
  }
  LUT lut = m_lutList[lutname];
  if ( cbShow )
    m_registLut.insert(lutname);

  // set ID
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set ImageSize
  npy_intp imgsz[2] = {m_imageSz[0], m_imageSz[1]};
  PyObject *pImgSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(imgsz));
  if ( ! pImgSz || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set ArraySize
  npy_intp arrsz[2] = {m_arraySz[1], m_arraySz[0]};
  PyObject *pArrSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(arrsz));
  if ( ! pArrSz || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set viewPort
  PyObject* pVP =
      PyArray_SimpleNewFromData(1, dims4, NPY_REAL,
                                reinterpret_cast<void*>(m_viewPort));
  if ( ! pVP || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set contour type
  PyObject* pCtype = NULL;
  if ( vt == ColorContour )
    pCtype = PyLong_FromLong(0);
  else if ( vt == ContourLine )
    pCtype = PyLong_FromLong(1);
  if ( ! pCtype || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set veclen
  PyObject* pVlen = PyLong_FromLong(m_veclen);
  if ( ! pVlen || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set vecid
  npy_intp vid[2] = {m_vecid[0],  m_vecid[1]};
  PyObject *pVid = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(vid));
  if ( ! pVid || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set coord
  PyObject* pCoord;
  Real null_coord[1] = {0.0};
  if ( m_coord ) {
    sz = m_arraySz[0] * m_arraySz[1] * m_veclen;
    long int cdims[1] = {sz};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(m_coord));
  } else {
    long int cdims[1] = {1};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(null_coord));
  }
  if ( ! pCoord || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set data
  npy_intp z_dims[2] = {m_arraySz[1], m_arraySz[0]};
  PyObject* pZarr =
      PyArray_SimpleNewFromData(2, z_dims, NPY_REAL, (void*)(data));
  if ( ! pZarr || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set lut name
  PyObject* pLut;
#if PY_MAJOR_VERSION >= 3
  pLut = PyUnicode_FromString(lutname.c_str());
#else
  pLut = PyString_FromString(lutname.c_str());
#endif
  if ( ! pLut || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set nlevels
  PyObject* pNlevel = PyLong_FromLong(nlevels);
  if ( ! pNlevel || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set bgColor
  long int dims3[1] = {3};
  PyObject* pBGClr =
      PyArray_SimpleNewFromData(1, dims3, NPY_REAL,
                                reinterpret_cast<void*>(m_bgColor));
  if ( ! pBGClr || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set cbShow
  PyObject* pShow;
  if ( cbShow )
    pShow = Py_True;
  else
    pShow = Py_False;

  // set lineWidth
  PyObject* pWidth = PyFloat_FromDouble((double)m_lineWidth);
  if ( ! pWidth || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set Color
  sz = lut.colorList.size();
  Real vals[sz];
  Real clrs[sz][3];
  int cnt = 0;
  map<float, color_s>::iterator itr = lut.colorList.begin();
  while( itr != lut.colorList.end() ) {
    vals[cnt] = (*itr).first;
    clrs[cnt][0] = (*itr).second.red;
    clrs[cnt][1] = (*itr).second.green;
    clrs[cnt][2] = (*itr).second.blue;
    cnt++;
    ++itr;
  }
  npy_intp clr_dims1[1] = {sz};
  npy_intp clr_dims2[2] = {sz, 3};
  PyObject* pClrPos =
      PyArray_SimpleNewFromData(1, clr_dims1, NPY_REAL, (void*)vals);
  if ( ! pClrPos || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }
  PyObject* pClr =
      PyArray_SimpleNewFromData(2, clr_dims2, NPY_REAL, (void*)clrs);
  if ( ! pClr || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // call python function
  PyObject* pRet;
  if ( ret ) {
    pRet = PyObject_CallFunctionObjArgs(pFuncDrawS,
                   pId, pImgSz, pVP, pArrSz, pCoord, pVlen, pVid,
                   pCtype, pZarr, pLut, pNlevel, pBGClr, pShow,
                   pWidth, pClrPos, pClr, NULL);
    if ( ! pRet )
      ret = false;
    if ( pRet == Py_False )
      ret = false;
    if ( PyErr_Occurred() && s_debugprint )
      PyErr_Print();
  }

  if ( pId ) Py_DECREF(pId);
  if ( pImgSz ) Py_DECREF(pImgSz);
  if ( pVP ) Py_DECREF(pVP);
  if ( pArrSz ) Py_DECREF(pArrSz);
  if ( pCoord ) Py_DECREF(pCoord);
  if ( pCtype ) Py_DECREF(pCtype);
  if ( pVlen ) Py_DECREF(pVlen);
  if ( pVid ) Py_DECREF(pVid);
  if ( pZarr ) Py_DECREF(pZarr);
  if ( pLut ) Py_DECREF(pLut);
  if ( pNlevel ) Py_DECREF(pNlevel);
  if ( pBGClr ) Py_DECREF(pBGClr);
  if ( pShow ) Py_DECREF(pShow);
  if ( pWidth ) Py_DECREF(pWidth);
  if ( pClrPos ) Py_DECREF(pClrPos);
  if ( pClr ) Py_DECREF(pClr);
  if ( pRet ) Py_DECREF(pRet);

  return ret;
}

bool Pi2D::DrawV(const Real* data, const int veclen,
                 const int* vecid, const string lutname,
                 const int colid, bool cbShow)
{
  m_veclen_v = veclen;
  if ( vecid ) {
    m_vecid_v[0] = vecid[0];
    m_vecid_v[1] = vecid[1];
  } else {
    m_vecid_v[0] = 0;
    m_vecid_v[1] = 1;
  }

  int sz;

  bool ret = true;

  long int dims2[1] = {2};
  long int dims3[1] = {3};
  long int dims4[1] = {4};

  if ( lutname != "" ) {
    bool exist = false;
    map<std::string, LUT>::iterator itr = m_lutList.begin();
    while( itr != m_lutList.end() ) {
      if ( lutname == (*itr).first ) {
        exist = true;
        break;
      }
      ++itr;
    }
    if ( ! exist )
      return false;
  }
  LUT lut = m_lutList[lutname];
  if ( cbShow )
    m_registLut.insert(lutname);

  // set ID
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set ImageSize
  npy_intp imgsz[2] = {m_imageSz[0], m_imageSz[1]};
  PyObject *pImgSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(imgsz));
  if ( ! pImgSz || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set ArraySize
  npy_intp arrsz[2] = {m_arraySz[1], m_arraySz[0]};
  PyObject *pArrSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(arrsz));
  if ( ! pArrSz || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set viewPort
  PyObject* pVP =
      PyArray_SimpleNewFromData(1, dims4, NPY_REAL,
                                reinterpret_cast<void*>(m_viewPort));
  if ( ! pVP || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set veclen
  PyObject* pVlen = PyLong_FromLong(m_veclen);
  if ( ! pVlen || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set vecid
  npy_intp vid[2] = {m_vecid[0],  m_vecid[1]};
  PyObject *pVid = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(vid));
  if ( ! pVid || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set coord
  PyObject* pCoord;
  Real null_coord[1] = {0.0};
  if ( m_coord ) {
    sz = m_arraySz[0] * m_arraySz[1] * m_veclen;
    long int cdims[1] = {sz};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(m_coord));
  } else {
    long int cdims[1] = {1};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(null_coord));
  }
  if ( ! pCoord || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set data
  sz = m_arraySz[0] * m_arraySz[1] * m_veclen_v;
  long int cdims[1] = {sz};
  PyObject* pVal =
      PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(data));
  if ( ! pVal || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set lut name
  PyObject* pLut;
#if PY_MAJOR_VERSION >= 3
  pLut = PyUnicode_FromString(lutname.c_str());
#else
  pLut = PyString_FromString(lutname.c_str());
#endif
  if ( ! pLut || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set bgColor
  PyObject* pBGClr =
      PyArray_SimpleNewFromData(1, dims3, NPY_REAL,
                                reinterpret_cast<void*>(m_bgColor));
  if ( ! pBGClr || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set cbShow
  PyObject* pShow;
  if ( cbShow )
    pShow = Py_True;
  else
    pShow = Py_False;

  // set lineWidth
  PyObject* pWidth = PyFloat_FromDouble((double)m_lineWidth);
  if ( ! pWidth || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set veclen
  PyObject* pVlenV = PyLong_FromLong(m_veclen_v);
  if ( ! pVlenV || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set vecid
  npy_intp vvid[2] = {m_vecid_v[0], m_vecid_v[1]};
  PyObject *pVidV = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(vvid));
  if ( ! pVidV || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set vectorMag
  PyObject* pMag = PyFloat_FromDouble((double)m_vectorMag);
  if ( ! pMag || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set vectorHeadRatio
  PyObject* pRatio =
      PyArray_SimpleNewFromData(1, dims2, NPY_REAL,
                                reinterpret_cast<void*>(m_vectorHeadRatio));
  if ( ! pRatio || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set color list
  PyObject* pClrList;
  sz = m_arraySz[0] * m_arraySz[1];
  Real clist[sz][4];
  Real clist_1[1][4];
  if ( colid == -2 ) {
    clist_1[0][0] = 1.0;
    clist_1[0][1] = 1.0;
    clist_1[0][2] = 1.0;
    clist_1[0][3] = 1.0;
    npy_intp clr_dims0[2] = {1, 4};
    pClrList =
        PyArray_SimpleNewFromData(2, clr_dims0, NPY_REAL, (void*)clist_1);
  } else {
    if ( lut.colorList.size() == 1 ) {
      map<float, color_s>::iterator citr = lut.colorList.begin();
      clist_1[0][0] = (*citr).second.red;
      clist_1[0][1] = (*citr).second.green;
      clist_1[0][2] = (*citr).second.blue;
      clist_1[0][3] = 1.0;
      npy_intp clr_dims0[2] = {1, 4};
      pClrList =
          PyArray_SimpleNewFromData(2, clr_dims0, NPY_REAL, (void*)clist_1);
    } else {
      sz = m_arraySz[0] * m_arraySz[1];
      Real vmax = 0.0;
      Real v[sz];
      if ( colid > -1 ) {
        for ( int i = 0; i < sz; i++ ) {
          v[i] = data[i*m_veclen_v+colid];
          if ( v[i] > vmax ) vmax = v[i];
        }
      } else {
        for ( int i = 0; i < sz; i++ ) {
          Real v1 = data[i*m_veclen_v+m_vecid_v[0]];
          Real v2 = data[i*m_veclen_v+m_vecid_v[1]];
          complex<double> c((double)v1, (double)v2);
          v[i] = (Real)abs(c);
          if ( v[i] > vmax ) vmax = v[i];
        }
      }
      for ( int i = 0; i < sz; i++ ) {
        color_s clr = lut.ColorByValue(v[i]);
        clist[i][0] = clr.red;
        clist[i][1] = clr.green;
        clist[i][2] = clr.blue;
        clist[i][3] = 1.0;
      }
      npy_intp clr_dims0[2] = {sz, 4};
      pClrList =
          PyArray_SimpleNewFromData(2, clr_dims0, NPY_REAL, (void*)clist);
    }
  }
  if ( ! pClrList || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set Color
  sz = lut.colorList.size();
  Real vals[sz];
  Real clrs[sz][3];
  int cnt = 0;
  map<float, color_s>::iterator itr = lut.colorList.begin();
  while( itr != lut.colorList.end() ) {
    vals[cnt] = (*itr).first;
    clrs[cnt][0] = (*itr).second.red;
    clrs[cnt][1] = (*itr).second.green;
    clrs[cnt][2] = (*itr).second.blue;
    cnt++;
    ++itr;
  }
  npy_intp clr_dims1[1] = {sz};
  npy_intp clr_dims2[2] = {sz, 3};
  PyObject* pClrPos =
      PyArray_SimpleNewFromData(1, clr_dims1, NPY_REAL, (void*)vals);
  if ( ! pClrPos || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }
  PyObject* pClr =
      PyArray_SimpleNewFromData(2, clr_dims2, NPY_REAL, (void*)clrs);
  if ( ! pClr || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // call python function
  PyObject* pRet;
  if ( ret ) {
    pRet = PyObject_CallFunctionObjArgs(pFuncDrawV,
                   pId, pImgSz, pVP, pArrSz, pCoord, pVlen, pVid,
                   pVal, pVlenV, pVidV, pLut, pBGClr, pShow, pWidth,
                   pMag, pRatio, pClrList, pClrPos, pClr, NULL);
    if ( ! pRet )
      ret = false;
    if ( pRet == Py_False )
      ret = false;
    if ( PyErr_Occurred() && s_debugprint )
      PyErr_Print();
  }

  // decref python object
  if ( pId ) Py_DECREF(pId);
  if ( pImgSz ) Py_DECREF(pImgSz);
  if ( pVP ) Py_DECREF(pVP);
  if ( pArrSz ) Py_DECREF(pArrSz);
  if ( pCoord ) Py_DECREF(pCoord);
  if ( pVlen ) Py_DECREF(pVlen);
  if ( pVid ) Py_DECREF(pVid);
  if ( pVal ) Py_DECREF(pVal);
  if ( pVlenV ) Py_DECREF(pVlenV);
  if ( pVidV ) Py_DECREF(pVidV);
  if ( pLut ) Py_DECREF(pLut);
  if ( pBGClr ) Py_DECREF(pBGClr);
  if ( pShow ) Py_DECREF(pShow);
  if ( pWidth ) Py_DECREF(pWidth);
  if ( pMag ) Py_DECREF(pMag);
  if ( pRatio ) Py_DECREF(pRatio);
  if ( pClrList ) Py_DECREF(pClrList);
  if ( pClrPos ) Py_DECREF(pClrPos);
  if ( pClr ) Py_DECREF(pClr);
  if ( pRet ) Py_DECREF(pRet);

  return ret;
}

bool Pi2D::Output(const int step, const int row, const int col,
                const int proc)
{
  PyObject* pRet;

  bool ret = true;

  // set ID
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // draw colorbar for each registered LUT
  set<string>::iterator itr = m_registLut.begin();
  for ( ; itr != m_registLut.end(); itr++ ) {
    string lutname = *itr;
    LUT lut = m_lutList[lutname];

    // set lut name
    PyObject* pLut;
#if PY_MAJOR_VERSION >= 3
    pLut = PyUnicode_FromString(lutname.c_str());
#else
    pLut = PyString_FromString(lutname.c_str());
#endif
    if ( ! pLut || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }

    // set Color
    int sz = lut.colorList.size();
    Real vals[sz];
    Real clrs[sz][3];
    int cnt = 0;
    map<float, color_s>::iterator itr = lut.colorList.begin();
    while( itr != lut.colorList.end() ) {
      vals[cnt] = (*itr).first;
      clrs[cnt][0] = (*itr).second.red;
      clrs[cnt][1] = (*itr).second.green;
      clrs[cnt][2] = (*itr).second.blue;
      cnt++;
      ++itr;
    }
    npy_intp clr_dims1[1] = {sz};
    npy_intp clr_dims2[2] = {sz, 3};
    PyObject* pClrPos =
        PyArray_SimpleNewFromData(1, clr_dims1, NPY_REAL, (void*)vals);
    if ( ! pClrPos || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }
    PyObject* pClr =
        PyArray_SimpleNewFromData(2, clr_dims2, NPY_REAL, (void*)clrs);
    if ( ! pClr || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }

    long int dims2[1] = {2};
    long int dims3[1] = {3};

    // set cbSize
    PyObject* pCbSz =
        PyArray_SimpleNewFromData(1, dims2, NPY_REAL,
                                  reinterpret_cast<void*>(lut.cbSize));
    if ( ! pCbSz || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }
    // set cbPos
    PyObject* pCbPos =
        PyArray_SimpleNewFromData(1, dims2, NPY_REAL,
                                  reinterpret_cast<void*>(lut.cbPos));
    if ( ! pCbPos || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }
    // set cbHoriz
    PyObject* pCbHrz;
    if ( lut.cbHoriz )
      pCbHrz = Py_True;
    else
      pCbHrz = Py_False;
    // set cbNumTic
    PyObject* pCbTic = PyLong_FromSize_t(lut.cbNumTic);
    if ( ! pCbTic || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }
    // set cbTicColor
    PyObject* pCbTicClr =
        PyArray_SimpleNewFromData(1, dims3, NPY_REAL,
                                  reinterpret_cast<void*>(lut.cbTicColor));
    if ( ! pCbTicClr || PyErr_Occurred() ) {
      if ( s_debugprint ) PyErr_Print();
      ret = false;
    }

    // call python function
    if ( ret ) {
      pRet = PyObject_CallFunctionObjArgs(pFuncDrawCB,
                     pId, pLut, pClrPos, pClr, pCbSz, pCbPos,
                     pCbHrz, pCbTic, pCbTicClr, NULL);
      if ( ! pRet )
        ret = false;
      if ( pRet == Py_False )
        ret = false;
      if ( PyErr_Occurred() && s_debugprint )
	PyErr_Print();
    }

    // decref python object
    if ( pLut ) Py_DECREF(pLut);
    if ( pClrPos ) Py_DECREF(pClrPos);
    if ( pClr ) Py_DECREF(pClr);
    if ( pCbSz ) Py_DECREF(pCbSz);
    if ( pCbPos ) Py_DECREF(pCbPos);
    if ( pCbHrz ) Py_DECREF(pCbHrz);
    if ( pCbTic ) Py_DECREF(pCbTic);
    if ( pCbTicClr ) Py_DECREF(pCbTicClr);
    if ( pRet ) Py_DECREF(pRet);

    if ( ! ret ) {
      if ( pId ) Py_DECREF(pId);
      return false;
    }
  }

  // set outputPtn
  PyObject* pOutName;
#if PY_MAJOR_VERSION >= 3
  pOutName = PyUnicode_FromString(m_outputPtn.c_str());
#else
  pOutName = PyString_FromString(m_outputPtn.c_str());
#endif
  if ( ! pOutName || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set step
  PyObject* pStep = PyLong_FromLong(step);
  if ( ! pStep || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set row
  PyObject* pRow = PyLong_FromLong(row);
  if ( ! pRow || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set col
  PyObject* pCol = PyLong_FromLong(col);
  if ( ! pCol || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // set proc
  PyObject* pProc = PyLong_FromLong(proc);
  if ( ! pProc || PyErr_Occurred() ) {
    if ( s_debugprint ) PyErr_Print();
    ret = false;
  }

  // call python function
  if ( ret ) {
    pRet = PyObject_CallFunctionObjArgs(pFuncOut, pId, pOutName,
					pStep, pRow, pCol, pProc, NULL);
    if ( ! pRet )
      ret = false;
    if ( pRet == Py_False )
      ret = false;
    if ( PyErr_Occurred() && s_debugprint )
      PyErr_Print();
  }

  // decref PyObject
  if ( pId ) Py_DECREF(pId);
  if ( pOutName ) Py_DECREF(pOutName);
  if ( pStep ) Py_DECREF(pStep);
  if ( pRow ) Py_DECREF(pRow);
  if ( pCol ) Py_DECREF(pCol);
  if ( pProc ) Py_DECREF(pProc);
  if ( pRet ) Py_DECREF(pRet);

  // clear set of LUT name;
  m_registLut.clear();

  return ret;
}
