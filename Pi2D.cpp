#include "Pi2D.h"
//#include <Python.h>
#include "numpy/arrayobject.h"

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
  //LUT lut;
  //m_lutList["default"] = lut;
  m_lutList.clear();
  m_lineWidth = 1.0;
  m_vectorMag = 1.0;
  m_vectorHeadRatio[0] = -1.0;
  m_vectorHeadRatio[1] = -1.0;
 
  pModule = NULL;
  //pClass = NULL;
  pFuncDrawS = NULL;
  pFuncDrawV = NULL;
  pFuncOut = NULL;

  if ( s_id == 0 ) {
    setenv("PYTHONPATH", ".", 0);
    Py_Initialize();
    init_numpy();

    string cmd("#!/usr/bin/env python\n");
    cmd += "# -*- coding: utf-8 -*-\n";
    cmd += "from __future__ import print_function\n";
    PyRun_SimpleString(cmd.c_str());

    cmd = "import matplotlib\n";
    cmd += "matplotlib.use('Agg')\n";
    cmd += "import numpy as np\n";
    cmd += "import matplotlib.cm as cm\n";
    cmd += "import matplotlib.pyplot as plt\n";
    cmd += "from matplotlib.colors import LinearSegmentedColormap\n";
    PyRun_SimpleString(cmd.c_str());
  }

  m_id = s_id;
  s_id++;

  pModule = PyImport_ImportModule("Pi2D");
  if ( ! pModule || PyErr_Occurred() ) {
    PyErr_Print();
    return;
  }
/*
  pClass = PyObject_GetAttrString(pModule, "Pi2D");
  if ( ! pClass || PyErr_Occurred() ) {
    PyErr_Print();
    return;
  }
*/
  pFuncDrawS = PyObject_GetAttrString(pModule, "DrawS");
  if ( ! pFuncDrawS || PyErr_Occurred() ) {
    PyErr_Print();
    return;
  }
  pFuncDrawV = PyObject_GetAttrString(pModule, "DrawV");
  if ( ! pFuncDrawV || PyErr_Occurred() ) {
    PyErr_Print();
    return;
  }
  pFuncOut = PyObject_GetAttrString(pModule, "Output");
  if ( ! pFuncOut || PyErr_Occurred() ) {
    PyErr_Print();
    return;
  }

  // call python function
  PyObject* pRet;
  pRet = PyObject_CallFunctionObjArgs(pModule, NULL);
  if ( ! pRet || PyErr_Occurred() ) {
    PyErr_Print();
    return;
  }
}

Pi2D::~Pi2D()
{
  if ( m_coord )
    delete[] m_coord;
  //map<string, LUT>::iterator itr;
  //for ( itr = m_lutList.begin(); itr != m_lutList.end(); itr++ ) {
  //  LUT lut = (*itr).second;
    //if ( *lut )
      //delete &lut;
  //}
  m_lutList.clear();

  // python  decref
  if ( pFuncDrawS )
    Py_DECREF(pFuncDrawS);
  if ( pFuncDrawV )
    Py_DECREF(pFuncDrawV);
  if ( pFuncOut )
    Py_DECREF(pFuncOut);
  //if ( pClass )
  //  Py_DECREF(pClass);
  if ( pModule )
    Py_DECREF(pModule);
}

bool Pi2D::SetAttrib(const string arg)
{
  // check empty argument
  if ( arg.empty() ) {
    //printf("error: empty string\n");
    return false;
  }

  // separete into attribute and value
  size_t p = arg.find("=");
  if  ( p == string::npos ) {
    //printf("error: invalid attribute: %s\n", arg.c_str());
    return false;
  }
  string attr = arg.substr(0, p);
  string vals = arg.substr(p+1);
  //printf("key: %s, value: %s\n", attr.c_str(), vals.c_str());

  if ( attr == "imageSize" ) {
    int w, h;
    int n = sscanf(vals.c_str(), "%d, %d", &w, &h);
    if ( n != 2 ) return false;
    if ( w < 0 || h < 0 ) return false;
    m_imageSz[0] = w;
    m_imageSz[1] = h;
  } else if ( attr == "arraySize" ) {
    int w, h;
    int n = sscanf(vals.c_str(), "%d, %d", &w, &h);
    if ( n != 2 ) return false;
    if ( w < 0 || h < 0 ) return false;
    m_arraySz[0] = w;
    m_arraySz[1] = h;
  } else if ( attr == "viewport" ) {
    Real x0, x1, y0, y1;
    int n = sscanf(vals.c_str(), "%f, %f, %f, %f", &x0, &x1, &y0, &y1);
    if ( n != 4 ) return false;
    m_viewPort[0] = x0;
    m_viewPort[1] = x1;
    m_viewPort[2] = y0;
    m_viewPort[3] = y1;
  } else if ( attr == "outfilePat" ) {
    if ( vals.empty() ) return false;
    m_outputPtn = vals;
  } else if ( attr == "lineWidth" ) {
    Real w;
    int n = sscanf(vals.c_str(), "%f", &w);
    if ( n != 1 ) return false;
    if ( w < 0.0 ) return false;
    m_lineWidth = w;
  } else if ( attr == "vectorMag" ) {
    Real v;
    int n = sscanf(vals.c_str(), "%f", &v);
    if ( n != 1 ) return false;
    if ( v < 0.0 ) return false;
    m_vectorMag = v;
  } else if ( attr == "vectorHeadRatio" ) {
    Real r0, r1;
    int n = sscanf(vals.c_str(), "%f, %f", &r0, &r1);
    if ( n != 2 ) return false;
    if ( r0 != -1 && r0 < 0.0 ) return false;
    if ( r1 != -1 && r1 < 0.0 ) return false;
    m_vectorHeadRatio[0] = r0;
    m_vectorHeadRatio[1] = r1;
  } else {
    //printf("error: invalid attribute: %s\n", arg.c_str());
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

  // set ID
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set ImageSize
  npy_intp imgsz[2] = {m_imageSz[0], m_imageSz[1]};
  PyObject *pImgSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(imgsz));
  if ( ! pImgSz || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set ArraySize
  npy_intp arrsz[2] = {m_arraySz[1], m_arraySz[0]};
  PyObject *pArrSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(arrsz));
  if ( ! pArrSz || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set viewPort
  PyObject* pVP =
      PyArray_SimpleNewFromData(1, dims4, NPY_REAL,
                                reinterpret_cast<void*>(m_viewPort));
  if ( ! pVP || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set contour type
  PyObject* pCtype = NULL;
  if ( vt == ColorContour )
    pCtype = PyLong_FromLong(0);
  else if ( vt == ContourLine )
    pCtype = PyLong_FromLong(1);
  if ( ! pCtype || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set veclen
  PyObject* pVlen = PyLong_FromLong(m_veclen);
  if ( ! pVlen || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set vecid
  npy_intp vid[2] = {m_vecid[0],  m_vecid[1]};
  PyObject *pVid = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(vid));
  if ( ! pVid || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set coord
  PyObject* pCoord;
  if ( m_coord ) {
    sz = m_arraySz[0] * m_arraySz[1] * m_veclen;
    long int cdims[1] = {sz};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(m_coord));
  } else {
    long int cdims[1] = {1};
    Real null_coord[1] = {0.0};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(null_coord));
  }
  if ( ! pCoord || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set data
  npy_intp z_dims[2] = {m_arraySz[1], m_arraySz[0]};
  PyObject* pZarr =
      PyArray_SimpleNewFromData(2, z_dims, NPY_REAL, (void*)(data));
  if ( ! pZarr || PyErr_Occurred() ) {
    PyErr_Print();
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
    PyErr_Print();
    ret = false;
  }

  // set nlevels
  PyObject* pNlevel = PyLong_FromLong(nlevels);
  if ( ! pNlevel || PyErr_Occurred() ) {
    PyErr_Print();
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
    PyErr_Print();
    ret = false;
  }

  //printf("dbg A\n");
  LUT lut = m_lutList[lutname];
  //printf("dbg B\n");

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
    PyErr_Print();
    ret = false;
  }
  PyObject* pClr =
      PyArray_SimpleNewFromData(2, clr_dims2, NPY_REAL, (void*)clrs);
  if ( ! pClr || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set cbSize
  PyObject* pCbSz =
      PyArray_SimpleNewFromData(1, dims2, NPY_REAL,
                                reinterpret_cast<void*>(lut.cbSize));
  if ( ! pCbSz || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set cbPos
  PyObject* pCbPos =
      PyArray_SimpleNewFromData(1, dims2, NPY_REAL,
                                reinterpret_cast<void*>(lut.cbPos));
  if ( ! pCbPos || PyErr_Occurred() ) {
    PyErr_Print();
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
    PyErr_Print();
    ret = false;
  }

  // call python function
  PyObject* pRet;
  if ( ret ) {
    pRet = PyObject_CallFunctionObjArgs(pFuncDrawS,
                   pId, pImgSz, pVP, pArrSz, pCoord, pVlen, pVid,
                   pCtype, pZarr, pLut, pNlevel, pShow, pWidth,
                   pClrPos, pClr, pCbSz, pCbPos, pCbHrz, pCbTic,
                   NULL);
    if ( ! pRet || PyErr_Occurred() ) {
      PyErr_Print();
      ret = false;
    }
  }

//  PyDataMem_FREE(zarr);
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
  if ( pWidth ) Py_DECREF(pWidth);
  if ( pShow ) Py_DECREF(pShow);
  if ( pClrPos ) Py_DECREF(pClrPos);
  if ( pClr ) Py_DECREF(pClr);
  if ( pCbSz ) Py_DECREF(pCbSz);
  if ( pCbPos ) Py_DECREF(pCbPos);
  if ( pCbHrz ) Py_DECREF(pCbHrz);
  if ( pCbTic ) Py_DECREF(pCbTic);

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
  long int dims4[1] = {4};

  // set ID
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set ImageSize
  npy_intp imgsz[2] = {m_imageSz[0], m_imageSz[1]};
  PyObject *pImgSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(imgsz));
  if ( ! pImgSz || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set ArraySize
  npy_intp arrsz[2] = {m_arraySz[1], m_arraySz[0]};
  PyObject *pArrSz = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(arrsz));
  if ( ! pArrSz || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set viewPort
  PyObject* pVP =
      PyArray_SimpleNewFromData(1, dims4, NPY_REAL,
                                reinterpret_cast<void*>(m_viewPort));
  if ( ! pVP || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set veclen
  PyObject* pVlen = PyLong_FromLong(m_veclen);
  if ( ! pVlen || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set vecid
  npy_intp vid[2] = {m_vecid[0],  m_vecid[1]};
  PyObject *pVid = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(vid));
  if ( ! pVid || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set coord
  PyObject* pCoord;
  if ( m_coord ) {
    sz = m_arraySz[0] * m_arraySz[1] * m_veclen;
    long int cdims[1] = {sz};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(m_coord));
  } else {
    long int cdims[1] = {1};
    Real null_coord[1] = {0.0};
    pCoord =
        PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(null_coord));
  }
  if ( ! pCoord || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set data
  sz = m_arraySz[0] * m_arraySz[1] * m_veclen_v;
  long int cdims[1] = {sz};
  PyObject* pVal =
      PyArray_SimpleNewFromData(1, cdims, NPY_REAL, (void*)(data));
  if ( ! pVal || PyErr_Occurred() ) {
    PyErr_Print();
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
    PyErr_Print();
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
    PyErr_Print();
    ret = false;
  }

  // set veclen
  PyObject* pVlenV = PyLong_FromLong(m_veclen_v);
  if ( ! pVlenV || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set vecid
  npy_intp vvid[2] = {m_vecid_v[0], m_vecid_v[1]};
  PyObject *pVidV = PyArray_SimpleNewFromData(1, dims2, NPY_LONG,
                         reinterpret_cast<void*>(vvid));
  if ( ! pVidV || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set vectorMag
  PyObject* pMag = PyFloat_FromDouble((double)m_vectorMag);
  if ( ! pMag || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set vectorHeadRatio
  PyObject* pRatio =
      PyArray_SimpleNewFromData(1, dims2, NPY_REAL,
                                reinterpret_cast<void*>(m_vectorHeadRatio));
  if ( ! pRatio || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // call python function
  PyObject* pRet;
  if ( ret ) {
    pRet = PyObject_CallFunctionObjArgs(pFuncDrawV,
                   pId, pImgSz, pVP, pArrSz, pCoord, pVlen, pVid,
                   pVal, pVlenV, pVidV, pLut, pShow, pWidth,
                   pMag, pRatio, NULL);
    if ( ! pRet || PyErr_Occurred() ) {
      PyErr_Print();
      ret = false;
    }
  }

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
  if ( pWidth ) Py_DECREF(pWidth);
  if ( pMag ) Py_DECREF(pMag);
  if ( pRatio ) Py_DECREF(pRatio);

  return ret;
}

bool Pi2D::Output(const int step, const int row, const int col,
                const int proc)
{
  PyObject* pOutName;

  bool ret = true;

  /*
  if ( step < 0 )
    return false;
  if ( row < 0 )
    return false;
  if ( col < 0 )
    return false;
  if ( proc < 0 )
    return false;
  */

  // set ID
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set outputPtn
#if PY_MAJOR_VERSION >= 3
  pOutName = PyUnicode_FromString(m_outputPtn.c_str());
#else
  pOutName = PyString_FromString(m_outputPtn.c_str());
#endif
  if ( ! pOutName || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set step
  PyObject* pStep = PyLong_FromLong(step);
  if ( ! pStep || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set row
  PyObject* pRow = PyLong_FromLong(row);
  if ( ! pRow || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set col
  PyObject* pCol = PyLong_FromLong(col);
  if ( ! pCol || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // set proc
  PyObject* pProc = PyLong_FromLong(proc);
  if ( ! pProc || PyErr_Occurred() ) {
    PyErr_Print();
    ret = false;
  }

  // call python function
  if ( ret ) {
    PyObject* pRet = PyObject_CallFunctionObjArgs(pFuncOut,
                         pId, pOutName, pStep, pRow, pCol, pProc, NULL);
    if ( ! pRet || PyErr_Occurred() ) {
      PyErr_Print();
      ret = false;
    }
  }

  // decref PyObject
  if ( pId ) Py_DECREF(pId);
  if ( pOutName ) Py_DECREF(pOutName);
  if ( pStep ) Py_DECREF(pStep);
  if ( pRow ) Py_DECREF(pRow);
  if ( pCol ) Py_DECREF(pCol);
  if ( pProc ) Py_DECREF(pProc);

  return ret;
}
