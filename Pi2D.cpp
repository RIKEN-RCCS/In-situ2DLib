#include "Pi2D.h"
//#include <Python.h>
#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION >= 3
int init_numpy() {import_array();}
#else
void init_numpy() {import_array();}
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
    m_viewPoint[i] = 0.0;
  m_outputPtn = "./outimage_%S6.png";
  m_coord = NULL;
  //LUT lut;
  //m_lutList["default"] = lut;
  m_lutList.clear();
  m_lineWidth = 1.0;
  m_vectorMag = 1.0;
 
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
  }
  pClass = PyObject_GetAttrString(pModule, "Pi2D");
  if ( ! pClass || PyErr_Occurred() ) {
    PyErr_Print();
  }
  pFuncDrawS = PyObject_GetAttrString(pClass, "DrawS");
  if ( ! pFuncDrawS || PyErr_Occurred() ) {
    PyErr_Print();
  }
  pFuncDrawV = PyObject_GetAttrString(pClass, "DrawV");
  if ( ! pFuncDrawV || PyErr_Occurred() ) {
    PyErr_Print();
  }
  pFuncOut = PyObject_GetAttrString(pClass, "Output");
  if ( ! pFuncOut || PyErr_Occurred() ) {
    PyErr_Print();
  }
}

Pi2D::~Pi2D()
{
  //if ( m_id == 0 )
  //  Py_Finalize();
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
  Py_DECREF(pFuncDrawS);
  Py_DECREF(pFuncDrawV);
  Py_DECREF(pFuncOut);
  Py_DECREF(pClass);
  Py_DECREF(pModule);
}

bool Pi2D::SetAttribute(const string arg)
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
    int n = sscanf(vals.c_str(), "[%d, %d]", &w, &h);
    if ( n != 2 ) return false;
    if ( w < 0 || h < 0 ) return false;
    m_imageSz[0] = w;
    m_imageSz[1] = h;
  } else if ( attr == "arraySize" ) {
    int w, h;
    int n = sscanf(vals.c_str(), "[%d, %d]", &w, &h);
    if ( n != 2 ) return false;
    if ( w < 0 || h < 0 ) return false;
    m_arraySz[0] = w;
    m_arraySz[1] = h;
  } else if ( attr == "viewport" ) {
    Real x0, x1, y0, y1;
    int n = sscanf(vals.c_str(), "[%f, %f, %f, %f]", &x0, &x1, &y0, &y1);
    if ( n != 4 ) return false;
    m_viewPoint[0] = x0;
    m_viewPoint[1] = x1;
    m_viewPoint[2] = y0;
    m_viewPoint[3] = y1;
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
  } else {
    //printf("error: invalid attribute: %s\n", arg.c_str());
    return false;
  }

  return true;
}

bool Pi2D::SetCoord(const Real* arr, const int veclen,
                    const int* vecid)
{
  if ( ! vecid ) {
    m_vecid[0] = 0;
    m_vecid[1] = 1;
  }

  // set null
  if ( ! arr ) {
    free(m_coord);
    //m_coord = NULL;
    return true;
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

  int sz = m_arraySz[0] * m_arraySz[1] * veclen;
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
  string pystr;
  char pycmd[256];

  int n = m_arraySz[0] * m_arraySz[1];
  Real* zarr = (Real*)PyDataMem_NEW(sizeof(Real) * n);

  npy_intp img_dims[2] = {m_imageSz[0],  m_imageSz[1]};
  long int idims[1] = {2};
  PyObject *pImgSz = PyArray_SimpleNewFromData(1, idims, NPY_LONG,
                         reinterpret_cast<void*>(img_dims));

  for ( int i = 0 ; i < n ; i++ )
    zarr[i] = data[i];

  npy_intp z_dims[2] = {m_arraySz[0],  m_arraySz[1]};
  PyObject* pZarr =
      PyArray_SimpleNewFromData(2, z_dims, NPY_FLOAT,
                                reinterpret_cast<void*>(zarr));
  if ( ! pZarr || PyErr_Occurred() ) {
    PyErr_Print();
  }
  PyObject* pId = PyLong_FromSize_t(m_id);
  if ( ! pId || PyErr_Occurred() ) {
    PyErr_Print();
  }

  PyObject* pRet = PyObject_CallFunctionObjArgs(pClass, NULL);
  if ( ! pRet || PyErr_Occurred() ) {
    PyErr_Print();
  }
  pRet = PyObject_CallFunctionObjArgs(pFuncDrawS, pClass,
                       pId, pImgSz, pZarr, NULL);
  if ( ! pRet || PyErr_Occurred() ) {
    PyErr_Print();
  }

  PyDataMem_FREE(zarr);
  Py_DECREF(pImgSz);
  Py_DECREF(pZarr);
  Py_DECREF(pId);
  //Py_DECREF(pRet);

  return true;
}

bool Pi2D::DrawV(const Real* data, const int veclen,
                 const int* vecid, const string lutname,
                 const int colid, bool cbShow)
{
  if ( ! vecid ) {
    m_vecid_v[0] = 0;
    m_vecid_v[1] = 1;
  }

  return true;
}

bool Pi2D::Save(const int step, const int row, const int col,
                const int proc)
{
  PyObject* pOutName;
  //string path = m_outputPtn.replace();

#if PY_MAJOR_VERSION >= 3
  pOutName = PyUnicode_FromString(m_outputPtn.c_str());
#else
  pOutName = PyString_FromString(m_outputPtn.c_str());
#endif
  if ( ! pOutName || PyErr_Occurred() ) {
    PyErr_Print();
  }
  PyObject* pRet = PyObject_CallFunctionObjArgs(pFuncOut, pClass, 
                       pOutName, NULL);
  if ( ! pRet || PyErr_Occurred() ) {
    PyErr_Print();
  }
  Py_DECREF(pOutName);

  return true;
}

bool Pi2D::ImportAttrib(const string path)
{
  return true;
}

bool Pi2D::ExportAttrib(const string path)
{
  return true;
}
