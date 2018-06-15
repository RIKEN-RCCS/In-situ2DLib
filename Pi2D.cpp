#include "Pi2D.h"
#include <Python.h>

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
 
  m_arrSz[0] = 0;
  m_arrSz[1] = 0;

  if ( s_id == 0 ) {
    Py_Initialize();

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

    //string cmd;
    //PyRun_SimpleString(cmd.c_str());
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
  } else if ( attr == "arrSize" ) {
    int sz[2];
    int n = sscanf(vals.c_str(), "[%i, %i]", &sz[0], &sz[1]);
    if ( n != 1 ) return false;
    if ( sz[0] < 0 || sz[1] < 0) return false;
    m_arrSz[0] = sz[0];
    m_arrSz[1] = sz[1];
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
    //m_coord = NULL;
    return true;
  }

  if ( veclen < 2 )
    return false;
  if ( vecid[0] < 0  || vecid[0] >= veclen )
    return false;
  if ( vecid[1] < 0  || vecid[1] >= veclen )
    return false;

  int sz = m_arrSz[0] * m_arrSz[1] * veclen;
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
                 const string lutname, const int nlevels)
{
  return true;
}

bool Pi2D::DrawV(const Real* data, const int veclen, const int* vecid,
                 const string lutname, const int colid)
{
  return true;
}

bool Pi2D::Save(const int step, const int row, const int col)
{
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
