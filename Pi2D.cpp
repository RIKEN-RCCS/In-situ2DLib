#include "Pi2D.h"
#include <Python.h>

using namespace std;

size_t Pi2D::s_id = 0;

Pi2D::Pi2D()
{
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
}

bool Pi2D::SetAttrib(const string arg)
{
  return true;
}

bool Pi2D::SetCoord(const float* arr, const int veclen, 
                    const int* vecid)
{
  return true;
}

bool Pi2D::SetLUT()
{
  return true;
}

bool Pi2D::DrawS(const CVType vt, const float* data,
                 const std::string lutname, const int nlevels)
{
  return true;
}

bool Pi2D::DrawV(const float* data, const int veclen, const int* vecid,
                 const std::string lutname, const int colid)
{
  return true;
}

bool Pi2D::Save(const int step, const int row, const int col)
{
  return true;
}
