#ifndef _PI2D_H_
#define _PI2D_H_

#include <stdio.h>
#include <iostream>
#include <string>
#include <map>
#include "LUT.h"
#include <Python.h>

#include "Pi2Ddefs.h"


class Pi2D {
public:
  static size_t s_id;

  Pi2D();
  ~Pi2D();
  bool SetAttrib(const std::string);
  bool SetCoord(const Real*, const int veclen=2, const int* vecidx=NULL);
  bool SetLUT(const std::string, const LUT*);
  bool DrawS(const CVType, const Real*, const std::string lutname="",
             const int nlevels=10, bool cbShow=false);
  bool DrawV(const Real*, const int veclen=2, const int* vecidx=NULL, 
             const std::string lutname="", const int colidx=-1,
             bool cbShow=false);
  bool Output(const int step=0, const int row=0, const int col=0,
            const int proc=0);
  bool ImportAttrib(const std::string);
  bool ExportAttrib(const std::string);

  int m_imageSz[2];
  int m_arraySz[2];
  Real m_viewPort[4];
  std::string m_outputPtn;
  Real* m_coord;
  std::map<std::string, LUT> m_lutList;
  Real m_lineWidth;
  Real m_vectorMag;
  Real m_vectorHeadRatio[2];
  size_t m_id;

private:
  int m_veclen;
  int m_vecid[2];
  int m_veclen_v;
  int m_vecid_v[2];

  PyObject *pModule;
  PyObject *pClass;
  PyObject *pFuncDrawS;
  PyObject *pFuncDrawV;
  PyObject *pFuncOut;

};

# endif // _PI2D_H_
