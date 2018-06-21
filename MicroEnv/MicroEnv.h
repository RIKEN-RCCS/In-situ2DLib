// MicroEnv
#ifndef _MICRO_ENV_H_
#define _MICRO_ENV_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <Python.h>

#if PY_MAJOR_VERSION < 3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include "numpy/arrayobject.h"

class MicroEnv {
public:
  ~MicroEnv();

  static MicroEnv* GetInstance();

  bool initialize(const std::string& dconfpath=std::string());
  bool execute(const std::string& pypath);
  void finalize();

private:
  MicroEnv();

  bool m_initialized;
  PyObject* p_data_dict;

  static MicroEnv* s_instance;
  static bool s_debugprint;
};

#endif // _MICRO_ENV_H_
