// MicroEnv
#ifndef _MICRO_ENV_H_
#define _MICRO_ENV_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include <Python.h>

#if PY_MAJOR_VERSION < 3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include "numpy/arrayobject.h"

class MicroEnv {
public:
  ~MicroEnv();

  static MicroEnv* GetInstance();

  bool initialize();
  bool execute(const std::string& pypath);
  void finalize();

  struct DataInfo {
    std::string name;
    NPY_TYPES dtype;
    int nd;
    npy_intp dims[8];
    void* p;
    DataInfo() : dtype(NPY_INT), nd(1), p(NULL) {
      dims[0] = dims[1] = dims[2] = dims[3]
	= dims[4] = dims[5] = dims[6] = dims[7] = 0;
    }
    void operator=(const DataInfo& x) {
      name = x.name; dtype = x.dtype; nd = x.nd; p = x.p;
      for ( int i = 0; i < 8; i++ ) dims[i] = x.dims[i];
    }
  };

  std::map<std::string, DataInfo>& Dmap() {return m_dmap;}
  bool registDmap(DataInfo& di);

private:
  MicroEnv();

  bool m_initialized;
  std::map<std::string, DataInfo> m_dmap;

  static MicroEnv* s_instance;
  static bool s_debugprint;
};

#endif // _MICRO_ENV_H_
