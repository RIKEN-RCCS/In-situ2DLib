// MicroEnv
#ifndef _MICRO_ENV_H_
#define _MICRO_ENV_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <Python.h>

class MicroEnv {
public:
  ~MicroEnv();

  static MicroEnv* GetInstance();

  bool initialize();
  bool execute(const std::string& pypath);
  void finalize();

private:
  MicroEnv();

  bool m_initialized;
  PyObject* p_data_dict;

  static MicroEnv* s_instance;
};

#endif // _MICRO_ENV_H_
