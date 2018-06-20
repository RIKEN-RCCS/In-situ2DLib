// MicroEnv
#include "MicroEnv.h"
#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION >= 3
int init_numpy() {import_array();}
#else
void init_numpy() {import_array();}
#endif


// STATIC
MicroEnv* MicroEnv::s_instance = NULL;
bool MicroEnv::s_debugprint = true;

// STATIC
MicroEnv* MicroEnv::GetInstance() {
  if ( ! s_instance ) {
    s_instance = new MicroEnv();
  }
  return s_instance;
}

// CONSTRUCTOR / DESTRUCTOR
MicroEnv::MicroEnv()
  : m_initialized(false), p_data_dict(NULL) {
}

MicroEnv::~MicroEnv() {
  finalize();
}

// METHODS
bool MicroEnv::initialize() {
  if ( m_initialized ) return true; // already initialized

  // initialize
  Py_Initialize();
  init_numpy();
  if ( s_debugprint && PyErr_Occurred() ) {
    PyErr_Print();
  }

  // run initial python code
  std::string cmd = "import matplotlib\n";
  cmd += "matplotlib.use('Agg')\n";
  cmd += "import sys; sys.path.append('.')\n";
  PyRun_SimpleString(cmd.c_str());
  if ( s_debugprint && PyErr_Occurred() ) {
    PyErr_Print();
  }

  // load ME.py and get data_dict
  PyObject* pModule = PyImport_ImportModule("ME");
  if ( ! pModule ) return false;
  p_data_dict = PyObject_GetAttrString(pModule, "data_dict");
  Py_DECREF(pModule);
  if ( ! p_data_dict ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    return false;
  }
  
  m_initialized = true;
  return true;
}

bool MicroEnv::execute(const std::string& pypath) {
  std::string xpath = pypath;
  size_t pathlen = xpath.length();
  if ( pathlen < 1 ) return false;
  if ( pathlen > 3 && xpath.substr(pathlen-3, 3) == std::string(".py") )
    xpath = xpath.substr(0, pathlen-3);
  
  PyObject* pModule = PyImport_ImportModule(xpath.c_str());
  if ( ! pModule ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    return false;
  }
  
  PyObject* pFunc = PyObject_GetAttrString(pModule, "FUNC");
  Py_DECREF(pModule);
  if ( ! pFunc ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    return false;
  }

  PyObject *pRet = PyObject_CallFunctionObjArgs(pFunc, p_data_dict, NULL);
  Py_DECREF(pFunc);
  if ( pRet ) {
    long ret = PyLong_AsLong(pRet);
    Py_DECREF(pRet);
    if ( ret != 0 ) {
      if ( s_debugprint && PyErr_Occurred() ) {
	PyErr_Print();
      }
      return false;
    }
    PyObject* pa = PyDict_GetItemString(p_data_dict, "a");
    if ( ! pa ) {
      if ( s_debugprint && PyErr_Occurred() ) {
	PyErr_Print();
      }
      return false;
    }

    PyArrayObject* pnarr = reinterpret_cast<PyArrayObject*>(pa);
    npy_intp* na_shape = PyArray_SHAPE(pnarr);
    long double* darr = reinterpret_cast<long double*>(PyArray_DATA(pa));
    for ( long l = 0; l < na_shape[0]*na_shape[1]; l++ ) {
      printf("%lg ", darr[l]);
    }
    printf("\n");
    
    return true;
  }
  if ( s_debugprint && PyErr_Occurred() ) {
    PyErr_Print();
  }
  return false;
}

void MicroEnv::finalize() {
  if ( ! m_initialized ) return;
  if ( p_data_dict ) {
    Py_DECREF(p_data_dict);
    p_data_dict = NULL;
  }
  Py_Finalize();
  m_initialized = false;
}
