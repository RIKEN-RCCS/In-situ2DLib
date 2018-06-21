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

// MicroEnv module methods
static PyObject* getData(PyObject *self, PyObject *args) {
  char* dname;
  if ( ! PyArg_ParseTuple(args, "s", &dname) ) return NULL;

  
  Py_RETURN_NONE;
}

static PyObject* setData(PyObject *self, PyObject *args) {
  Py_RETURN_NONE;
}

// MicroEnv module description
static PyMethodDef meMethods[] = {
  {"getData", getData, METH_VARARGS, NULL},
  {"setData", setData, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};


// METHODS
bool MicroEnv::initialize(const std::string& dconfpath) {
  if ( m_initialized ) return true; // already initialized

  // python initialize
  Py_Initialize();
  init_numpy();

  if ( s_debugprint && PyErr_Occurred() ) {
    PyErr_Print();
  }
  
  // run initial python code
  std::string cmd = "import sys; sys.path.append('.')\n";
  PyRun_SimpleString(cmd.c_str());
  if ( s_debugprint && PyErr_Occurred() ) {
    PyErr_Print();
  }

  // create MicroEnv module
  PyObject* pModule
    = PyImport_AddModuleObject(PyUnicode_FromString("MicroEnv"));
  if ( ! pModule ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    return false;
  }

  // add data_dict to MicroEnv module
  p_data_dict = PyDict_New();
  if ( ! p_data_dict ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    Py_DECREF(pModule);
    return false;
  }
  
  if ( PyModule_AddObject(pModule, "data_dict", p_data_dict) != 0 ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    Py_DECREF(pModule);
    return false;
  }

  // add methds to MicroEnv module
  if ( PyModule_AddFunctions(pModule, meMethods) != 0 ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    Py_DECREF(pModule);
    return false;
  }

  // process dconfpath
  
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

  PyObject *pRet = PyObject_CallFunctionObjArgs(pFunc, NULL);
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
