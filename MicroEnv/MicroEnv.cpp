/**
 * @file   MicroEnv.cpp
 * @brief  Python micro environment for numerical simulation
 * @author Yoshikawa, Hiroyuki
 */
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
MicroEnv::MicroEnv() : m_initialized(false) {
}

MicroEnv::~MicroEnv() {
  finalize();
}

// MicroEnv module methods
static PyObject* getArray(PyObject *self, PyObject *args) {
  MicroEnv* me = MicroEnv::GetInstance();
  if ( ! me ) Py_RETURN_NONE;
  char* dname;
  if ( ! PyArg_ParseTuple(args, "z", &dname) ) Py_RETURN_NONE;

  std::map<std::string, MicroEnv::DataInfo>::iterator it
    = me->Dmap().find(dname);
  if ( it == me->Dmap().end() ) Py_RETURN_NONE;
  MicroEnv::DataInfo& di = it->second;
  
  return PyArray_SimpleNewFromData(di.nd, di.dims, di.dtype, di.p);
}

static PyObject* setArray(PyObject *self, PyObject *args) {
  MicroEnv* me = MicroEnv::GetInstance();
  if ( ! me ) Py_RETURN_NONE;
  char* dname;
  PyObject* arr;
  if ( ! PyArg_ParseTuple(args, "zO", &dname, &arr) ) Py_RETURN_NONE;

  std::map<std::string, MicroEnv::DataInfo>::iterator it
    = me->Dmap().find(dname);
  if ( it == me->Dmap().end() ) Py_RETURN_NONE;
  MicroEnv::DataInfo& di = it->second;

  int nd = PyArray_NDIM(arr);
  npy_intp* dims = PyArray_DIMS(arr);
  if ( nd < 1 || ! dims ) Py_RETURN_NONE;
  //npy_intp* shape = PyArray_SHAPE(arr);
  //npy_intp itemSz = PyArray_ITEMSIZE(arr);
  //npy_intp* strides = PyArray_STRIDES(arr);
  npy_intp dimSz = dims[0];
  for ( int i = 1; i < nd; i++ ) dimSz *= dims[i];
  if ( dimSz < 1 ) Py_RETURN_NONE;
  
  switch ( di.dtype ) {
  case NPY_INT: {
    int* pi = (int*)di.p;
    int* arr_pi = (int*)PyArray_DATA(arr);
    for ( int i = 0; i < dimSz; i++ ) {
      pi[i] = arr_pi[i];
    }
    break;
  }
  case NPY_FLOAT: {
    float* pf = (float*)di.p;
    float* arr_pf = (float*)PyArray_DATA(arr);
    for ( int i = 0; i < dimSz; i++ ) {
      pf[i] = arr_pf[i];
    }
    break;
  }
  case NPY_DOUBLE: {
    double* pd = (double*)di.p;
    double* arr_pd = (double*)PyArray_DATA(arr);
    for ( int i = 0; i < dimSz; i++ ) {
      pd[i] = arr_pd[i];
    }
    break;
  }
  default:
    break;
  } // end of switch  
  return Py_BuildValue("i", 1);
}

// MicroEnv module description
static PyMethodDef meMethods[] = {
  {"getArray", getArray, METH_VARARGS, NULL},
  {"setArray", setArray, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};


// METHODS
bool MicroEnv::initialize() {
  if ( m_initialized ) return true; // already initialized

  // python initialize
  Py_Initialize();
  init_numpy();
  if ( s_debugprint && PyErr_Occurred() ) {
    PyErr_Print();
  }
  
  // run initial python code
  std::string cmd = "import sys; sys.path.append('.')\n";
  cmd += "import mpi4py\n";
  cmd += "mpi4py.rc.initialize = False\n";
  cmd += "mpi4py.rc.finalize = False\n";
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
  PyObject* pDict = PyDict_New();
  if ( ! pDict ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    Py_DECREF(pModule);
    return false;
  }
  
  if ( PyModule_AddObject(pModule, "data_dict", pDict) != 0 ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    Py_DECREF(pDict);
    Py_DECREF(pModule);
    return false;
  }

  // add methds to MicroEnv module
  if ( PyModule_AddFunctions(pModule, meMethods) != 0 ) {
    if ( s_debugprint && PyErr_Occurred() ) {
      PyErr_Print();
    }
    Py_DECREF(pDict);
    Py_DECREF(pModule);
    return false;
  }

  // clear dmap
  m_dmap.clear();

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
  Py_Finalize();
  m_initialized = false;
}

bool MicroEnv::registDmap(MicroEnv::DataInfo& di) {
  if ( di.name.empty() ) return false;
  if ( di.nd < 1 ) return false;
  for ( int i = 0; i < di.nd; i++ )
    if ( di.dims[i] < 1 ) return false;
  if ( ! di.p ) return false;
  switch ( di.dtype ) {
  case NPY_INT: case NPY_FLOAT: case NPY_DOUBLE:
    break;
  default:
    return false;
  }
  m_dmap[di.name] = di;
  return true;
}
