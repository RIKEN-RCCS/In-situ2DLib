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
  
  PyObject* iter = PyObject_GetIter(arr);
  if ( ! iter ) Py_RETURN_NONE;
  PyObject* item;
  int* pi = (int*)di.p;
  float* pf = (float*)di.p;
  double* pd = (double*)di.p;
  switch ( di.dtype ) {
  case NPY_INT:
    while ( item = PyIter_Next(iter) ) {
      *pi++ = (int)PyLong_AsLong(item);
      Py_DECREF(item);
    } // end of while
    break;
  case NPY_FLOAT:
    while ( item = PyIter_Next(iter) ) {
      *pf++ = (float)PyFloat_AsDouble(item);
      Py_DECREF(item);
    } // end of while
    break;
  case NPY_DOUBLE:
    while ( item = PyIter_Next(iter) ) {
      *pd = PyFloat_AsDouble(item);
      if ( PyErr_Occurred() ) {
	PyErr_Print();
      }
      Py_DECREF(item);
      pd++;
    } // end of while
    break;
  } // end of switch
  Py_DECREF(iter);
  
  Py_RETURN_NONE;
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
