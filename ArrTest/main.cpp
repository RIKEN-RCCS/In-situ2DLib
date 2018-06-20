#include <cstdio>
#include <cstdlib>
#include <string>
#include <Python.h>
#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION >= 3
int init_numpy() {import_array();}
#else
void init_numpy() {import_array();}
#endif


int main(int argc, char** argv) {
  PyObject *pModule, *pFunc;

  Py_Initialize();
  init_numpy();

  std::string cmd = "import matplotlib\n";
  cmd += "matplotlib.use('Agg')\n";
  cmd += "import sys; sys.path.append('.')\n";
  PyRun_SimpleString(cmd.c_str());

  pModule = PyImport_ImportModule("DRAW");
  if ( ! pModule || PyErr_Occurred() ) {
    PyErr_Print();
  }
  pFunc = PyObject_GetAttrString(pModule, "draw");
  if ( ! pFunc || PyErr_Occurred() ) {
    PyErr_Print();
  }

  const int SIZE = 30;
  npy_intp dims[2] = {SIZE, SIZE};
  double* c_arr = new double[SIZE * SIZE];
  for (int i = 0; i < SIZE; i++){
    for (int j = 0; j < SIZE; j++){
      c_arr[SIZE*j + i] = i + j;}
  }
  PyObject *pArray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE,
					       reinterpret_cast<void*>(c_arr));
  long int xdims[1] = {2};
  PyObject *pDims = PyArray_SimpleNewFromData(1, xdims, NPY_LONG,
					      reinterpret_cast<void*>(dims));

  PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc, pDims, pArray, NULL);
  //PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
  if ( pReturn ) {
    printf("ret = %d\n", PyLong_AsLong(pReturn));
  } else {
    printf("ret = NULL\n");
  }

  Py_Finalize();
  return 0;
}
