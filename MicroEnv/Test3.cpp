#include "MicroEnv.h"

static double* A = NULL;
static int nA = 0;

static PyObject* getA(PyObject *self, PyObject *args) {
  if ( ! A || nA < 1 ) return NULL;
  
  PyObject* lstA = PyList_New(nA);  
  for ( int i = 0; i < nA; i++ ) {
    PyList_SET_ITEM(lstA, i, PyFloat_FromDouble(A[i]));  
  }
  return Py_BuildValue("O", lstA);
}

static PyObject* setA(PyObject *self, PyObject *args) {
  PyArrayObject *array;
  if ( ! PyArg_ParseTuple(args, "O", &array) ) return NULL;
  printf("***\n");
  printf("NDIM=%d\n", PyArray_NDIM(array));
  npy_intp* dims = PyArray_DIMS(array);
  printf("DIMS[0]=%d\n", dims[0]);
  npy_intp* shape = PyArray_SHAPE(array);
  printf("SHAPE[0]=%d\n", shape[0]);
  npy_intp itemsz = PyArray_ITEMSIZE(array);
  printf("itemsz=%d\n", itemsz);
  npy_intp* strides = PyArray_STRIDES(array);
  printf("strides=%d\n", strides[0]);
  double* dp = (double*)PyArray_DATA(array);
  for ( int i = 0; i < nA; i++ ) {
    A[i] = dp[i];
  }
  Py_RETURN_NONE;
}

static PyMethodDef embMethods[] = {
  {"getA", getA, METH_VARARGS, NULL},
  {"setA", setA, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef embModule = {
  PyModuleDef_HEAD_INIT,
  "emb", NULL, -1, embMethods,
  NULL, NULL, NULL, NULL
};
static PyObject* PyInit_emb(void) {
  return PyModule_Create(&embModule);
}
#endif


int main(int argc, char** argv) {
#if PY_MAJOR_VERSION >= 3
  PyImport_AppendInittab("emb", &PyInit_emb); // must be before Py_Initialize
#endif

  MicroEnv* me = MicroEnv::GetInstance();
  me->initialize(); // Py_Initialize will be called

#if PY_MAJOR_VERSION < 3
  Py_InitModule("emb", embMethods); // must be after Py_Initialize
#endif

  if ( PyErr_Occurred() ) {
    PyErr_Print();
  }

  // preapre data
  nA = 10;
  A = new double[nA];
  for ( int i = 0; i < nA; i++ )
    A[i] = (double)i;

  // exec Z.py
  me->execute(std::string("Z"));

  for ( int i = 0; i < nA; i++ )
    printf("A[%d] = %g\n", i, A[i]);

  me->finalize();
  delete me;
  return 0;
}
