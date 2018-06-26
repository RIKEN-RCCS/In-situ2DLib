# MicroEnv検討

## 概要
---
MicroEnvは、九州大学情報基盤研究開発センターの小野先生が構想する
InSitu/InTransit用のフレームワークの機能で、データ分析、可視化、
シミュレーションの制御を、小さなPythonコードで行うことを目的と
するものである。

MicroEnvを使用するシミュレーションコードでは、以下のような使用方法が
イメージされている。

```
int main () {
  MicroEnv* me;
  me->getInstance();
  me->initialize();

  for (int step=0; step<MaxTimeStep; step++) {
    me->execute(“target.py”);
  }
  me->finalize();
}
```

ここでは、Pythonコード`target.py`にC++側から配列ポインターなどを渡して
データ処理などを実行する。
`target.py`はタイムステップループ中で、必要に応じて動的に変更していくことが
想定されている。

## 機能要件
---
MicroEnvの機能実現に必要な要件として、以下のものが考えられる。

### シングルトン

MicroEnvはC++のクラスとして実装されるが、プログラム全体を通して１個の
インスタンのみが存在することを保証することで、プログラムのどの位置からでも
インスタンスを取得し、同一のMicroEnvに対する参照・操作を行うことが可能となる。

そのために、MicroEnvはシングルトンパターンで実装することが求められる。

### Pythonコードでのデータ参照

MicroEnv内で実行されるPythonコードでは、シミュレーションプログラム内のデータ
(配列)やパラメータ変数を参照できる機能が求められる。

C/C++プログラムからPythonコードにデータを渡す場合、呼び出すPython関数
(またはクラスメソッド)に引数として渡す方法と、呼び出すPython関数が含まれる
実行フレーム(モジュール)にデータを登録しておく方法が考えられる。

いずれの場合も、MicroEnvとPythonコード側の両方で、参照するデータについての
共通の認知が必要である。これは、参照するデータについてのリストを事前に
MicroEnvに登録(登録用API、またはJSON等の設定ファイルで)しておくことで
実現可能と考えられる。

### ファイルの更新チェック

シミュレーションプログラムの制御(ステアリング)機能自体は、パラメータが記述された
設定ファイルの更新チェックと、更新時の再読み込み・パラメータ反映によって
行われる。

この機能は、POSIXの`stat()`システムコール(Windowsでは`_stat()`関数)で
実現することができる。以下にサンプルコードを示す。

```
#include <sys/stat.h>
#include <time.h>
#include <cstdio>
#include <unistd.h>
int main(int argc, char** argv) {
  const char* fpath = "/tmp/AAA";
  struct stat sbuf;
  time_t now = time(NULL);
  printf("sleep 5 sec, touch %s if you want.\n", fpath);
  sleep(5);
  if ( stat(fpath, &sbuf) ) {
    fprintf(stderr, "file not found: %s\n", fpath);
    return 1;
  }
  printf("mtime = %s\n", ctime(&sbuf.st_mtime));
  if ( now > sbuf.st_mtime )
    printf("%s not modified\n", fpath);
  else
    printf("%s modified\n", fpath);
  return 0;
}
```

プログラムを実行し、5秒間のsleep中に`"touch /tmp/AAA"`を実行すると
"/tmp/AAA modified"と出力される。

尚、設定ファイルの更新チェックと、更新時の再読み込み・パラメータ反映は
MicroEnvの機能の範囲外であるが、MicroEnvから呼び出されるPythonコードが
設定ファイルの更新を行うことがユースケースとして想定される。

### Pythonコードからのデータ更新

C++コード側(シミュレーションプログラム内またはMicroEnv内)に、Pythonコードから
データを設定するためのSetterインターフェースを用意しておく必要がある。

単純なパラメータの場合は、パラメータ記述設定ファイルの更新で実現でき、
また配列の場合は実装工数が大きくなることが予想され、かつメモリ効率の悪さも
想定されるため、実現の可否については要検討である。

### Pythonコード内でのMPI関数実行

プログラムの母体となるシミュレーションプログラムはMPI並列化されているため、
ここから呼び出されるPythonコードも各MPIプロセス毎にそれぞれ呼び出される
ことになる。
ここで、Pythonコード内で他のMPIプロセスとの通信を行う必要がある場合、
Pythonコード側で母体のMPIコンテキストを参照し、かつMPIのAPIをPythonから
利用できることが必要となる。

PythonのMPIモジュールとしてはmpi4pyが有名であるが、通常はPythonで
メインプログラムを記述することが想定されており、モジュールの初期化時に
`MPI_Init`の呼び出しが行われる。
ただし、以下のようなコードを記述することで、mpi4pyを使用したPythonコードを
embeddingすることは可能である。

```
#include <Python.h>
#include <mpi.h>

const char helloworld[] = \
  "import mpi4py \n"
  "mpi4py.rc.initialize = False \n"
  "mpi4py.rc.finalize = False \n"
  "from mpi4py import MPI \n"
  "hwmess = 'Hello, World! I am process %d of %d on %s.' \n"
  "myrank = MPI.COMM_WORLD.Get_rank() \n"
  "nprocs = MPI.COMM_WORLD.Get_size() \n"
  "procnm = MPI.Get_processor_name() \n"
  "print (hwmess % (myrank, nprocs, procnm)) \n";

int main(int argc, char *argv[]) {
  int i, n=5;
  MPI_Init(&argc, &argv);
  Py_Initialize();
  for (i = 0; i<n; i++ ) {
    PyRun_SimpleString(helloworld);
  }
  Py_Finalize();
  MPI_Finalize();
  return 0;
}
```

このプログラムは、以下のようにコンパイル・実行を行うことができる。

```
mpicxx -o MPI4PyTest MPI4PyTest.cpp `python-config --cflags --ldflags`
mpiexec -n 3 ./MPI4PyTest
```

## プロトタイプ実装
---
MicroEnvのプロトタイプ実装を行った。
このプロトタイプは、C++のMicroEnvクラスとして実装されている。

MicroEnvクラスはシングルトンパターンで実装されており、プログラム中で１つの
インスタンスしか作成できない。
ユーザープログラム内では、`MicroEnv::GetInstance()`メソッドで(唯一の)
インスタンスを取得し、このインスタンスを介してPythonコードの実行を行う。

```
MicroEnv* me = MicroEnv::GetInstance();
me->initialize();
me->execute(std::string("MyScript"));
me->finalize();
```

上記はユーザープログラム内を想定した擬似コードであり、
ユーザー作成のPythonスクリプト`MyScript.py`が実行される。

尚、MicroEnvのプロトタイプ実装ではPython3.3以降で実装されたAPIを
使用しているため、コンパイル・実行にはにはPython3環境が必要である。

### ユーザー作成のPythonスクリプト

MicroEnvは、`execute()`メソッドで指定されたPythonスクリプト中の関数`FUNC()`を
引数無しで呼び出し実行する。従ってユーザー作成のPythonスクリプト中には、
`FUNC()`という関数が定義されている必要がある。

ユーザー作成のPythonスクリプトでは、`MicroEnv`モジュールをimportし、
このモジュールを通してユーザープログラムとのデータのやり取りを行う。

```
import MicroEnv as me
def FUNC():
  rho = me.getArray('rho')
  rho = rho * 0.98
  ret = me.setArray('rho', rho)
  return 0
```

上記はユーザー作成のPythonスクリプトを想定した擬似コードで、
ユーザープログラム中の配列`rho`を取得し、全要素を0.98倍してユーザープログラムに
戻している。

この`rho`は、ユーザープログラム中でMicroEnvクラスの`registDmap()`メソッドで
登録しておく必要がある。

`registDmap()`メソッドでのデータ登録は、MicroEnvの内部クラス`DataInfo`を用いて
行う。

```
  struct DataInfo {
    std::string name;
    NPY_TYPES dtype;
    int nd;
    npy_intp dims[8];
    void* p;
  };
```

データ登録は以下のように行う。

```
MicroEnv* me = MicroEnv::GetInstance();
MicroEnv::DataInfo di;
di.name = "rho";
di.dtype = NPY_DOUBLE;
di.nd = 3;
di.dims[0] = 64; di.dims[1] = 32; di.dims[2] = 32;
di.p = pho;
```

---
- updated: 2018/06/26
- author: Yoshikawa, Hiroyuki, FUJITSU LIMITED
