#include <iostream> // std::cout
#include <fstream>  // std::ifstream
#include <cassert>  // std::assert
#include <cstdio>

#include "picojson.h"

using namespace std;

int main(int argc, char** argv) {
  if ( argc < 2 ) {
    printf("usage: %s file.json\n", argv[0]);
    return 1;
  }
  
  // ファイルを読み込むための変数
  std::ifstream fs;
  
  // ファイルを読み込む
  fs.open(argv[1]);
  
  // 読み込みチェック
  // fs変数にデータがなければエラー
  assert(fs);

  // Picojsonへ読み込む
  picojson::value val;
  fs >> val;
  
  // fs変数はもう使わないので閉鎖
  fs.close();
  
  // Playerの名前を取得
  picojson::object o, o2;
  string nm;
  try {
    o = val.get<picojson::object>();
  } catch (const exception& e) {
    printf("can not get TOP object\n");
    return 2;
  }
  try {
    o2 = o["Player"].get<picojson::object>();
  } catch (const exception& e) {
    printf("can not get Player object\n");
    return 2;
  }
  try {
    nm = o2["Name"].get<string>();
  } catch ( const exception& e) {
    printf("can not get Player/Name object\n");
    return 2;
  }
  printf("Player/Name = %s\n", nm.c_str());

  // PlayerのITEMを取得
  picojson::array arr;
  try {
    arr = o2["ITEM"].get<picojson::array>();
  } catch ( const exception& e) {
    printf("can not get Player/Name object\n");
    return 2;
  }
  for (int i = 0; i < arr.size(); i++)
    printf("ITEM[%d] = %s\n", i, arr[i].get<string>().c_str());

  // イテレータテスト
  picojson::object::iterator it;
  for (it = o2.begin(); it != o2.end(); it++) {
    printf("key=%s\n", it->first.c_str());
  }
  
  return 0;
}

