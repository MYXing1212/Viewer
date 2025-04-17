#define _AFXDLL
#pragma once
#include<iostream>
#include<stdio.h>
#include<string>
#include<stdlib.h>
#include<vector>
#include<opencv2/opencv.hpp>
#include <afx.h>  
#include <afxdlgs.h>  
#include <iostream> 
#include<fstream>
#include<io.h>
#include"MyString.h"
#include<sstream>
#include<io.h>
#include <direct.h>  
//#include<zlib.h>
//#include"huffman.h"
//#include<assimp\cimport.h>
//#include<assimp/scene.h>
//#include<assimp\postprocess.h>
//#include<assimp/Importer.hpp>

#define NO_WIN32_LEAN_AND_MEAN // 从 Windows 头中排除极少使用的资料
#include <shlobj.h>

using namespace std;

// 【文件后缀】返回文件后缀
string getFileSuffix(string filename);

// 【文件后缀】返回文件名称 返回文件名 hello.txt 则返回hello
string getFileName(string filename);

// 【文件后缀】返回文件后缀
CString getFileSuffix(CString fileName);

// 【文件名】返回文件名 hello.txt 则返回hello
CString getFileName(CString fileName);

// 【查询文件】获取某一路径下的所有后缀为suffix的文件
CStringArray* findAllSpecFiles(CString path, CString suffix);

// 【查询文件】获取某一路径下的所有txt文件
CStringArray* findAllTxt(CString path);


// 【查询文件】获取某一路径下的所有文件夹
CStringArray* findAllFolder(CString path);

// 保存cv::Mat到xml文件  filepath必须以.xml结尾 文件名不能为纯数字
bool saveMat(string filepath, cv::Mat data);

// 【查询文件】 获取某一路径下的所有文件并保存至txt， saveFolder默认为true，该路径下的文件夹也存储进txt
bool saveAllFilePaths(string FolderPath, string savepath, bool saveFolder = false);

//  【读文件】每行作为一个字符串存入
std::vector<string> readTextInLines(std::string filepath);

// 【读文件】每行作为一个字符串存入
bool readStringList(const string& filename, vector<string>& l);

// 【存文件】 保存yml文件
bool getSavePath(string& filepath);

// 【读文件】从txt中读入二维点集
void readIntPoints2D(CString filename, vector<cv::Point> &pts);

// 【读文件】从文件中读取下一个字段的数据，直到遇见下一个制表符（默认）或文件尾
// 首先清除string结果变量中以前的任何内容（这样可以在循环中更方便地使用函数，在循环中，同样的变量将被重复使用）。
// 然后保留一些空格，以避免频繁地重新定位。最后循环读取字符。
void get_chunk(istream& in, string& s, char terminator = '\t');

// 【读文件】从txt中读取二维点集 实数
bool readRealPoints2D(CString filename, double* Point2D, long PointCount);

// 【读文件】从txt中读取三维点集 实数
bool readRealPoints3D(CString filename, double* Point3D, long PointCount);

// 【写文件】写float*进入txt
void BmpSerialize(string filename, float *imgBuf, int dataSize);

// 【写文件】写float*进入txt
void writeFloatArrayInTxt(float* src, int len, std::string filepath, int offset = 0);

// 【写文件】写cv::Mat进入txt 
void writeMatInTxt(cv::Mat src, std::string filepath);

// 【读文件】读txtMat
void readMatInTxt(cv::Mat &data, std::string filepath);

// 【写文件】写vector<Point2f>进入txt
void writePts2DInTxt(vector<cv::Point2f> src, std::string filepath);

// 【写文件】写vector<Point3d>进入txt
void writePts3DInTxt(vector<cv::Point3d> src, std::string filepath);

// 【写文件】写vector<Point>进入txt
void writePts2DInTxt(vector<cv::Point> src, std::string filepath);

// 【文件夹】选择一个文件夹
CString selectFolder();

// 【文件夹】判断文件夹是否存在
bool isFolderValidate(CString folderpath);

// 【文件夹】判断文件夹是否存在
bool isFolderValidate(string folderpath);

// 【文件】判断文件是否存在
bool isFileExist(CString filepath);

// 【文件】判断文件是否存在
bool isFileExist(string filepath);

// 读取txt第一行并存入string
string readFstLineInTxt(string filepath);

// 【读取】读取矩阵
cv::Mat readMat(string filepath);

// 利用winWin32_FIND_DATA 读取文件夹下的文件名
void readImgNamefromFile(char *fileName, vector<string> &imgNames);

// 压缩一组文件到目标文件中
bool CompressFiles(CString dstFilePath, vector<CString> srcFilePaths);

//【Huffman压缩】	  Huffman压缩文件   文件路径文件夹后使用俩斜杠“\\”
bool HuffmanCompressFile(CString srcFilePath, CString dstFilePath);

// 【Huffman解压缩】   Huffman解压缩压缩文件 文件路径文件夹后使用俩斜杠"\\"
bool HuffmanDecompressFile(CString srcFilePath, CString dstFilePath);

// 【stl文件】导入stl文件
bool loadSTLFile(string filepath);

// 【文件重命名】
// 给folderPath文件夹中的所有文件，增加前缀prefix
// useOld 是否使用原来的名字，如果是，则改后的文件只是加前缀否则就重新命名 为前缀后面加数字编号
// subfix 使用后缀 若为空 则后缀不变，若不为空，则改变后缀为subfix
bool renameFilesInFolder(string folderPath, string prefix, bool useOld = true, string suffix = "");

// 【清空】清空文件
void clearFile(string filepath);

// 【文件夹】创建文件夹
void createFolder(string folder);

// 【文件夹】删除文件夹
void deleteFolder(string folder);

// 【读文件】读STL文件 返回值是三角面片的数量
bool ReadSTLFile(string cfilename, vector<float>& coorX, vector<float>& coorY, vector<float>& coorZ,
	vector<float>& normX, vector<float>& normY, vector<float>& normZ);

// 选择保存文件路径
string getSavePath(string dlgName, string suffix);

// 【剪贴板】复制文件到剪贴板
// 路径中的斜杠用反斜杠\ 而且得是俩杠\\ ，比如 f:\\3.bmp
int CopyFileToClipboard(string filename);

// 通过对话框选择保存路径 保存文件对话框
// filetype 是文件类型 比如 图片 文本 视频 音频等
// suffix 文件的后缀
// defaultName 是文件的默认名称
std::string selectSavePath(std::string filetype, std::string suffix, std::string defaultName = "output");

// 保存Mat到二值文件
cv::Mat readMatInBinaryFloat(std::string filename);

size_t getFileSize(const std::string &file_name);

// 保存Mat到二值文件
void saveMatInBinaryFloat(const cv::Mat& mat, std::string filename);