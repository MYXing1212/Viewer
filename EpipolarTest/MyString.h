#define _AFXDLL
#pragma once
#include<math.h>
#include<stdio.h>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<afxwin.h>
#include<iostream>
#include<atlstr.h>
#include<algorithm>
#include<functional>
#include<stack>
#include<strsafe.h>
#include<crtdbg.h>
#include<atltime.h>
#include<sstream>
#include"MyVector.h"
#include<tchar.h>

#define COMPUTE_TYPE_COUNT 9

#define POW_INDEX 0
#define SIN_INDEX 1
#define COS_INDEX 2
#define TAN_INDEX 3
#define LOG_INDEX 4
#define ADD_INDEX 5
#define MNS_INDEX 6		// 减法
#define MUL_INDEX 7
#define DIV_INDEX 8

#define STRSAFE_NO_DEPRECATE

using namespace std;

#ifndef NATURAL_LOGARITHM
#define NATURAL_LOGARITHM 2.718281828459
#endif

#pragma once
// 字符串类
class MyString
{
public:
	MyString(void);
	~MyString(void);



	// 字符串转数字
	long str2long(string s);

	

	

	// 判断一个字符串是否为回文字符串
	bool ifHuiwenString(string a);

	// 分割字符串，分割结果存在result里面，分割的标志为flag
	void splitString(string input, string flag, vector<string>& result);

	// 分割字符串，获得第一段分割结果
	void getFirstSplit(string input, string flag, string& result);

	// 分割字符串，获得最后一段分割结果
	void getLastSplit(string input, string flag, string& result);

	

	

	// Unicode 转 多字节
	int ConvertUnicodeToAscii(const CString &csData, char* pszData);

	// 字符转换
	BOOL WChar2MByte(LPCWSTR srcBuff, LPSTR destBuff, int nlen);

	// 获取当前时间的字符串格式 2015/6/11 22:40:33
	CString getTimeCString();
};

// 将char*字符串转换为LPCTSTR字符串
wchar_t* Char2LPCTSTR(char * str);

//// string转LPCSTR
//wchar_t* string2LPCSTR(string str);

// const char* 转 char*
char* pConstC2pC(const char* ch);

// 将string转const char*字符串
const char* string2pChar(string str);

// 将CString字符串转换为const char*字符串
const char* CString2pChar(CString cstring);

// 将CString字符串转换为string字符串
string CString2string(CString cstring);

// 整形转字符
string int2string(int a);

// int转 CString
CString int2CString(int i);

// float转 CString
CString float2CString(float f);

// double转 CString
CString double2CString(double d);

// 浮点数转字符 accuracy = 1代表小数点后一位
string double2string(double a,int accuracy = 1);

// 将CString字符串转换为string字符串
string CString2string(CString cstring);

// 将CString字符串转换为char* 字符串
char* CString2pchar(CString cstring);

// 将string字符串转换为CString字符串
CString string2CString(string str);

// 将char* 转换为 LPCWSTR
LPCWSTR pchar2LPCWSTR(char* p);

// 将char*字符串转换为LPCTSTR字符串
LPCTSTR pchar2LPCTSTR(char * str);

// 将char*转wchar_t*
wchar_t* pchar2pwchar_t(char* str);

// char* 转 TCHAR*
TCHAR* pchar2pTCHAR(const char* str);

// 取CString最右的count个字符
CString RightCString(CString input, int count);

// CString从后往前找"XXX",所在的位置   2\1.txt , '\\' 结果为1
int b2fFind(CString src, char c);

// 取CString最左的count个字符
CString LeftCString(CString input, int count);

// 取CString从后往前找‘x’ 之前的字符
CString LStr_b2fFind(CString input, char c);

// 取CString从后往前找‘x’ 之后的字符
CString RStr_b2fFind(CString input, char c);

// 【打印字符串】打印CStingArray	需要用指针的形式
void printCStringArray(CStringArray* strArray);

// 【字符串】转换为double
double string2double(string str);

// 【字符串】转换为bool
bool string2bool(string str);

// CString 转换为 int
int CString2int(CString str);

// 【字符串】转换为int
int string2int(string str);

// 【路径】获取工作路径
CString getWorkDir();

// 【当前时间】
CString getCurrentTimeString(bool bAbbrev = false);

// 【当前日期】
CString getCurrentDateString();

// 删除所有字符a
bool removeAllSpecChar(string& str, char c);

// 找到配对的括号
int findMatchBracket(string str, int index);

// 找最右侧的小括号内的内容	需要更新掩膜数据
bool simplifyRightestBrackets(string& str, string& minus_mask);

// 对没有括号的式子求值
double calcEquationWithoutBrackets(string content, string& minus_mask);

// 判断是否是单项式
bool isMonomial(string input, vector<string> alph);

// 基本求导
string baseDerivation(string input, string flag);

// 基本运算
double baseCompute(double A, double B, string flag);

// 基本运算
string baseCompute(string strA, string strB, string flag);

// 化简算式
bool simplifyEquation(string& content, string flag, string& minus_mask);

// 计算算式的值
double calculateEquation(string equation);

// 判断string中某一位置是否在括号内
bool isInBracket(string input, int pos);

// 替换string input中所有string A为string B
bool replaceAll(string& input, string SearchTerm, string ReplacementTerm);

// 变量代入
string substituteVariables(string& input, string SearchTerm, string ReplacementTerm);

// 判断是否为特殊字符串
bool isSpecialString(string input);

// 计算单项式的值
double calculateMonomial(string monomial, double value, string unknownNum = "x");

// 计算多项式的值，多项式必须是 x[0], x[1], x[2]的形式
double calculatePolynomialToSingleValue(string polynomial, cv::Mat input);

// 计算多项式的值，多项式必须是 x[0], x[1], x[2]的形式
cv::Mat calculatePolynomial(string polynomial, cv::Mat input);

// 计算多项式的值，多项式必须是 x[0], x[1], x[2]的形式
cv::Mat calculatePolynomial(vector<string> polynomial, cv::Mat input);

// 计算矩阵形式多项式的值 多项式 Hess = [x1+x2, x1-x2;x1 ,x2]; 由逗号和分号决定式子行列
cv::Mat calculatePolynomialInMat(vector<vector<string>> matPolynomial, cv::Mat input);

// 计算矩阵形式多项式的值 多项式 Hess = [x1+x2, x1-x2;x1 ,x2]; 由逗号和分号决定式子行列
cv::Mat calculatePolynomialInMat(string matPolynomial, cv::Mat input);

// 分离多个算式
vector<string> splitManyPolynomialToSingleValue(string input);

// 分离多个算式 主要用于分解Hessen矩阵
vector<vector<string>> splitPolynomialInMat(string input);

// 判断字符串string是否是数字
bool isnum(string s);

// char单字符 转 string
string char2string(char c);

// 中缀表达式转换为后缀表达式 infixExp为带括号的四则运算
string convert2PostfixExp(string infixExp);

// 字符串合并
string mergeVecStr(vector<string> strs);

// 在字符串str中，从第idx位置开始往后找，找到最长的可能构成数字的子串
string findNum(string str, int idx);

// 后缀表达式计算值
double calculatePostfixExp(string postfixExp);

// Add 20170617
// KMP模式匹配算法实现 《大话数据结构》P141 
vector<int> get_next(string T);

// 【打印】打印调试信息 Bool
void OutputDebugStringBool(bool flag, string name);

// 【打印】打印调试信息 Int
void OutputDebugStringInt(int v, string name);

// 【打印】打印调试信息 Double
void OutputDebugStringDouble(double v, string name);


// 字符串倒序 倒过来
//string string1 = "mark kwain";
//reverse(string1.begin(), string1.end());
// ************************************* //
// char array1[] = "mark twain";
// int N1 = strlen(array1);
// reverse(&array[0], &array1[N1]);