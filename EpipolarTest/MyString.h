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
#define MNS_INDEX 6		// ����
#define MUL_INDEX 7
#define DIV_INDEX 8

#define STRSAFE_NO_DEPRECATE

using namespace std;

#ifndef NATURAL_LOGARITHM
#define NATURAL_LOGARITHM 2.718281828459
#endif

#pragma once
// �ַ�����
class MyString
{
public:
	MyString(void);
	~MyString(void);



	// �ַ���ת����
	long str2long(string s);

	

	

	// �ж�һ���ַ����Ƿ�Ϊ�����ַ���
	bool ifHuiwenString(string a);

	// �ָ��ַ������ָ�������result���棬�ָ�ı�־Ϊflag
	void splitString(string input, string flag, vector<string>& result);

	// �ָ��ַ�������õ�һ�ηָ���
	void getFirstSplit(string input, string flag, string& result);

	// �ָ��ַ�����������һ�ηָ���
	void getLastSplit(string input, string flag, string& result);

	

	

	// Unicode ת ���ֽ�
	int ConvertUnicodeToAscii(const CString &csData, char* pszData);

	// �ַ�ת��
	BOOL WChar2MByte(LPCWSTR srcBuff, LPSTR destBuff, int nlen);

	// ��ȡ��ǰʱ����ַ�����ʽ 2015/6/11 22:40:33
	CString getTimeCString();
};

// ��char*�ַ���ת��ΪLPCTSTR�ַ���
wchar_t* Char2LPCTSTR(char * str);

//// stringתLPCSTR
//wchar_t* string2LPCSTR(string str);

// const char* ת char*
char* pConstC2pC(const char* ch);

// ��stringתconst char*�ַ���
const char* string2pChar(string str);

// ��CString�ַ���ת��Ϊconst char*�ַ���
const char* CString2pChar(CString cstring);

// ��CString�ַ���ת��Ϊstring�ַ���
string CString2string(CString cstring);

// ����ת�ַ�
string int2string(int a);

// intת CString
CString int2CString(int i);

// floatת CString
CString float2CString(float f);

// doubleת CString
CString double2CString(double d);

// ������ת�ַ� accuracy = 1����С�����һλ
string double2string(double a,int accuracy = 1);

// ��CString�ַ���ת��Ϊstring�ַ���
string CString2string(CString cstring);

// ��CString�ַ���ת��Ϊchar* �ַ���
char* CString2pchar(CString cstring);

// ��string�ַ���ת��ΪCString�ַ���
CString string2CString(string str);

// ��char* ת��Ϊ LPCWSTR
LPCWSTR pchar2LPCWSTR(char* p);

// ��char*�ַ���ת��ΪLPCTSTR�ַ���
LPCTSTR pchar2LPCTSTR(char * str);

// ��char*תwchar_t*
wchar_t* pchar2pwchar_t(char* str);

// char* ת TCHAR*
TCHAR* pchar2pTCHAR(const char* str);

// ȡCString���ҵ�count���ַ�
CString RightCString(CString input, int count);

// CString�Ӻ���ǰ��"XXX",���ڵ�λ��   2\1.txt , '\\' ���Ϊ1
int b2fFind(CString src, char c);

// ȡCString�����count���ַ�
CString LeftCString(CString input, int count);

// ȡCString�Ӻ���ǰ�ҡ�x�� ֮ǰ���ַ�
CString LStr_b2fFind(CString input, char c);

// ȡCString�Ӻ���ǰ�ҡ�x�� ֮����ַ�
CString RStr_b2fFind(CString input, char c);

// ����ӡ�ַ�������ӡCStingArray	��Ҫ��ָ�����ʽ
void printCStringArray(CStringArray* strArray);

// ���ַ�����ת��Ϊdouble
double string2double(string str);

// ���ַ�����ת��Ϊbool
bool string2bool(string str);

// CString ת��Ϊ int
int CString2int(CString str);

// ���ַ�����ת��Ϊint
int string2int(string str);

// ��·������ȡ����·��
CString getWorkDir();

// ����ǰʱ�䡿
CString getCurrentTimeString(bool bAbbrev = false);

// ����ǰ���ڡ�
CString getCurrentDateString();

// ɾ�������ַ�a
bool removeAllSpecChar(string& str, char c);

// �ҵ���Ե�����
int findMatchBracket(string str, int index);

// �����Ҳ��С�����ڵ�����	��Ҫ������Ĥ����
bool simplifyRightestBrackets(string& str, string& minus_mask);

// ��û�����ŵ�ʽ����ֵ
double calcEquationWithoutBrackets(string content, string& minus_mask);

// �ж��Ƿ��ǵ���ʽ
bool isMonomial(string input, vector<string> alph);

// ������
string baseDerivation(string input, string flag);

// ��������
double baseCompute(double A, double B, string flag);

// ��������
string baseCompute(string strA, string strB, string flag);

// ������ʽ
bool simplifyEquation(string& content, string flag, string& minus_mask);

// ������ʽ��ֵ
double calculateEquation(string equation);

// �ж�string��ĳһλ���Ƿ���������
bool isInBracket(string input, int pos);

// �滻string input������string AΪstring B
bool replaceAll(string& input, string SearchTerm, string ReplacementTerm);

// ��������
string substituteVariables(string& input, string SearchTerm, string ReplacementTerm);

// �ж��Ƿ�Ϊ�����ַ���
bool isSpecialString(string input);

// ���㵥��ʽ��ֵ
double calculateMonomial(string monomial, double value, string unknownNum = "x");

// �������ʽ��ֵ������ʽ������ x[0], x[1], x[2]����ʽ
double calculatePolynomialToSingleValue(string polynomial, cv::Mat input);

// �������ʽ��ֵ������ʽ������ x[0], x[1], x[2]����ʽ
cv::Mat calculatePolynomial(string polynomial, cv::Mat input);

// �������ʽ��ֵ������ʽ������ x[0], x[1], x[2]����ʽ
cv::Mat calculatePolynomial(vector<string> polynomial, cv::Mat input);

// ���������ʽ����ʽ��ֵ ����ʽ Hess = [x1+x2, x1-x2;x1 ,x2]; �ɶ��źͷֺž���ʽ������
cv::Mat calculatePolynomialInMat(vector<vector<string>> matPolynomial, cv::Mat input);

// ���������ʽ����ʽ��ֵ ����ʽ Hess = [x1+x2, x1-x2;x1 ,x2]; �ɶ��źͷֺž���ʽ������
cv::Mat calculatePolynomialInMat(string matPolynomial, cv::Mat input);

// ��������ʽ
vector<string> splitManyPolynomialToSingleValue(string input);

// ��������ʽ ��Ҫ���ڷֽ�Hessen����
vector<vector<string>> splitPolynomialInMat(string input);

// �ж��ַ���string�Ƿ�������
bool isnum(string s);

// char���ַ� ת string
string char2string(char c);

// ��׺���ʽת��Ϊ��׺���ʽ infixExpΪ�����ŵ���������
string convert2PostfixExp(string infixExp);

// �ַ����ϲ�
string mergeVecStr(vector<string> strs);

// ���ַ���str�У��ӵ�idxλ�ÿ�ʼ�����ң��ҵ���Ŀ��ܹ������ֵ��Ӵ�
string findNum(string str, int idx);

// ��׺���ʽ����ֵ
double calculatePostfixExp(string postfixExp);

// Add 20170617
// KMPģʽƥ���㷨ʵ�� �������ݽṹ��P141 
vector<int> get_next(string T);

// ����ӡ����ӡ������Ϣ Bool
void OutputDebugStringBool(bool flag, string name);

// ����ӡ����ӡ������Ϣ Int
void OutputDebugStringInt(int v, string name);

// ����ӡ����ӡ������Ϣ Double
void OutputDebugStringDouble(double v, string name);


// �ַ������� ������
//string string1 = "mark kwain";
//reverse(string1.begin(), string1.end());
// ************************************* //
// char array1[] = "mark twain";
// int N1 = strlen(array1);
// reverse(&array[0], &array1[N1]);