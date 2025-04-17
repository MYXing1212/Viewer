#pragma once 
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<vector>
#include<cmath>
#include<string>

#define RESULT 24			// 计算24点函数
#define PRECISION 1E-6

#define Positive_Order 1
#define Inverted_Order 0


using namespace std;

#ifndef max
#define max(a,b) ((a>=b)?a:b)
#define min(a,b) ((a<=b)?a:b)
#endif

// int 类型能存储的最大值和最小值 int最大值 int最小值 int 最大值 int 最小值
// INT_MAX
// INT_MIN

// 随机产生一个Int 在[a,b]范围内
static inline int rndInt(int b, int a = 0)
{
	int c = b - a + 1;
	return (rand()%c+a);
	
	//// 另一种实现方法
	//RNG rng(12345); //随机数产生器  
	//return (rng.uniform(a, b + 1));
}


// 找到不大于这个数的4的倍数
int get4DivisableNumLess(int src);

// 找到不小于这个数的4的倍数
int get4DivisableNumGreater(int src);

// 返回不大于input的最大的2的整数次幂
int floor2power(int input);

// 返回不小于input的最小的2的整数次幂
int roof2power(int input);

// vector 码位倒置后的结果
void bitreverseVec(double* input, int len, int N);

// 码位倒置  input的二进制为10010 倒置后为 01001
// ref为参考位数，比如3二进制 = 11，如果给出参考位后，假如ref=5 那么3对应的二进制为00011
int bitreverse(int input, int ref=0);

// 判断是否是2的整数次幂
bool is2ofIntPow(int input);

// 阶乘
int Fact(int n);

class MyInteger
{
public:
	MyInteger();
	~MyInteger();

	int SolveGCD(int x, int y);  // 计算最大公约数
	int SolveLCM(int x, int y);  // 计算最小公倍数
	int getFiguresNum(int num);  // 得到一个数的位数
	int getInvertedNum(int num); // 得到一个数的反序数

    bool isEqual(int a,int b);         // 检测两个数是否相等
	bool CheckSymmetryNum(int num);    // 检测一个数是否为对称数
	bool CheckPrime(int x);            // 检测一个数是不是素数
	bool CheckPerfectNum(int num);     // 检测一个数是不是完全数
	bool CheckNarcissisticNum(int num);// 检测一个数是不是水仙花数

	void InsertSort(int a[], int M);      // 插入排序算法
	void Bubble(int a[], int M);          // 冒泡排序算法
	void SelectSort(int a[], int M);      // 选择排序法

	void PrintMultiplicationTable();      // 打印九九乘法表
	void PrintYangHuiTriangle(int r);     // 打印杨辉三角

	void getDivisors(int num,vector<int>& divisors, bool IfAll); // 求一个数的所有约数
	void getAllFigures(int num,vector<int>& figures, int type);  // 得到一个数字的各位上的数
	void getPrimeFactors(int num,vector<int>& factors);          // 分解质因数

	string GetChineseNums(int n);         // 数字小写变大写

	bool Cal(int n, double *number, string *equation);
};

