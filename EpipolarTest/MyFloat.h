#pragma once
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<iostream>
#include<vector>
#include<cmath>
#include<opencv2/opencv.hpp>

using namespace std;
// 浮点数最大值 浮点数最小值
// float 类型能存储的最大值和最小值 float最大值 float最小值 float 最大值 float 最小值
// FLT_MAX
// FLT_MIN
// double 类型能存储的最大值和最小值 double最大值 double最小值 double 最大值 double 最小值 
// DBL_MAX
// DBL_MIN

int RoundOff(float input);

//返回浮点数四舍五入取整后的浮点数
double ROUND(double dbA);

// 生成高斯分布/正态分布的随机数
double GaussRand(double mu=0.0, double sigma=1.0);

static inline double rndDouble()
{
	return (rand() / double(RAND_MAX));
}

// 符号函数
double signValue(double input);

// 最高位数字 1代表是个位 -1代表小数点后1位
int topDigit(double input);

// 算取区间段上边界
double limitUp(double input, double ref);

// 算取区间段下边界
double limitDown(double input, double ref);

// 算取某数上界数，参照数为ref
double calTopNum(double src, double ref = 0);

// 算取某数下界数，参照数为ref
double calFloorNum(double src, double ref = 0);

// 计算幅度
double getAmpl(double real, double imag);

// 计算相位
double getAngl(double real, double imag);

// 符号函数
double sign(double x);

// the real array X, returns an
//array F of real numbers, usually in the range 0.5 <= abs(F) < 1,
//	and an array E of integers, so that X = F .* 2.^E.
// 第一个元素为F， 第二个元素为E
pair<double, int> log2Double(double x);

// 返回高于某数的一个数量级单位，比如3的话返回1， 0.5的话返回0.1
// -0.02的话返回0.01
double resolution(double x);

//返回指定范围内的随机浮点数
// 产生的随机数在区间内[dbLow, dbUpper)
double rnd(double dbLow, double dbUpper);

//用当前时间点初始化随机种子，防止每次运行的结果都相同
void setRandomSeed();

// 交换两个数
template<typename T>
void swapV(T& a, T& b){
	T c;
	c = a;
	a = b;
	b = c;
}