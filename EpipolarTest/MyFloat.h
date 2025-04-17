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
// ���������ֵ ��������Сֵ
// float �����ܴ洢�����ֵ����Сֵ float���ֵ float��Сֵ float ���ֵ float ��Сֵ
// FLT_MAX
// FLT_MIN
// double �����ܴ洢�����ֵ����Сֵ double���ֵ double��Сֵ double ���ֵ double ��Сֵ 
// DBL_MAX
// DBL_MIN

int RoundOff(float input);

//���ظ�������������ȡ����ĸ�����
double ROUND(double dbA);

// ���ɸ�˹�ֲ�/��̬�ֲ��������
double GaussRand(double mu=0.0, double sigma=1.0);

static inline double rndDouble()
{
	return (rand() / double(RAND_MAX));
}

// ���ź���
double signValue(double input);

// ���λ���� 1�����Ǹ�λ -1����С�����1λ
int topDigit(double input);

// ��ȡ������ϱ߽�
double limitUp(double input, double ref);

// ��ȡ������±߽�
double limitDown(double input, double ref);

// ��ȡĳ���Ͻ�����������Ϊref
double calTopNum(double src, double ref = 0);

// ��ȡĳ���½�����������Ϊref
double calFloorNum(double src, double ref = 0);

// �������
double getAmpl(double real, double imag);

// ������λ
double getAngl(double real, double imag);

// ���ź���
double sign(double x);

// the real array X, returns an
//array F of real numbers, usually in the range 0.5 <= abs(F) < 1,
//	and an array E of integers, so that X = F .* 2.^E.
// ��һ��Ԫ��ΪF�� �ڶ���Ԫ��ΪE
pair<double, int> log2Double(double x);

// ���ظ���ĳ����һ����������λ������3�Ļ�����1�� 0.5�Ļ�����0.1
// -0.02�Ļ�����0.01
double resolution(double x);

//����ָ����Χ�ڵ����������
// �������������������[dbLow, dbUpper)
double rnd(double dbLow, double dbUpper);

//�õ�ǰʱ����ʼ��������ӣ���ֹÿ�����еĽ������ͬ
void setRandomSeed();

// ����������
template<typename T>
void swapV(T& a, T& b){
	T c;
	c = a;
	a = b;
	b = c;
}