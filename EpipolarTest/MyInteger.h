#pragma once 
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<vector>
#include<cmath>
#include<string>

#define RESULT 24			// ����24�㺯��
#define PRECISION 1E-6

#define Positive_Order 1
#define Inverted_Order 0


using namespace std;

#ifndef max
#define max(a,b) ((a>=b)?a:b)
#define min(a,b) ((a<=b)?a:b)
#endif

// int �����ܴ洢�����ֵ����Сֵ int���ֵ int��Сֵ int ���ֵ int ��Сֵ
// INT_MAX
// INT_MIN

// �������һ��Int ��[a,b]��Χ��
static inline int rndInt(int b, int a = 0)
{
	int c = b - a + 1;
	return (rand()%c+a);
	
	//// ��һ��ʵ�ַ���
	//RNG rng(12345); //�����������  
	//return (rng.uniform(a, b + 1));
}


// �ҵ��������������4�ı���
int get4DivisableNumLess(int src);

// �ҵ���С���������4�ı���
int get4DivisableNumGreater(int src);

// ���ز�����input������2����������
int floor2power(int input);

// ���ز�С��input����С��2����������
int roof2power(int input);

// vector ��λ���ú�Ľ��
void bitreverseVec(double* input, int len, int N);

// ��λ����  input�Ķ�����Ϊ10010 ���ú�Ϊ 01001
// refΪ�ο�λ��������3������ = 11����������ο�λ�󣬼���ref=5 ��ô3��Ӧ�Ķ�����Ϊ00011
int bitreverse(int input, int ref=0);

// �ж��Ƿ���2����������
bool is2ofIntPow(int input);

// �׳�
int Fact(int n);

class MyInteger
{
public:
	MyInteger();
	~MyInteger();

	int SolveGCD(int x, int y);  // �������Լ��
	int SolveLCM(int x, int y);  // ������С������
	int getFiguresNum(int num);  // �õ�һ������λ��
	int getInvertedNum(int num); // �õ�һ�����ķ�����

    bool isEqual(int a,int b);         // ����������Ƿ����
	bool CheckSymmetryNum(int num);    // ���һ�����Ƿ�Ϊ�Գ���
	bool CheckPrime(int x);            // ���һ�����ǲ�������
	bool CheckPerfectNum(int num);     // ���һ�����ǲ�����ȫ��
	bool CheckNarcissisticNum(int num);// ���һ�����ǲ���ˮ�ɻ���

	void InsertSort(int a[], int M);      // ���������㷨
	void Bubble(int a[], int M);          // ð�������㷨
	void SelectSort(int a[], int M);      // ѡ������

	void PrintMultiplicationTable();      // ��ӡ�žų˷���
	void PrintYangHuiTriangle(int r);     // ��ӡ�������

	void getDivisors(int num,vector<int>& divisors, bool IfAll); // ��һ����������Լ��
	void getAllFigures(int num,vector<int>& figures, int type);  // �õ�һ�����ֵĸ�λ�ϵ���
	void getPrimeFactors(int num,vector<int>& factors);          // �ֽ�������

	string GetChineseNums(int n);         // ����Сд���д

	bool Cal(int n, double *number, string *equation);
};

