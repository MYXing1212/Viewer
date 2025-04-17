#pragma once
#include"MyMatrix.h"

using namespace std;


// ���Է��� ��С���˽�
//solve(phi, signal, coe, DECOMP_SVD);    //�����С��������

// ���Է��� ��С���˽� AX = b    ��С���˽�Ϊx0 = A.inv() * b
double* LeastSquareSoluForLiEq(double* A, int m, int n, double* b);

// ������Է����� AX = b ;A��n*n����, b��n*1����
double* solveLiEq(double *A, double *b, int n);

// ���һԪ���η���
vector<double> solveQuadraticEqu(double a, double b, double c);

// ׷�Ϸ�������Խ����Է����� Ax = b
// A��n*n�ľ���
// b��n*1������
// c�����Խ�������һ���Խ����ϵ�����
// d�����Խ�������һ���Խ����ϵ�����
double* solve3Diag(double* a, double* c, double* d, double* b, int n);

// �����Խ��߷��������ⷽ��
// ϵ������Ϊ���µ������ԽǾ���
//		��												��
//		�� a[1]    c[1]							   d[1] ��
//		�� d[2]    a[2]    c[2]							��
//		��           �v       �v       �v					��
//		��					�v       �v	    �v			��
//	A = ��						  d[n-1]  a[n-1]  c[n-1]��
//		�� c[n]					           d[n]    a[n]	��
//		��												��
// �����Է�����Ax=b��Ϊ�����Խ����Է����飬��༼���������������Ϊ����������Է�����
// ע������ʱ��c1��d1��λ�ã�����
vector<double> solveQuasi3Diag(vector<double> a, vector<double> c, vector<double> d, vector<double> b);
