#pragma once
#include"MyMatrix.h"

using namespace std;


// 线性方程 最小二乘解
//solve(phi, signal, coe, DECOMP_SVD);    //求解最小二乘问题

// 线性方程 最小二乘解 AX = b    最小二乘解为x0 = A.inv() * b
double* LeastSquareSoluForLiEq(double* A, int m, int n, double* b);

// 求解线性方程组 AX = b ;A是n*n矩阵, b是n*1矩阵
double* solveLiEq(double *A, double *b, int n);

// 求解一元二次方程
vector<double> solveQuadraticEqu(double a, double b, double c);

// 追赶法求解三对角线性方程组 Ax = b
// A是n*n的矩阵
// b是n*1的向量
// c是主对角线上面一条对角线上的数据
// d是主对角线下面一条对角线上的数据
double* solve3Diag(double* a, double* c, double* d, double* b, int n);

// 拟三对角线方程组的求解方法
// 系数矩阵为如下的拟三对角矩阵
//		┏												┓
//		┃ a[1]    c[1]							   d[1] ┃
//		┃ d[2]    a[2]    c[2]							┃
//		┃           ╲       ╲       ╲					┃
//		┃					╲       ╲	    ╲			┃
//	A = ┃						  d[n-1]  a[n-1]  c[n-1]┃
//		┃ c[n]					           d[n]    a[n]	┃
//		┗												┛
// 的线性方程组Ax=b称为拟三对角线性方程组，许多技术问题往往最后归结为求解这种线性方程组
// 注意输入时，c1和d1的位置！！！
vector<double> solveQuasi3Diag(vector<double> a, vector<double> c, vector<double> d, vector<double> b);
