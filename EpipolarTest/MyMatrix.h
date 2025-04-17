#pragma once

#include<malloc.h>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"MyVector.h"
#include"MyFloat.h"
#include"MyInteger.h"
#include<time.h>
#include<opencv2/opencv.hpp>

using namespace std;

// ������������Ϊ���溯���ķ���ֵ
// �ж��Ƿ�����������
//int isPositiveDefinite(cv::Mat input);
#define POSITIVE_DEFINITE 1
#define NEGITIVE_DEFINITE -1
#define POSITIVE_SEMIDEFINITE 0

#define DERIV_R		1 /* Derivative in row direction */
#define DERIV_C		2 /* Derivative in column direction */
#define DERIV_RR	3 /* Second derivative in row direction */
#define DERIV_RC	4 /* Second derivative in row and column direction */
#define DERIV_CC	5 /* Second derivative in column direction */

#define CMP_EQU					0
#define CMP_EQU_OR_GREATER		1
#define CMP_GREATER				2
#define CMP_EQU_OR_LESS			3
#define CMP_LESS				4


// ��������*��ʽ��**��ʽs

// �����󡿳�ʼ�� ��ʼֵ��Ϊ0
int** initMatrixInt(int row, int col);

// �����󡿳�ʼ�� ��ʼֵ��Ϊ0
double** initMatrix(int row, int col);

// �����󡿳�ʼ��
double* initMatrix(cv::Mat A);

// �����󡿳�ʼ����λ��
double** eye(int len);

// ��������󡿾��ȷֲ� ����a��b
cv::Mat randnMatUniform(cv::Size size, double a, double b);

// �����������ֵ�ʸ�˹�ֲ�������Size��mean����std
cv::Mat randnMat(cv::Size size, double mean, double std);

// ��������� ��ʼֵ���0��1֮��
cv::Mat randMatFloat(int row, int col);

// ��������� ��ʼֵ���0��1֮��
cv::Mat randMat(int row, int col);

// ����ʼ���������ʼ�� ԭʼ����src �е�ĳЩ �±�ΪIdxs��Ԫ�ع��ɵľ���
cv::Mat initMatIdxs(cv::Mat src, cv::Mat idxs);

// ����ʼ���������ʼ�� ԭʼ����src �е�ĳЩ�к�ĳЩ��Ԫ�ع����µľ���
cv::Mat initMatRowColIdxs(cv::Mat src, cv::Mat row_idxs, cv::Mat col_idxs);

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
cv::Mat colSelect(cv::Mat src, set<int> colIndexs);

// �󷴶Գƾ���
cv::Mat skewSymMat(cv::Mat input);

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
cv::Mat rowSelect(cv::Mat src, set<int> rowIndexs);

// // ��ĳЩ�� ĳ�� �з�Χ ��Χ�С�����ѡȡĳЩ�й����µľ���
cv::Mat colSelect(cv::Mat src, int startIdx, int endIdx);

// // ��ĳЩ�� ĳ�� �з�Χ ��Χ�С�����ѡȡĳЩ�й����µľ���
cv::Mat rowSelect(cv::Mat src, int startIdx, int endIdx);

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
cv::Mat colSelect(cv::Mat src, vector<int> colIndexs);

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
cv::Mat rowSelect(cv::Mat src, vector<int> rowIndexs);

// ����ʼ�����ԽǾ��� data��һά����
cv::Mat diagMat(cv::Mat data);

// �����󡿳�ʼ������ʼֵ��� 0��1֮��
double** randMatrix(int row, int col);

// ������ ����
void copyMatInt(int**src, int ** dst, int row, int col);

// �����󡿸���
double* copyMat(double* input, int row, int col);

// �����󡿸���
int** copyMatInt(int** src, int row, int col);

// �����󡿸���
double** copyMat(double** src, int row, int col);

// �����󡿸��Ƶ� ��data���Ƶ�src��(c,r)λ�ô� ��r�У���c��
void setMat(const cv::Mat& src, cv::Mat data, int r, int c);

// ���������öԽ���Ԫ��
void setDiagf(cv::Mat& src, float s, int i1, int i2 = -1);

// �����󡿶Խ�������Ϊ�Խ�Ԫ��Vec������Ԫ��Ϊ0
double** diagMatrix(double*vec, int len);

// �����󡿷�����ת���� ��z��
double** Rot_z(double angle);

// �����󡿷�����ת���� ��x��
double** Rot_x(double angle);

// �����󡿳�ʼ�� ���������г�ʼ��
double** initMatrix(double* vec, int row, int col);

// �����󡿵õ�ĳһ�� ע��ӵ�0�п�ʼ
double* rowM(double* mat, int row, int col, int rowNo);

// �����󡿵õ�ĳһ��
double* rowM(double** mat, int rowOffset);

// �������������
void exchange2rows(double* mat, int row, int col, int rowNo1, int rowNo2);

// �����󡿵õ�����ĳһ��
double* colM(double** mat, int row, int colOffset);

// ������ת��
double** T_Mat(double **src, int row, int col);

// ������ת��
double* T_Mat(double *src, int row, int col);

// ����������
double* InvMat(double *mat, int n);

// ����������
double** InvMat(double** ppDbMat, int nLen);

// �����������Ĺ����棨M*N�ľ���r(A) = N, �����ȣ�ʵ����
double* InvHighMat(double* A, int m, int n);

// �����������Ĺ����棨M*N�ľ���r(A) = M, �����ȣ�ʵ����
double* InvLowMat(double* A, int m, int n);

// ���������
double** sumM(double** mat1, double** mat2, int row, int col);

// ���������
double *sumM(double *a, double *b, int row, int col);

// ��������� ��֤input��ͨ����Ϊ1
double sumM(cv::Mat input);

// �����󡿱����任 ÿ��Ԫ�س���һϵ��
double** scaleM(double** mat, double scale, int row, int col);

// ���������
double** subM(double** mat, double offset, int row, int col);

// ���������
double *subM(double *a, double *b, int row, int col);

// ��������
double traceM(double** mat, int size);

// ��������
double traceM(double *m, int n);

// ��������ȡ�
int rankM(cv::Mat src);

// ������ʽ��ֵ
double detM(cv::Mat src);

// ������ʽ��ֵ
double detM(double* mat, int n);

// ��Э�������������Э�������
cv::Mat covMat(cv::Mat A);

// ������ ���ϵ������
cv::Mat corrcoef(cv::Mat A);

// ��������ȡ�
int rankM(double *mat, int m, int n);

// ��������������� mat1������Ӧ����mat2������
double** MmulM(double** mat1, double** mat2, int row1, int col1, int col2);

// ��������������� mat1������Ӧ����mat2������
double* MmulM(double *p, double *y, int row1, int col1, int col2);

// ��������������� mat1������nӦ����mat2������n   mat1 m*n    mat2 n*k
float* MmulM(float *p0, float *p1, int m, int n, int k);

// ������ �������ӦԪ����ˣ�mat1�ĳߴ�Ӧ�õ���mat2�ĳߴ�
float* MmulMEle(float* m1, float* m2, int row, int col);

// ������ �������ӦԪ����ˣ�mat1�ĳߴ�Ӧ�õ���mat2�ĳߴ�
double* MmulMEle(double* m1, double* m2, int row, int col);

// �����󡿴�ӡ����
void printMatrix(double **m, int row, int col);

// �����󡿴�ӡ����
void printMatrix(int **m, int row, int col);

// �����󡿴�ӡ����
void printMatrix(double *m, int row, int col);

// �����󡿼���Գƾ������������ֵ�Ͷ�Ӧ����������
void getMaxMatEigen(double* m, double& eigen, vector<double> &q, int n);

// �����󡿼���Գƾ������ֵ��С������ֵ�Ͷ�Ӧ���������� �����ǶԳƾ��󣡣���
double* getAbsMinMatEigen(double* a, double& eigen, int n);

// �����󡿼���ʵ�Գƾ����ȫ������ֵ����������
int calEgiensAndVectors(double* a, int n, double *lambda, double** vec);

// �����󡿼��� ����ֵ����������
int getEigensAndVecs(cv::Mat input, cv::Mat& eigens, cv::Mat& vectors);

// �����󡿼��� ����ֵ����������
//  eigens �� vectors Ӧ�����Ѿ�������ڴ��
int getEigensAndVecs(double *input, int n, double*eigens, double* vectors);

// ��MATLAB��QR�ֽ���ص�householder���� 
// ��������ԳƵ�Householder����H��ʹHx = rho*y������rho = -sign(x(1))* || x || / || y ||
// ����˵��
// x��������
// y����������x��y���������ͬ��ά��
cv::Mat householder(cv::Mat x, cv::Mat y);

// ����Householder�任��������A�ֽ�ΪA = QR������QΪ��������RΪ��������
// ����˵��
// A����Ҫ����QR�ֽ�ķ���
// Q���ֽ�õ�����������
// R���ֽ�õ�����������
void QR(cv::Mat A, cv::Mat& Q, cv::Mat& R);

// �ֿ���� A, B�ڶԽ��ߣ�����Ϊ0
cv::Mat blkdiag(cv::Mat A, cv::Mat B);

cv::Mat blkdiag(cv::Mat& Ht, cv::Size& rect);

bool QR_householder(cv::Mat A, cv::Mat &Q, cv::Mat& R);


// ������ �ֽ⡿�Գ����������Cholesky�ֽ�
// ��һ���Գ������ľ����ʾ��һ�������Ǿ���L����ת�õĳ˻��ķֽ� A = L*L.t()
double* cholesky(double* A, int n);

// ������ �ֽ⡿�Գ����������Cholesky�ֽ�
// ��һ���Գ������ľ����ʾ��һ�������Ǿ���L����ת�õĳ˻��ķֽ� A = L*L.t()
cv::Mat cholesky(cv::Mat A);

// ������ �ֽ⡿�Գ����������Cholesky�ֽ�
// ��һ���Գ������ľ����ʾ��һ�������Ǿ���L����ת�õĳ˻��ķֽ� A = C.t()*L
cv::Mat cholesky2(cv::Mat A);

// �����󡿽�����Լ��Ϊ��ɭ������� �������Ǿ��󻹶�һб�ľ���
/*
| x x x x x|
| x x x x x|
| 0 x x x x|
| 0 0 x x x|
| 0 0 0 x x|
*/
double* hessenberg(double* a, int n);

// ������ �����桿 α�� �Ӻ���
cv::Mat InvMat(cv::Mat src);

// ������ �����桿
double* InvMat(double* input, int row, int col);

/************************************************************************/
/* input:
/* a:���m*nʵ����A,����ʱ�����������
/* m:���� n������
/* u:���m*m����������, v:���n*n����������
/* eps:��������Ҫ��,  ka: max(m,n)+1
/* output:
/* ����ֵ���Ϊ��������ʾ������60�Σ���δ�������ֵ������ֵΪ�Ǹ���������
/************************************************************************/
int svdDecomp(double *a, int m, int n, double *u, double *v, int ka, double eps=1e-6);

// ����ֵ�ֽ� SVD ��������
static void ppp(double a[], double e[], double s[], double v[], int m, int n);

// ����ֵ�ֽ� SVD  ��������
static void sss(double fg[], double cs[]);

// ������ֽ⡿SVD�ֽ� A = P*delta*Q.t() P��m*r�ľ��� delta��r*r�ľ��������Խ���Ԫ��ΪA������ֵ�� QΪn*r�ľ���
void svdDecomp(double* input, int m, int n, double *P, double* delta, double *Q);

// ������ֽ⡿SVD�ֽ� A = P*delta*Q.t() P��m*r�ľ��� delta��r*r�ľ��������Խ���Ԫ��ΪA������ֵ�� QΪn*r�ľ���
void svdDecomp(cv::Mat src, cv::Mat& P, cv::Mat& delta, cv::Mat& Q);

// �����������������������ȫ1��
void AddOnesRow(cv::Mat& input);

// �����󡿵�� ��������ĵ����Դ����mat1��mat2��ά������һ�£���������result��
// mat1		M*N
// mat2		M*N
// result	1*N
// ��һ�� mat1��mat2�ж�ӦԪ����� �� temp
// �ڶ��� result(0, i) = temp��i��Ԫ�غ�
bool dotarray(cv::Mat mat1, cv::Mat mat2, cv::Mat& result);


// ��MATLAB��ͬ����������һ�£���Դ���鰴rows��cols��ֵ������չ�������dst��
// ����ԭ���ľ���AΪ	| 1 0 0 |
//				A = | 0 2 0 |
//					| 0 0 3 | 
// ����rows = 3, cols = 3
// ����Ϊ
//					| 1 0 0 1 0 0 1 0 0 |
//					| 0 2 0 0 2 0 0 2 0 |
//					| 0 0 3 0 0 3 0 0 3 |
//					| 1 0 0 1 0 0 1 0 0 |
//			DST = 	| 0 2 0 0 2 0 0 2 0 |
//					| 0 0 3 0 0 3 0 0 3 |
//					| 1 0 0 1 0 0 1 0 0 |
//					| 0 2 0 0 2 0 0 2 0 |
//					| 0 0 3 0 0 3 0 0 3 |
cv::Mat repmat(cv::Mat src, int rows, int cols);

// ��ʵ�Գ����ȫ������ֵ����������
int cjcbi(double*a, int n, double*v, double eps, int jt);

// ������������Сֵ
void maxAndmin(double &bmax, double &bmin, double* data, int col, int row);

// �ж��Ƿ��ǶԳƾ���
bool isSymmetry(cv::Mat input);

// �ж��Ƿ�����������
int isPositiveDefinite(cv::Mat input);

// ����������cond
double cond(cv::Mat input);

// Doolittle �ֽ� LU �ֽ�
bool decompDoolittle(cv::Mat input, cv::Mat& L, cv::Mat& U);

// Crout �ֽ� LU �ֽ�
bool decompCrout(cv::Mat input, cv::Mat& L_, cv::Mat& U_);

// ���������жϾ����Ƿ�Ϊ������
bool isSingular(cv::Mat A);

// �����ȡ��жϾ����Ƿ�����
bool isFullRank(cv::Mat A);

// �������жϾ����Ƿ�Ϊ����
bool isSquare(cv::Mat A);

// ��������������Ԫ�ؾ���ֵ���ֵ
double infiniteNorm(cv::Mat A);

// ��1������������1���� 1-����
double norm_L1(cv::Mat A);

// ���Խ���Ԫ�����ֵ��
double maxValueInDiag(cv::Mat input);

// svd�ֽ� ��MATLAB����һ�� �����V��MATLAB�е�v��ת�ù�ϵ
bool svd(cv::Mat A, cv::Mat& U, cv::Mat &S, cv::Mat &V, bool useDouble = true);

// ���ά����Ҷ�任 double�� ����Ϊ��ͨ�� ʵ��; ����ֵΪ����
cv::Mat fft(cv::Mat input, cv::Mat& mag, cv::Mat& angle);

//// ������չ��1ά
//cv::Mat Mat2Vec(cv::Mat input);

// �ֿ���������һ�� ���� 
cv::Mat combine2MatV(cv::Mat Up, cv::Mat Down);

// �ֿ���������һ�� ����
cv::Mat combine2MatH(cv::Mat Left, cv::Mat Right);

// Hermiteת�� input.channel() == 2
cv::Mat hermiteT(cv::Mat input);

// ������Ȼָ�� 
cv::Mat complexExp(cv::Mat X);

// �����˷� X,Y ����˫ͨ����  ����ֵҲ��˫ͨ����
cv::Mat complexMul(cv::Mat X, cv::Mat Y);

// �����˷� ����ϵ�� Point2d X * cv::Mat Y
cv::Mat complexScale(cv::Mat Y, cv::Point2d X);

// �����˷� ������
cv::Mat complexMulEle(cv::Mat X, cv::Mat Y);

// �����˷� ����ϵ�� Point2d X * Point2d Y
cv::Point2d complexMul(cv::Point2d X, cv::Point2d Y);

// ���������ֵ
cv::Mat complexAbsMat(cv::Mat input);

// ���������ֵ
cv::Mat complexAbsMat(cv::Mat input);

// cv::Matȡʵ�� input˫ͨ��
cv::Mat real(cv::Mat input);

// cv::Matȡ�鲿 input˫ͨ��
cv::Mat imag(cv::Mat input);

// ʵ�����鲿�ϲ�Ϊ2ͨ������
cv::Mat merge(cv::Mat real, cv::Mat imag);

// ʵ������ת�������� �൱�ڲ���һ��ȫ����
cv::Mat real2complex(cv::Mat input);


// ���ɷַ���
// ���룺 A		---	��������ÿ��Ϊһ������
//		 k		--- ��ά��kά
// ����� pcaA	--- ��ά���kά��������������ɵľ���ÿ��һ������������kΪ��ά�������������
//		 V		--- ���ɷַ���
void PCATrans(cv::Mat input, int k, cv::Mat& pcaA, cv::Mat& V);

// չ�������� ����չ��
cv::Mat convert2Vec(cv::Mat input);



// Hadamard ����
//% HADAMARD Hadamard matrix.
//%   HADAMARD(N) is a Hadamard matrix of order N, that is,
//%   a matrix H with elements 1 or - 1 such that H'*H = N*EYE(N).
//%   An N - by - N Hadamard matrix with N > 2 exists only if REM(N, 4) = 0.
//%   This function handles only the cases where N, N / 12 or N / 20
//% is a power of 2.
//%
//%   HADAMARD(N, CLASSNAME) produces a matrix of class CLASSNAME.
//%   CLASSNAME must be either 'single' or 'double' (the default).
//
//%   Nicholas J.Higham
//%   Copyright 1984 - 2005 The MathWorks, Inc.
//
//%   Reference:
//%   S.W.Golomb and L.D.Baumert, The search for Hadamard matrices,
//%   Amer.Math.Monthly, 70 (1963) pp. 12 - 17.
cv::Mat hadamard(int N);

// toeplitz�����Ĺ����������������ȣ�toeplitz������
//	�������Ⱦ�����ص��ǣ�����һ�С���һ���⣬����ÿ��Ԫ�ض��������Ͻǵ�Ԫ����ͬ��
//	���ø�ʽ��
//	A = toeplitz(��1��Ԫ�����飬��1��Ԫ�����飩
cv::Mat toeplitz(cv::Mat c, cv::Mat r);

// hankel�����Ĺ���������Hankel����
//	Hankel������ص��ǣ�����һ�С����һ���⣬����ÿ��Ԫ�ض��������½ǵ�Ԫ����ͬ��
//	���ø�ʽ��
//	A = hankel(��1��Ԫ������,���һ��Ԫ�����飩
// ע�⣺���һ�еĵ�1��Ԫ��Ӧ���1�еĵ�1��Ԫ����ͬ���������һ�еĵ�һ��Ԫ�ؽ��Զ���Ϊ1�еĵ�1��Ԫ�ء�
cv::Mat hankel(cv::Mat c, cv::Mat r);

// �ϲ������� Ĭ��Ϊ����ϲ� ����ƴ�� ���Һϲ� ����ƴ�� ���ºϲ�
cv::Mat combine2Mat(cv::Mat A, cv::Mat B, bool CmbHor = true);

// �������о���ֵС��val��Ԫ������
cv::Mat set2zeroAbsBelowThresh(cv::Mat& input, double val);

// �ж����������Ƿ���ͬ ����ͼ���Ƿ���ͬ
bool isTwoMatEqual(cv::Mat A, cv::Mat B);

// ��� ��˹��
void convolve_gauss(cv::Mat src, cv::Mat& dst, double sigma, long deriv_type);

// Gaussian filtering and Gaussian derivative filters
void gfilter(cv::Mat src, cv::Mat& dst, double sigma, cv::Mat orders);

// GAUSSIAN DERIVATIVE KERNEL - creates a gaussian deivative kernel.
void gaussiankernel(float sigma, int order, cv::Mat& outputArray);

//// �����ֵ�����ؾ�������ֵ
//double maxM(cv::Mat input);

// ����Сֵ�����ؾ������Сֵ
double minM(cv::Mat input);

// �����ֵ�����ؾ�������ֵ����ƫ��ֵ
int maxIndex(cv::Mat input);

// ����Сֵ�����ؾ������Сֵ����ƫ��ֵ
int minIndex(cv::Mat input);

// ���Ƚϡ������뵥����ֵ ���ؾ���������inputͬ�ߴ磬����cv::Mat��������ΪCV_32S1 
// ����cmpFlag = CMP_EQU				ʱ����ĳλ�ó�cv::MatԪ�ص���value�򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_EQU_OR_GREATER	ʱ����ĳλ�ó�cv::MatԪ�ش��ڵ���value�򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_GREATER			ʱ����ĳλ�ó�cv::MatԪ�ش���value�򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_EQU_OR_LESS		ʱ����ĳλ�ó�cv::MatԪ��С�ڵ���value�򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_LESS			ʱ����ĳλ�ó�cv::MatԪ��С��value�򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
cv::Mat cmpMatVSVal(cv::Mat input, double value, int cmpFlag = CMP_EQU);

// ���Ƚϡ����������Ƚ� ���ؾ����������������ߴ磬����cv::Mat��������ΪCV_32S1 
// ����cmpFlag = CMP_EQU				ʱ����ĳλ�ó�AԪ�ص���BԪ���򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_EQU_OR_GREATER	ʱ����ĳλ�ó�AԪ�ش��ڵ���BԪ���򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_GREATER			ʱ����ĳλ�ó�AԪ�ش���BԪ���򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_EQU_OR_LESS		ʱ����ĳλ�ó�AԪ��С�ڵ���BԪ���򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_LESS			ʱ����ĳλ�ó�AԪ��С��BԪ���򣬽��cv::Mat�ô���Ϊ1��������Ϊ0
cv::Mat cmpMatVSMat(cv::Mat A, cv::Mat B, int cmpFlag = CMP_EQU);

// ����Ӿ��� get submatrix
cv::Mat getSubMat(cv::Mat src, int r1, int c1, int r2=-1, int c2=-1);

// ����N�׸���Ҷ����cv::Mat ˫ͨ��  ��j�е�k�е�Ԫ�ر��ʽΪexp��2��ijk/n��
cv::Mat FourierMat(int n);

// ������Ļ���� 
// ����A�Ļ������A�в�ͬ�еĹ�һ���ڻ��ľ���ֵ�е����ֵ
double CrossCorrelationMat(cv::Mat A);

// ������ֵ��cv::Mat�и�Ԫ��ȡ����ֵ
cv::Mat absMat(cv::Mat input);

// �е�λ��
void colNormalized(cv::Mat& input);

// vector<int> ���L��Ԫ�ض�Ӧ���±�
vector<int> maxValuesIndex(cv::Mat input, int L);

// ˳ʱ��ת90��
cv::Mat turnRight(cv::Mat input);

// ��ʱ��ת90��
cv::Mat turnLeft(cv::Mat input);

// �����ֵ ����������Щһ��һ�еľ���
double valM_double(cv::Mat input);

// �����ֵ ����������Щһ��һ�еľ���
double valM_int(cv::Mat input);

// ���ž��� �Ծ������sign���㣬�����ͬ����С�ľ��� ֵֻ��+1 ��-1
cv::Mat signMat(cv::Mat input);

// ��ȡĳԪ�ص�����Ԫ��
// type ��Ϊ4���� �� 8����
vector<cv::Point> adjacentPixels(cv::Mat data, cv::Point p, int type = 8);

// �ж����������Ƿ����
bool isEqual(cv::Mat A, cv::Mat B);

// ���󳤶� ��Ϊ��������Ϊ�������ȣ��������� �� ����Ϊ����
int length(cv::Mat m);


// ����Ԫ��ת����
template<typename T>
cv::Mat val2Mat(T val){
	return (Mat_<T>(1, 1) << val);
}

// ȡ�������Ԫ�� ��Ԫ�ؾ���ȡֵ
template<typename T>
T Mat2val(cv::Mat input){
	return (input.ptr<T>(0)[0]);
}

// ���Ӽ� �ض� roi range��vector�н�ȡĳһ���� ������
template<typename T>
T maxM(cv::Mat input){
	double maxValue;
	cv::minMaxLoc(input, NULL, &maxValue);
	T result = (T)maxValue;
	return result;
}

template<typename T>
T atM(cv::Mat input, cv::Point pt){
	CV_Assert(pt.x >= 0 && pt.x < input.cols && pt.y >= 0 && pt.y < input.rows);
	return input.ptr<T>(0)[pt.y*input.cols + pt.x];
}

template<typename T>
T atM(cv::Mat input, int r, int c){
	CV_Assert(r >= 0 && r < input.rows && c >= 0 && c < input.cols);
	return input.ptr<T>(0)[r*input.cols + c];
}

// ��������Ľ�Сֵ���ɵľ���
template<typename T> 
cv::Mat minMat(cv::Mat A, cv::Mat B){
	CV_Assert(A.size() == B.size());
	cv::Mat result = Mat::zeros(A.size(), A.type());
	for (int i = 0; i < (int)A.total(); i++){
		result.ptr<T>(0)[i] = min(A.ptr<T>(0)[i], B.ptr<T>(0)[i]);
	}
	return result;
}

// �������ĳֵ�� �ϴ�ֵ���ɵľ���
template<typename T>
cv::Mat maxMat(cv::Mat A, T val){
	cv::Mat result = Mat::zeros(A.size(), A.type());
	for (int i = 0; i < (int)A.total(); i++){
		result.ptr<T>(0)[i] = max(A.ptr<T>(0)[i], val);
	}
	return result;
}

// ����Ӿ��� ĳЩԪ�� ������ʼԪ���±� �����Ԫ���±� 
// �Ӿ���Ϊ������ 
template<typename T>
cv::Mat subMat(cv::Mat x, int start, int end){
	Mat r = Mat_<T>(end - start + 1, 1);
	for (int i = 0; i < end - start + 1; i++){
		r.ptr<T>(0)[i] = x.ptr<double>(0)[start + i];
	}
	return r;
}




// ���ţ��������˳��
//RNG rng(12345); //�����������  
//int i, sampleCount = rng.uniform(1, 1001);
//cv::Mat points(sampleCount, 1, CV_32FC2), labels;   //��������������ʵ����Ϊ2ͨ������������Ԫ������ΪPoint2f  
//randShuffle(points, 1, &rng);   //��ΪҪ���࣬�������������points����ĵ㣬ע��points��pointChunk�ǹ������ݵġ�  