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

// 以下三个定义为下面函数的返回值
// 判断是否是正定矩阵
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


// 凡矩阵，有*格式和**格式s

// 【矩阵】初始化 初始值皆为0
int** initMatrixInt(int row, int col);

// 【矩阵】初始化 初始值皆为0
double** initMatrix(int row, int col);

// 【矩阵】初始化
double* initMatrix(cv::Mat A);

// 【矩阵】初始化单位阵
double** eye(int len);

// 【随机矩阵】均匀分布 给出a和b
cv::Mat randnMatUniform(cv::Size size, double a, double b);

// 【随机矩阵】数值呈高斯分布，给出Size，mean，和std
cv::Mat randnMat(cv::Size size, double mean, double std);

// 【随机矩阵】 初始值随机0到1之间
cv::Mat randMatFloat(int row, int col);

// 【随机矩阵】 初始值随机0到1之间
cv::Mat randMat(int row, int col);

// 【初始化】矩阵初始化 原始矩阵src 中的某些 下标为Idxs的元素构成的矩阵
cv::Mat initMatIdxs(cv::Mat src, cv::Mat idxs);

// 【初始化】矩阵初始化 原始矩阵src 中的某些行和某些列元素构成新的矩阵
cv::Mat initMatRowColIdxs(cv::Mat src, cv::Mat row_idxs, cv::Mat col_idxs);

// // 【某些列 某列】矩阵选取某些列构成新的矩阵
cv::Mat colSelect(cv::Mat src, set<int> colIndexs);

// 求反对称矩阵
cv::Mat skewSymMat(cv::Mat input);

// // 【某些行 某行】矩阵选取某些行构成新的矩阵
cv::Mat rowSelect(cv::Mat src, set<int> rowIndexs);

// // 【某些列 某列 列范围 范围列】矩阵选取某些列构成新的矩阵
cv::Mat colSelect(cv::Mat src, int startIdx, int endIdx);

// // 【某些行 某行 行范围 范围行】矩阵选取某些行构成新的矩阵
cv::Mat rowSelect(cv::Mat src, int startIdx, int endIdx);

// // 【某些列 某列】矩阵选取某些列构成新的矩阵
cv::Mat colSelect(cv::Mat src, vector<int> colIndexs);

// // 【某些行 某行】矩阵选取某些行构成新的矩阵
cv::Mat rowSelect(cv::Mat src, vector<int> rowIndexs);

// 【初始化】对角矩阵 data是一维向量
cv::Mat diagMat(cv::Mat data);

// 【矩阵】初始化，初始值随机 0到1之间
double** randMatrix(int row, int col);

// 【矩阵】 复制
void copyMatInt(int**src, int ** dst, int row, int col);

// 【矩阵】复制
double* copyMat(double* input, int row, int col);

// 【矩阵】复制
int** copyMatInt(int** src, int row, int col);

// 【矩阵】复制
double** copyMat(double** src, int row, int col);

// 【矩阵】复制到 将data复制到src的(c,r)位置处 第r行，第c列
void setMat(const cv::Mat& src, cv::Mat data, int r, int c);

// 【矩阵】设置对角线元素
void setDiagf(cv::Mat& src, float s, int i1, int i2 = -1);

// 【矩阵】对角阵，输入为对角元素Vec，其他元素为0
double** diagMatrix(double*vec, int len);

// 【矩阵】返回旋转矩阵 绕z轴
double** Rot_z(double angle);

// 【矩阵】返回旋转矩阵 绕x轴
double** Rot_x(double angle);

// 【矩阵】初始化 用向量进行初始化
double** initMatrix(double* vec, int row, int col);

// 【矩阵】得到某一行 注意从第0行开始
double* rowM(double* mat, int row, int col, int rowNo);

// 【矩阵】得到某一行
double* rowM(double** mat, int rowOffset);

// 交换矩阵的两行
void exchange2rows(double* mat, int row, int col, int rowNo1, int rowNo2);

// 【矩阵】得到矩阵某一列
double* colM(double** mat, int row, int colOffset);

// 【矩阵】转置
double** T_Mat(double **src, int row, int col);

// 【矩阵】转置
double* T_Mat(double *src, int row, int col);

// 【矩阵】求逆
double* InvMat(double *mat, int n);

// 【矩阵】求逆
double** InvMat(double** ppDbMat, int nLen);

// 【矩阵】求高阵的广义逆（M*N的矩阵，r(A) = N, 列满秩）实数域
double* InvHighMat(double* A, int m, int n);

// 【矩阵】求低阵的广义逆（M*N的矩阵，r(A) = M, 行满秩）实数域
double* InvLowMat(double* A, int m, int n);

// 【矩阵】求和
double** sumM(double** mat1, double** mat2, int row, int col);

// 【矩阵】求和
double *sumM(double *a, double *b, int row, int col);

// 【矩阵】求和 保证input的通道数为1
double sumM(cv::Mat input);

// 【矩阵】比例变换 每个元素乘上一系数
double** scaleM(double** mat, double scale, int row, int col);

// 【矩阵】求差
double** subM(double** mat, double offset, int row, int col);

// 【矩阵】求差
double *subM(double *a, double *b, int row, int col);

// 【矩阵】求迹
double traceM(double** mat, int size);

// 【矩阵】求迹
double traceM(double *m, int n);

// 【矩阵的秩】
int rankM(cv::Mat src);

// 【行列式】值
double detM(cv::Mat src);

// 【行列式】值
double detM(double* mat, int n);

// 【协方差矩阵】求矩阵的协方差矩阵
cv::Mat covMat(cv::Mat A);

// 【矩阵 相关系数矩阵】
cv::Mat corrcoef(cv::Mat A);

// 【矩阵的秩】
int rankM(double *mat, int m, int n);

// 【矩阵】两矩阵相乘 mat1的列数应等于mat2的行数
double** MmulM(double** mat1, double** mat2, int row1, int col1, int col2);

// 【矩阵】两矩阵相乘 mat1的列数应等于mat2的行数
double* MmulM(double *p, double *y, int row1, int col1, int col2);

// 【矩阵】两矩阵相乘 mat1的列数n应等于mat2的行数n   mat1 m*n    mat2 n*k
float* MmulM(float *p0, float *p1, int m, int n, int k);

// 【矩阵】 两矩阵对应元素相乘，mat1的尺寸应该等于mat2的尺寸
float* MmulMEle(float* m1, float* m2, int row, int col);

// 【矩阵】 两矩阵对应元素相乘，mat1的尺寸应该等于mat2的尺寸
double* MmulMEle(double* m1, double* m2, int row, int col);

// 【矩阵】打印矩阵
void printMatrix(double **m, int row, int col);

// 【矩阵】打印矩阵
void printMatrix(int **m, int row, int col);

// 【矩阵】打印矩阵
void printMatrix(double *m, int row, int col);

// 【矩阵】计算对称矩阵的最大的特征值和对应的特征向量
void getMaxMatEigen(double* m, double& eigen, vector<double> &q, int n);

// 【矩阵】计算对称矩阵绝对值最小的特征值和对应的特征向量 必须是对称矩阵！！！
double* getAbsMinMatEigen(double* a, double& eigen, int n);

// 【矩阵】计算实对称矩阵的全部特征值与特征向量
int calEgiensAndVectors(double* a, int n, double *lambda, double** vec);

// 【矩阵】计算 特征值与特征向量
int getEigensAndVecs(cv::Mat input, cv::Mat& eigens, cv::Mat& vectors);

// 【矩阵】计算 特征值与特征向量
//  eigens 和 vectors 应该是已经分配好内存的
int getEigensAndVecs(double *input, int n, double*eigens, double* vectors);

// 与MATLAB中QR分解相关的householder方法 
// 求解正交对称的Householder矩阵H，使Hx = rho*y，其中rho = -sign(x(1))* || x || / || y ||
// 参数说明
// x：列向量
// y：列向量，x和y必须具有相同的维数
cv::Mat householder(cv::Mat x, cv::Mat y);

// 基于Householder变换，将方阵A分解为A = QR，其中Q为正交矩阵，R为上三角阵
// 参数说明
// A：需要进行QR分解的方阵
// Q：分解得到的正交矩阵
// R：分解得到的上三角阵
void QR(cv::Mat A, cv::Mat& Q, cv::Mat& R);

// 分块矩阵 A, B在对角线，其余为0
cv::Mat blkdiag(cv::Mat A, cv::Mat B);

cv::Mat blkdiag(cv::Mat& Ht, cv::Size& rect);

bool QR_householder(cv::Mat A, cv::Mat &Q, cv::Mat& R);


// 【矩阵 分解】对称正定矩阵的Cholesky分解
// 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解 A = L*L.t()
double* cholesky(double* A, int n);

// 【矩阵 分解】对称正定矩阵的Cholesky分解
// 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解 A = L*L.t()
cv::Mat cholesky(cv::Mat A);

// 【矩阵 分解】对称正定矩阵的Cholesky分解
// 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解 A = C.t()*L
cv::Mat cholesky2(cv::Mat A);

// 【矩阵】将矩阵约化为海森博格矩阵 比上三角矩阵还多一斜的矩阵
/*
| x x x x x|
| x x x x x|
| 0 x x x x|
| 0 0 x x x|
| 0 0 0 x x|
*/
double* hessenberg(double* a, int n);

// 【矩阵 广义逆】 伪逆 加号逆
cv::Mat InvMat(cv::Mat src);

// 【矩阵 广义逆】
double* InvMat(double* input, int row, int col);

/************************************************************************/
/* input:
/* a:存放m*n实矩阵A,返回时亦是奇异矩阵
/* m:行数 n：列数
/* u:存放m*m左奇异向量, v:存放n*n右奇异向量
/* eps:给定精度要求,  ka: max(m,n)+1
/* output:
/* 返回值如果为负数，表示迭代了60次，还未求出奇异值；返回值为非负数，正常
/************************************************************************/
int svdDecomp(double *a, int m, int n, double *u, double *v, int ka, double eps=1e-6);

// 奇异值分解 SVD 辅助函数
static void ppp(double a[], double e[], double s[], double v[], int m, int n);

// 奇异值分解 SVD  辅助函数
static void sss(double fg[], double cs[]);

// 【矩阵分解】SVD分解 A = P*delta*Q.t() P是m*r的矩阵 delta是r*r的矩阵，且主对角线元素为A的特征值， Q为n*r的矩阵
void svdDecomp(double* input, int m, int n, double *P, double* delta, double *Q);

// 【矩阵分解】SVD分解 A = P*delta*Q.t() P是m*r的矩阵 delta是r*r的矩阵，且主对角线元素为A的特征值， Q为n*r的矩阵
void svdDecomp(cv::Mat src, cv::Mat& P, cv::Mat& delta, cv::Mat& Q);

// 【矩阵】在输入矩阵的下面插入全1行
void AddOnesRow(cv::Mat& input);

// 【矩阵】点积 计算数组的点积，源数组mat1和mat2的维数必须一致，结果存放在result中
// mat1		M*N
// mat2		M*N
// result	1*N
// 第一步 mat1与mat2中对应元素相乘 得 temp
// 第二步 result(0, i) = temp第i列元素和
bool dotarray(cv::Mat mat1, cv::Mat mat2, cv::Mat& result);


// 与MATLAB的同名函数功能一致，将源数组按rows和cols的值进行扩展，存放与dst中
// 比如原来的矩阵A为	| 1 0 0 |
//				A = | 0 2 0 |
//					| 0 0 3 | 
// 设置rows = 3, cols = 3
// 则结果为
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

// 求实对称阵的全部特征值与特征向量
int cjcbi(double*a, int n, double*v, double eps, int jt);

// 求解矩阵的最大最小值
void maxAndmin(double &bmax, double &bmin, double* data, int col, int row);

// 判断是否是对称矩阵
bool isSymmetry(cv::Mat input);

// 判断是否是正定矩阵
int isPositiveDefinite(cv::Mat input);

// 【条件数】cond
double cond(cv::Mat input);

// Doolittle 分解 LU 分解
bool decompDoolittle(cv::Mat input, cv::Mat& L, cv::Mat& U);

// Crout 分解 LU 分解
bool decompCrout(cv::Mat input, cv::Mat& L_, cv::Mat& U_);

// 【奇异阵】判断矩阵是否为奇异阵
bool isSingular(cv::Mat A);

// 【满秩】判断矩阵是否满秩
bool isFullRank(cv::Mat A);

// 【方阵】判断矩阵是否为方阵
bool isSquare(cv::Mat A);

// 【无穷范数】求矩阵元素绝对值最大值
double infiniteNorm(cv::Mat A);

// 【1范数】求矩阵的1范数 1-范数
double norm_L1(cv::Mat A);

// 【对角线元素最大值】
double maxValueInDiag(cv::Mat input);

// svd分解 与MATLAB功能一致 这里的V和MATLAB中的v是转置关系
bool svd(cv::Mat A, cv::Mat& U, cv::Mat &S, cv::Mat &V, bool useDouble = true);

// 求二维傅里叶变换 double型 输入为单通道 实数; 返回值为复数
cv::Mat fft(cv::Mat input, cv::Mat& mag, cv::Mat& angle);

//// 将矩阵展成1维
//cv::Mat Mat2Vec(cv::Mat input);

// 分块矩阵组合在一起 纵向 
cv::Mat combine2MatV(cv::Mat Up, cv::Mat Down);

// 分块矩阵组合在一起 横向
cv::Mat combine2MatH(cv::Mat Left, cv::Mat Right);

// Hermite转置 input.channel() == 2
cv::Mat hermiteT(cv::Mat input);

// 复数自然指数 
cv::Mat complexExp(cv::Mat X);

// 复数乘法 X,Y 都是双通道的  返回值也是双通道的
cv::Mat complexMul(cv::Mat X, cv::Mat Y);

// 复数乘法 乘以系数 Point2d X * cv::Mat Y
cv::Mat complexScale(cv::Mat Y, cv::Point2d X);

// 复数乘法 逐像素
cv::Mat complexMulEle(cv::Mat X, cv::Mat Y);

// 复数乘法 乘以系数 Point2d X * Point2d Y
cv::Point2d complexMul(cv::Point2d X, cv::Point2d Y);

// 复数矩阵幅值
cv::Mat complexAbsMat(cv::Mat input);

// 复数矩阵幅值
cv::Mat complexAbsMat(cv::Mat input);

// cv::Mat取实部 input双通道
cv::Mat real(cv::Mat input);

// cv::Mat取虚部 input双通道
cv::Mat imag(cv::Mat input);

// 实部和虚部合并为2通道矩阵
cv::Mat merge(cv::Mat real, cv::Mat imag);

// 实数矩阵转复数矩阵 相当于补充一个全零阵
cv::Mat real2complex(cv::Mat input);


// 主成分分析
// 输入： A		---	样本矩阵，每行为一个样本
//		 k		--- 降维至k维
// 输出： pcaA	--- 降维后的k维样本特征向量组成的矩阵，每行一个样本，列数k为降维后的样本特征数
//		 V		--- 主成分分量
void PCATrans(cv::Mat input, int k, cv::Mat& pcaA, cv::Mat& V);

// 展成列向量 按行展开
cv::Mat convert2Vec(cv::Mat input);



// Hadamard 矩阵
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

// toeplitz函数的功能是生成托普利兹（toeplitz）矩阵。
//	托普利兹矩阵的特点是：除第一行、第一列外，其他每个元素都与它左上角的元素相同。
//	调用格式：
//	A = toeplitz(第1列元素数组，第1行元素数组）
cv::Mat toeplitz(cv::Mat c, cv::Mat r);

// hankel函数的功能是生成Hankel矩阵。
//	Hankel矩阵的特点是：除第一列、最后一行外，其他每个元素都与它左下角的元素相同。
//	调用格式：
//	A = hankel(第1列元素数组,最后一行元素数组）
// 注意：最后一行的第1个元素应与第1列的第1个元素相同，否则最后一行的第一个元素将自动改为1列的第1个元素。
cv::Mat hankel(cv::Mat c, cv::Mat r);

// 合并两矩阵 默认为横向合并 左右拼接 左右合并 上下拼接 上下合并
cv::Mat combine2Mat(cv::Mat A, cv::Mat B, bool CmbHor = true);

// 将矩阵中绝对值小于val的元素置零
cv::Mat set2zeroAbsBelowThresh(cv::Mat& input, double val);

// 判断两个矩阵是否相同 两幅图像是否相同
bool isTwoMatEqual(cv::Mat A, cv::Mat B);

// 卷积 高斯核
void convolve_gauss(cv::Mat src, cv::Mat& dst, double sigma, long deriv_type);

// Gaussian filtering and Gaussian derivative filters
void gfilter(cv::Mat src, cv::Mat& dst, double sigma, cv::Mat orders);

// GAUSSIAN DERIVATIVE KERNEL - creates a gaussian deivative kernel.
void gaussiankernel(float sigma, int order, cv::Mat& outputArray);

//// 【最大值】返回矩阵的最大值
//double maxM(cv::Mat input);

// 【最小值】返回矩阵的最小值
double minM(cv::Mat input);

// 【最大值】返回矩阵的最大值所在偏移值
int maxIndex(cv::Mat input);

// 【最小值】返回矩阵的最小值所在偏移值
int minIndex(cv::Mat input);

// 【比较】矩阵与单个数值 返回矩阵与输入input同尺寸，返回cv::Mat数据类型为CV_32S1 
// 参数cmpFlag = CMP_EQU				时，若某位置出cv::Mat元素等于value则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_EQU_OR_GREATER	时，若某位置出cv::Mat元素大于等于value则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_GREATER			时，若某位置出cv::Mat元素大于value则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_EQU_OR_LESS		时，若某位置出cv::Mat元素小于等于value则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_LESS			时，若某位置出cv::Mat元素小于value则，结果cv::Mat该处置为1，否则置为0
cv::Mat cmpMatVSVal(cv::Mat input, double value, int cmpFlag = CMP_EQU);

// 【比较】矩阵与矩阵比较 返回矩阵与两个输入矩阵尺寸，返回cv::Mat数据类型为CV_32S1 
// 参数cmpFlag = CMP_EQU				时，若某位置出A元素等于B元素则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_EQU_OR_GREATER	时，若某位置出A元素大于等于B元素则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_GREATER			时，若某位置出A元素大于B元素则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_EQU_OR_LESS		时，若某位置出A元素小于等于B元素则，结果cv::Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_LESS			时，若某位置出A元素小于B元素则，结果cv::Mat该处置为1，否则置为0
cv::Mat cmpMatVSMat(cv::Mat A, cv::Mat B, int cmpFlag = CMP_EQU);

// 获得子矩阵 get submatrix
cv::Mat getSubMat(cv::Mat src, int r1, int c1, int r2=-1, int c2=-1);

// 生成N阶傅里叶矩阵cv::Mat 双通道  第j行第k列的元素表达式为exp（2πijk/n）
cv::Mat FourierMat(int n);

// 求解矩阵的互相关 
// 矩阵A的互相关是A中不同列的归一化内积的绝对值中的最大值
double CrossCorrelationMat(cv::Mat A);

// 【绝对值】cv::Mat中各元素取绝对值
cv::Mat absMat(cv::Mat input);

// 列单位化
void colNormalized(cv::Mat& input);

// vector<int> 最大L个元素对应的下标
vector<int> maxValuesIndex(cv::Mat input, int L);

// 顺时针转90°
cv::Mat turnRight(cv::Mat input);

// 逆时针转90°
cv::Mat turnLeft(cv::Mat input);

// 矩阵的值 仅适用于那些一行一列的矩阵
double valM_double(cv::Mat input);

// 矩阵的值 仅适用于那些一行一列的矩阵
double valM_int(cv::Mat input);

// 符号矩阵 对矩阵进行sign运算，结果是同样大小的矩阵 值只有+1 和-1
cv::Mat signMat(cv::Mat input);

// 获取某元素的邻域元素
// type 分为4邻域 和 8邻域
vector<cv::Point> adjacentPixels(cv::Mat data, cv::Point p, int type = 8);

// 判断两个矩阵是否相等
bool isEqual(cv::Mat A, cv::Mat B);

// 矩阵长度 若为向量，则为向量长度，若非向量 则 长度为列数
int length(cv::Mat m);


// 单个元素转矩阵
template<typename T>
cv::Mat val2Mat(T val){
	return (Mat_<T>(1, 1) << val);
}

// 取矩阵的首元素 单元素矩阵取值
template<typename T>
T Mat2val(cv::Mat input){
	return (input.ptr<T>(0)[0]);
}

// 【子集 截断 roi range】vector中截取某一区域 子向量
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

// 求两矩阵的较小值构成的矩阵
template<typename T> 
cv::Mat minMat(cv::Mat A, cv::Mat B){
	CV_Assert(A.size() == B.size());
	cv::Mat result = Mat::zeros(A.size(), A.type());
	for (int i = 0; i < (int)A.total(); i++){
		result.ptr<T>(0)[i] = min(A.ptr<T>(0)[i], B.ptr<T>(0)[i]);
	}
	return result;
}

// 求矩阵与某值的 较大值构成的矩阵
template<typename T>
cv::Mat maxMat(cv::Mat A, T val){
	cv::Mat result = Mat::zeros(A.size(), A.type());
	for (int i = 0; i < (int)A.total(); i++){
		result.ptr<T>(0)[i] = max(A.ptr<T>(0)[i], val);
	}
	return result;
}

// 获得子矩阵 某些元素 给出起始元素下标 和最后元素下标 
// 子矩阵为列向量 
template<typename T>
cv::Mat subMat(cv::Mat x, int start, int end){
	Mat r = Mat_<T>(end - start + 1, 1);
	for (int i = 0; i < end - start + 1; i++){
		r.ptr<T>(0)[i] = x.ptr<double>(0)[start + i];
	}
	return r;
}




// 混排，随机打乱顺序
//RNG rng(12345); //随机数产生器  
//int i, sampleCount = rng.uniform(1, 1001);
//cv::Mat points(sampleCount, 1, CV_32FC2), labels;   //产生的样本数，实际上为2通道的列向量，元素类型为Point2f  
//randShuffle(points, 1, &rng);   //因为要聚类，所以先随机打乱points里面的点，注意points和pointChunk是共用数据的。  