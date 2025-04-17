#pragma once
#include<stdio.h>
#include<math.h>
#include<malloc.h>
#include<limits>
#include<vector>
#include<iomanip>
#include<algorithm>
#include<numeric>
#include<functional>
#include<opencv2/opencv.hpp>
#include"MyFloat.h"
#include"MyInteger.h"
#include<set>

#define PI_2 (CV_PI*2)

using namespace std;

// 产生一个随机数列 a 到 b，步长为1，顺序打乱
vector<int> randomVecInt(int b, int a = 0);

// double* 转换为 vector
vector<double> pdouble2vector(double* input, int n);

// 【向量】初始化
vector<double> initVectord(int len);

// 【向量】初始化 等差数列
vector<double> linspace(double minV, double maxV, int N);

//  【向量】初始化 logspace(a, b, n); 创建行向量 第一个元素为10^a, 最后一个元素为10^b, 形成总数为n个元素的等比数列
vector<double> logspace(double a, double b, int n);

// 【向量】初始化 产生一个函数delta， 在n0到n2的地方除了n1时值为1，其余都为0
vector<double> impseq(int n0, int n1, int n2);


// 【向量】返回最小元素值
template<typename T>
int minIndexV(std::vector<T> vec)
{
	std::vector<T>::iterator where = std::min_element(vec.begin(), vec.end());
	return (where - vec.begin());
}

// 【向量】初始化
float* initVectorf(int len);

// 【向量】初始化
cv::Mat initVectoriMat(int minValue, int maxValue, int step = 1);

// 【向量】初始化
double* initVector(int len);

// 【向量】随机向量
vector<double> initRandomVector(int len);

// 【向量】初始化
double* initVec(double minV, double step, double maxV);

// 【向量】初始化
vector<double> initVector(double minV, double step, double maxV);

// 【向量】初始化
int* initVectorInt(int len);

//// 【向量】全体赋值
//bool setAllTo(vector<double>& input, double value=0.0);

// 【向量】全体赋值
template<typename T>
bool setAllTo(std::vector<T>& input, T value = 0){
	std::vector<T>::iterator it = input.begin();
	std::vector<T>::iterator iend = input.end();

	while (it != iend){
		*it = value;
		it++;
	}
	return true;
}

// 【向量】初始化 升序向量
int* initAscendingVectorInt(int len, int initValue, int step);

// 【向量】降采样
vector<double> downSampling(vector<double> src, int targetCount);

// 【向量】BOOL型
bool* initVectorBool(int len);

// 【向量】计算列向量元素方和根
double getVecSquareAndRoot(double *v, int len);

// 【向量】2范数
double norm2(double *v, int len);

// 【范数】向量的1范数 绝对值和 
double normV_L1(cv::Mat vec);

// 【向量】单位化
void normalize(double *v, int len);

// 【向量】元素求和
double sumV(double *v, int len);

// 【向量】元素求和
double sumV(vector<double> input);

// 【连乘积】元素连乘积
double multV(vector<double> input);

// 【向量】均值
double meanV(double *v, int len);

// 【向量】均值
double meanV(vector<double> input);

// 【乘积加和】两向量 默认是从0元素开始到最后一个元素 对应元素相乘并相加
double sumProduct(vector<double> A, vector<double> B, int len = -1, int startOffset = 0);

// 【向量】协方差矩阵 返回协方差矩阵
cv::Mat covMat(vector<double> X, vector<double> Y);

// 【向量】协方差
double covV(cv::Mat X, cv::Mat Y);

// 【向量】协方差
double covV(vector<double> input);

// 【向量】协方差
double covV(vector<double> X, vector<double> Y);

// 【向量】方差
double varV(cv::Mat A);

// 【向量】标准差
double stdVector(double *v, int len, int type=1);

// 【标准差】
double stdVector(vector<double> x0, vector<double> x1);

// 【标准差】
double stdVector(vector<double> data);

// 求相关系数
double coefV(cv::Mat X, cv::Mat Y);

// 【向量】scale尺度变换 得到新向量
double* scaleV(double* vec, int len, double scale);

// 【向量】幂变换
double* powV(double* vec, int len, double index);

// 【向量】幂变换
vector<double> powV(vector<double> input, double index);

// 【向量】各元素平方的平均值
double meanQuadEle(double *vec, int len);

// 【向量】元素平方和
double quadSumV(double* vec, int len);

// 【向量】元素平方和
double quadSumV(vector<double> v);

//  【向量】两向量对应元素相加 result = vec1 + vec2
double* sumV(double *vec1, double* vec2, int len);

//  【向量】两向量对应元素相加 result = vec1 + vec2
vector<double> addV(vector<double> vec1, vector<double> vec2);

//  【向量】两向量对应元素相减 result = vec1 - vec2
double* subV(double *vec1, double* vec2, int len);

//  【向量】每个元素减去一个定值
double* subV(double *vec, double delta, int len);

//  【向量】每个元素减去vec的平均值
double* divV(double *vec, int len);

// 【向量】两向量元素对应做减法
vector<double> subtractV(vector<double> A, vector<double> B);

// 【向量】向量减去一个定值
vector<double> subtractV(vector<double> x, double val);

// 【向量】两向量元素对应做除法
vector<double> divisionV(vector<double> A, vector<double> B);

// 【向量】向量元素符号
vector<double> signV(vector<double> input);

//  【向量】两向量对应元素相乘 result = vec1 * vec2
double* mulV(double *vec1, double* vec2, int len);

// 【乘法】两向量对应元素相乘 乘法
vector<double> mulV(vector<double> vec1, vector<double> vec2);

// 【向量】点乘
double dotV(double* vec1, double *vec2, int len);

// 【向量】各元素取模
void absV(double* vec, int len);

//【向量】向量乘比例系数 乘法
vector<double> scaleV(vector<double> input, double scale, double offset = 0.0);

// 【向量】各元素取sin值
vector<double> sinV(vector<double> input);

// 【向量】各元素取sinc值
vector<double> sincV(vector<double> input);

// 【向量】各元素取cos值
vector<double> cosV(vector<double> input);

// 【log】各元素取log值
vector<double> logV(vector<double> input);

// 【sqrt】各元素取平方根
vector<double> sqrtV(vector<double> input);

// 【ln】各元素取ln值
vector<double> lnV(vector<double> input);

// 【exp】 各元素取exp值
vector<double> expV(vector<double> input);

// 【向量】两向量间的欧氏距离
double distV(double* vec1, double *vec2, int len);

// 【向量】相反数
void oppositeNumVec(float *vec, int len);

// 【向量】返回最大元素值
double maxV(double* vec, int len);

// 【向量】返回最大元素值
float maxV(float* vec, int len);

// 【向量】返回最大元素值
double maxV(vector<double> vec);

// 【向量】返回最小元素值
double minV(vector<double> vec);

// 【向量】返回最小元素值
float minV(float* vec, int len);

// 【向量】打印向量double
void printVector(vector<double> vec);

// 【向量】打印向量 double
void printVector(vector<double> vec, int len);

// 【向量】打印向量
void printVector(double* vec, int len);

// 【向量】打印向量 float
void printVector(float* vec, int len);

// 【向量】打印向量 int
void printVector(int* vec, int len);

// 【向量】打印向量
void printVector(vector<float> vec);

// 【最大值】pair<int, int> useSecond = true 是判断第二个元素的最大值,那么返回值就返回第二个元素的最大值
int maxPair(vector<pair<int, int>>input, int& otherResult, bool useSecond = true);

// 【最小值】pair<int, int> useSecond = true 是判断第二个元素的最小值,那么返回值就返回第二个元素的最小值
int minPair(vector<pair<int, int>>input, int& otherResult, bool useSecond = true);

// 【最大最小值】求vector 的最大最小值
void minMaxVector(vector<double> input, double& minV, double& maxV, int& minIndex, int& maxIndex, int N);

// 【最大最小值】返回值[0]为最小值 [1]为最大值
double* minMaxVector(double* input, int len);

// 【最大值】
double maxValV(double* input, int len);

// 【最大值】序号
int maxIndexV(double* input, int len);

// 【最小值】
double minValV(double* input, int len);

// 【最小值】序号
int minIndexV(double* input, int len);

// 【交换】两个值
void swap2int(int& a, int& b);

// 【交换】两个值
void swap2float(float& a, float& b);

// 【返回第一组末尾序号】这样一个问题 一列float 共m个，从小到大排列， 前n个数相差不大，从第n+1个开始，后面比前面的变化很多，求n
int numGrp1st(float* vec, int len);

// 【vector<double>】向量求和
double sumVector(vector<double> v);

// 【vector<double>】向量取平均
double meanVector(vector<double> v);

// 【vector<double>】最大最小值
void maxmin(vector<double>a, double &maxv, double &minv);

// 【复制】数组
void copyV(double* src, double *dst, int len);

// 【复制】数组
vector<double> copyV(vector<double> src);

// 【仿真】 生成带噪声的余弦信号
double* createNoiCos(double A, double freq, double phi, double C, double noiLev, int N);

// 【反序】逆序 倒序输出
bool reverseArray(vector<double> &input);

// 【偶序列】获得偶序列
double* getEvenArray(double* input, int n);

// 【偶序列】获得偶序列
vector<double> getEvenArray(vector<double> input);

// 【奇序列】获得奇序列
double* getOddArray(double* input, int n);

// 【奇序列】获得奇数序列
vector<double> getOddArray(vector<double> input);

// 【查找】单调递增列最接近的元素序号
int getNearestEleIndex(float* data, float target, int len);

struct COMPLEX{
	double* real;
	double* imag;
	int n;

	COMPLEX(const int& n_){
		n = n_;
		real = initVector(n);
		imag = initVector(n);
	}
};

struct POLAR_COMPLEX{
	double* ampl;
	double* angl;
	int n;

	POLAR_COMPLEX(const int& n_){
		n = n_;
		ampl = initVector(n);
		angl = initVector(n);
	}
};

// 【补零】补为2的整数次幂个元素
bool adjustToIntegralPowerOfTwo(vector<double>& input);

// 【FFT】
COMPLEX fftV(double* input, int n);

// 【FFT】
COMPLEX fftV(vector<double> input, int N);

// 复数变换为极坐标
POLAR_COMPLEX convert2Polar(COMPLEX input);

// 实现离散卷积 x(n) 和 h(n) 的卷积 卷积结果元素有M+N-1个
vector<double> discreteConv(vector<double> x, vector<double> h);

// 变量替换
// vec是初始变量
// index是待换入值的坐标序列
// data是待换入的值
void substitute(cv::Mat vec, vector<int> index, cv::Mat data);

// 一维插值 3次样条插值
// ddy1 是第一点的二阶导数
// ddyn 是最后一点的二阶导数
vector<double> interpSpline(vector<double> x, vector<double> y, vector<double> x_new, double ddyStart = 0, double ddyEnd = 0);

// Lagrange插值 拉格朗日插值
// x与y为已知的插值点及其函数值
// x0为要求的插值点的x坐标值
vector<double> interpLagrange(vector<double> x, vector<double> y, vector<double> x0);

// Newton插值 牛顿插值
// x与y为已知的插值点及其函数值
// x0为要求的插值点的x坐标值，nn为Newton插值多项式的阶数
vector<double> interpNewton(vector<double> x, vector<double> y, vector<double> x0, int nn);

// 多项式求值
// coef 为多项式系数，从高次项到低次项排列
// x是自变量
vector<double> polyval(vector<double> coef, vector<double> x);

// 一维傅里叶变换
double* FFT(double* x, int N);

// 离散信号的自相关函数

// 离散随机信号的互相关函数
vector<double> xcorr(vector<double> x, vector<double> y);

// 离散随机信号的互协方函数
vector<double> xcov(vector<double> x, vector<double> y);

// X = F.*2
//[F, E] = log2(X) for each element of the real array X, returns an
//array F of real numbers, usually in the range 0.5 <= abs(F) < 1,
//and an array E of integers, so that X = F .* 2.^E.Any zeros in X
//produce F = 0 and E = 0.  This corresponds to the ANSI C function
//frexp() and the IEEE floating point standard function logb().
vector<pair<double, int>> log2(vector<double> input);

// 卷积运算
vector<double> convV(vector<double> a, vector<double> b);

// 求两个一维数组之间的差异 一维数组可能长度不一样
double dist2Vec(vector<double> a, vector<double> b);
 
// 【向量】归一化 对自身数据进行操作
void NormalizeV(vector<double>& vec, double minValue  = 0, double maxValue = 1.0);

// 【向量】归一化
vector<double> normalizeV(vector<double> vec, double minValue = 0, double maxValue = 1.0);

// 阈值化处理，小于某个值的就置零
vector<double> thresholdV(vector<double> signal, double thresh);

// 指定元素赋值 输入为向量 Mat的指定元素赋值 某些元素赋值
void assignByIndexs(cv::Mat& input, cv::Mat values, vector<int> idxs);

// 指定元素赋值 输入为向量 Mat的指定元素赋值 某些元素赋值
void assignByIndexs(cv::Mat& input, double val, vector<int> idxs);

// 生成正弦信号
// phi0为弧度值 b是直流分量
vector<double> sinSignal(double T, double A, double phi0, double b, int len, bool useCos = false);

// 1D傅里叶变换 返回值为Mat通道1为实部 通道2为虚部 一维傅里叶变换
cv::Mat fft(vector<double> signal);

// 1D傅里叶变换 取幅值 FFT 一维傅里叶变换
vector<double> fft_Amplitude(vector<double> signal);

// 1D傅里叶变换 取相位 FFT 一维傅里叶变换
vector<double> fft_Phase(vector<double> signal);


// 【子集 截断 roi range】vector中截取某一区域 子向量
template<typename T>
vector<T> subVec(vector<T> input, int b, int a = 0)
{	
	//CV_Assert(b >= a);
	if (b < a || a <0 || a >= input.size() || b >= input.size())
	{
		printf("subVec Error! a = %d  b = %d size = %d\n", a, b, input.size());
		return vector<T>();
	}
	int n = b - a + 1;
	vector<T> result(n);
	for (int i = 0; i < n; i++)
	{
		result[i] = input[a + i];
	}
	return result;
}



// 查找大于a的所有元素序号 查找 筛选 选取 某些
vector<int> findGTR(cv::Mat input, double a);

// 查找小于a的所有元素序号 查找 筛选 选取 某些
vector<int> findLESS(cv::Mat input, double a);

// 找峰值点 interval是最大容许 极值间隔
vector<int> findPeaks(vector<double> data, int interval);

// 将矩阵转换为vector<double>
vector<double> Mat2Vec(cv::Mat input);