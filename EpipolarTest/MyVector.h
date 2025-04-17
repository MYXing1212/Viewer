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

// ����һ��������� a �� b������Ϊ1��˳�����
vector<int> randomVecInt(int b, int a = 0);

// double* ת��Ϊ vector
vector<double> pdouble2vector(double* input, int n);

// ����������ʼ��
vector<double> initVectord(int len);

// ����������ʼ�� �Ȳ�����
vector<double> linspace(double minV, double maxV, int N);

//  ����������ʼ�� logspace(a, b, n); ���������� ��һ��Ԫ��Ϊ10^a, ���һ��Ԫ��Ϊ10^b, �γ�����Ϊn��Ԫ�صĵȱ�����
vector<double> logspace(double a, double b, int n);

// ����������ʼ�� ����һ������delta�� ��n0��n2�ĵط�����n1ʱֵΪ1�����඼Ϊ0
vector<double> impseq(int n0, int n1, int n2);


// ��������������СԪ��ֵ
template<typename T>
int minIndexV(std::vector<T> vec)
{
	std::vector<T>::iterator where = std::min_element(vec.begin(), vec.end());
	return (where - vec.begin());
}

// ����������ʼ��
float* initVectorf(int len);

// ����������ʼ��
cv::Mat initVectoriMat(int minValue, int maxValue, int step = 1);

// ����������ʼ��
double* initVector(int len);

// ���������������
vector<double> initRandomVector(int len);

// ����������ʼ��
double* initVec(double minV, double step, double maxV);

// ����������ʼ��
vector<double> initVector(double minV, double step, double maxV);

// ����������ʼ��
int* initVectorInt(int len);

//// ��������ȫ�帳ֵ
//bool setAllTo(vector<double>& input, double value=0.0);

// ��������ȫ�帳ֵ
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

// ����������ʼ�� ��������
int* initAscendingVectorInt(int len, int initValue, int step);

// ��������������
vector<double> downSampling(vector<double> src, int targetCount);

// ��������BOOL��
bool* initVectorBool(int len);

// ������������������Ԫ�ط��͸�
double getVecSquareAndRoot(double *v, int len);

// ��������2����
double norm2(double *v, int len);

// ��������������1���� ����ֵ�� 
double normV_L1(cv::Mat vec);

// ����������λ��
void normalize(double *v, int len);

// ��������Ԫ�����
double sumV(double *v, int len);

// ��������Ԫ�����
double sumV(vector<double> input);

// �����˻���Ԫ�����˻�
double multV(vector<double> input);

// ����������ֵ
double meanV(double *v, int len);

// ����������ֵ
double meanV(vector<double> input);

// ���˻��Ӻ͡������� Ĭ���Ǵ�0Ԫ�ؿ�ʼ�����һ��Ԫ�� ��ӦԪ����˲����
double sumProduct(vector<double> A, vector<double> B, int len = -1, int startOffset = 0);

// ��������Э������� ����Э�������
cv::Mat covMat(vector<double> X, vector<double> Y);

// ��������Э����
double covV(cv::Mat X, cv::Mat Y);

// ��������Э����
double covV(vector<double> input);

// ��������Э����
double covV(vector<double> X, vector<double> Y);

// ������������
double varV(cv::Mat A);

// ����������׼��
double stdVector(double *v, int len, int type=1);

// ����׼�
double stdVector(vector<double> x0, vector<double> x1);

// ����׼�
double stdVector(vector<double> data);

// �����ϵ��
double coefV(cv::Mat X, cv::Mat Y);

// ��������scale�߶ȱ任 �õ�������
double* scaleV(double* vec, int len, double scale);

// ���������ݱ任
double* powV(double* vec, int len, double index);

// ���������ݱ任
vector<double> powV(vector<double> input, double index);

// ����������Ԫ��ƽ����ƽ��ֵ
double meanQuadEle(double *vec, int len);

// ��������Ԫ��ƽ����
double quadSumV(double* vec, int len);

// ��������Ԫ��ƽ����
double quadSumV(vector<double> v);

//  ����������������ӦԪ����� result = vec1 + vec2
double* sumV(double *vec1, double* vec2, int len);

//  ����������������ӦԪ����� result = vec1 + vec2
vector<double> addV(vector<double> vec1, vector<double> vec2);

//  ����������������ӦԪ����� result = vec1 - vec2
double* subV(double *vec1, double* vec2, int len);

//  ��������ÿ��Ԫ�ؼ�ȥһ����ֵ
double* subV(double *vec, double delta, int len);

//  ��������ÿ��Ԫ�ؼ�ȥvec��ƽ��ֵ
double* divV(double *vec, int len);

// ��������������Ԫ�ض�Ӧ������
vector<double> subtractV(vector<double> A, vector<double> B);

// ��������������ȥһ����ֵ
vector<double> subtractV(vector<double> x, double val);

// ��������������Ԫ�ض�Ӧ������
vector<double> divisionV(vector<double> A, vector<double> B);

// ������������Ԫ�ط���
vector<double> signV(vector<double> input);

//  ����������������ӦԪ����� result = vec1 * vec2
double* mulV(double *vec1, double* vec2, int len);

// ���˷�����������ӦԪ����� �˷�
vector<double> mulV(vector<double> vec1, vector<double> vec2);

// �����������
double dotV(double* vec1, double *vec2, int len);

// ����������Ԫ��ȡģ
void absV(double* vec, int len);

//�������������˱���ϵ�� �˷�
vector<double> scaleV(vector<double> input, double scale, double offset = 0.0);

// ����������Ԫ��ȡsinֵ
vector<double> sinV(vector<double> input);

// ����������Ԫ��ȡsincֵ
vector<double> sincV(vector<double> input);

// ����������Ԫ��ȡcosֵ
vector<double> cosV(vector<double> input);

// ��log����Ԫ��ȡlogֵ
vector<double> logV(vector<double> input);

// ��sqrt����Ԫ��ȡƽ����
vector<double> sqrtV(vector<double> input);

// ��ln����Ԫ��ȡlnֵ
vector<double> lnV(vector<double> input);

// ��exp�� ��Ԫ��ȡexpֵ
vector<double> expV(vector<double> input);

// �����������������ŷ�Ͼ���
double distV(double* vec1, double *vec2, int len);

// ���������෴��
void oppositeNumVec(float *vec, int len);

// ���������������Ԫ��ֵ
double maxV(double* vec, int len);

// ���������������Ԫ��ֵ
float maxV(float* vec, int len);

// ���������������Ԫ��ֵ
double maxV(vector<double> vec);

// ��������������СԪ��ֵ
double minV(vector<double> vec);

// ��������������СԪ��ֵ
float minV(float* vec, int len);

// ����������ӡ����double
void printVector(vector<double> vec);

// ����������ӡ���� double
void printVector(vector<double> vec, int len);

// ����������ӡ����
void printVector(double* vec, int len);

// ����������ӡ���� float
void printVector(float* vec, int len);

// ����������ӡ���� int
void printVector(int* vec, int len);

// ����������ӡ����
void printVector(vector<float> vec);

// �����ֵ��pair<int, int> useSecond = true ���жϵڶ���Ԫ�ص����ֵ,��ô����ֵ�ͷ��صڶ���Ԫ�ص����ֵ
int maxPair(vector<pair<int, int>>input, int& otherResult, bool useSecond = true);

// ����Сֵ��pair<int, int> useSecond = true ���жϵڶ���Ԫ�ص���Сֵ,��ô����ֵ�ͷ��صڶ���Ԫ�ص���Сֵ
int minPair(vector<pair<int, int>>input, int& otherResult, bool useSecond = true);

// �������Сֵ����vector �������Сֵ
void minMaxVector(vector<double> input, double& minV, double& maxV, int& minIndex, int& maxIndex, int N);

// �������Сֵ������ֵ[0]Ϊ��Сֵ [1]Ϊ���ֵ
double* minMaxVector(double* input, int len);

// �����ֵ��
double maxValV(double* input, int len);

// �����ֵ�����
int maxIndexV(double* input, int len);

// ����Сֵ��
double minValV(double* input, int len);

// ����Сֵ�����
int minIndexV(double* input, int len);

// ������������ֵ
void swap2int(int& a, int& b);

// ������������ֵ
void swap2float(float& a, float& b);

// �����ص�һ��ĩβ��š�����һ������ һ��float ��m������С�������У� ǰn�������󣬴ӵ�n+1����ʼ�������ǰ��ı仯�ܶ࣬��n
int numGrp1st(float* vec, int len);

// ��vector<double>���������
double sumVector(vector<double> v);

// ��vector<double>������ȡƽ��
double meanVector(vector<double> v);

// ��vector<double>�������Сֵ
void maxmin(vector<double>a, double &maxv, double &minv);

// �����ơ�����
void copyV(double* src, double *dst, int len);

// �����ơ�����
vector<double> copyV(vector<double> src);

// �����桿 ���ɴ������������ź�
double* createNoiCos(double A, double freq, double phi, double C, double noiLev, int N);

// ���������� �������
bool reverseArray(vector<double> &input);

// ��ż���С����ż����
double* getEvenArray(double* input, int n);

// ��ż���С����ż����
vector<double> getEvenArray(vector<double> input);

// �������С����������
double* getOddArray(double* input, int n);

// �������С������������
vector<double> getOddArray(vector<double> input);

// �����ҡ�������������ӽ���Ԫ�����
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

// �����㡿��Ϊ2���������ݸ�Ԫ��
bool adjustToIntegralPowerOfTwo(vector<double>& input);

// ��FFT��
COMPLEX fftV(double* input, int n);

// ��FFT��
COMPLEX fftV(vector<double> input, int N);

// �����任Ϊ������
POLAR_COMPLEX convert2Polar(COMPLEX input);

// ʵ����ɢ��� x(n) �� h(n) �ľ�� ������Ԫ����M+N-1��
vector<double> discreteConv(vector<double> x, vector<double> h);

// �����滻
// vec�ǳ�ʼ����
// index�Ǵ�����ֵ����������
// data�Ǵ������ֵ
void substitute(cv::Mat vec, vector<int> index, cv::Mat data);

// һά��ֵ 3��������ֵ
// ddy1 �ǵ�һ��Ķ��׵���
// ddyn �����һ��Ķ��׵���
vector<double> interpSpline(vector<double> x, vector<double> y, vector<double> x_new, double ddyStart = 0, double ddyEnd = 0);

// Lagrange��ֵ �������ղ�ֵ
// x��yΪ��֪�Ĳ�ֵ�㼰�亯��ֵ
// x0ΪҪ��Ĳ�ֵ���x����ֵ
vector<double> interpLagrange(vector<double> x, vector<double> y, vector<double> x0);

// Newton��ֵ ţ�ٲ�ֵ
// x��yΪ��֪�Ĳ�ֵ�㼰�亯��ֵ
// x0ΪҪ��Ĳ�ֵ���x����ֵ��nnΪNewton��ֵ����ʽ�Ľ���
vector<double> interpNewton(vector<double> x, vector<double> y, vector<double> x0, int nn);

// ����ʽ��ֵ
// coef Ϊ����ʽϵ�����Ӹߴ���ʹ�������
// x���Ա���
vector<double> polyval(vector<double> coef, vector<double> x);

// һά����Ҷ�任
double* FFT(double* x, int N);

// ��ɢ�źŵ�����غ���

// ��ɢ����źŵĻ���غ���
vector<double> xcorr(vector<double> x, vector<double> y);

// ��ɢ����źŵĻ�Э������
vector<double> xcov(vector<double> x, vector<double> y);

// X = F.*2
//[F, E] = log2(X) for each element of the real array X, returns an
//array F of real numbers, usually in the range 0.5 <= abs(F) < 1,
//and an array E of integers, so that X = F .* 2.^E.Any zeros in X
//produce F = 0 and E = 0.  This corresponds to the ANSI C function
//frexp() and the IEEE floating point standard function logb().
vector<pair<double, int>> log2(vector<double> input);

// �������
vector<double> convV(vector<double> a, vector<double> b);

// ������һά����֮��Ĳ��� һά������ܳ��Ȳ�һ��
double dist2Vec(vector<double> a, vector<double> b);
 
// ����������һ�� ���������ݽ��в���
void NormalizeV(vector<double>& vec, double minValue  = 0, double maxValue = 1.0);

// ����������һ��
vector<double> normalizeV(vector<double> vec, double minValue = 0, double maxValue = 1.0);

// ��ֵ������С��ĳ��ֵ�ľ�����
vector<double> thresholdV(vector<double> signal, double thresh);

// ָ��Ԫ�ظ�ֵ ����Ϊ���� Mat��ָ��Ԫ�ظ�ֵ ĳЩԪ�ظ�ֵ
void assignByIndexs(cv::Mat& input, cv::Mat values, vector<int> idxs);

// ָ��Ԫ�ظ�ֵ ����Ϊ���� Mat��ָ��Ԫ�ظ�ֵ ĳЩԪ�ظ�ֵ
void assignByIndexs(cv::Mat& input, double val, vector<int> idxs);

// ���������ź�
// phi0Ϊ����ֵ b��ֱ������
vector<double> sinSignal(double T, double A, double phi0, double b, int len, bool useCos = false);

// 1D����Ҷ�任 ����ֵΪMatͨ��1Ϊʵ�� ͨ��2Ϊ�鲿 һά����Ҷ�任
cv::Mat fft(vector<double> signal);

// 1D����Ҷ�任 ȡ��ֵ FFT һά����Ҷ�任
vector<double> fft_Amplitude(vector<double> signal);

// 1D����Ҷ�任 ȡ��λ FFT һά����Ҷ�任
vector<double> fft_Phase(vector<double> signal);


// ���Ӽ� �ض� roi range��vector�н�ȡĳһ���� ������
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



// ���Ҵ���a������Ԫ����� ���� ɸѡ ѡȡ ĳЩ
vector<int> findGTR(cv::Mat input, double a);

// ����С��a������Ԫ����� ���� ɸѡ ѡȡ ĳЩ
vector<int> findLESS(cv::Mat input, double a);

// �ҷ�ֵ�� interval��������� ��ֵ���
vector<int> findPeaks(vector<double> data, int interval);

// ������ת��Ϊvector<double>
vector<double> Mat2Vec(cv::Mat input);