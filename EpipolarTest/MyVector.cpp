#include"stdafx.h"
#include"MyVector.h"

using namespace cv;

// ����һ��������� a �� b������Ϊ1��˳����� ע�� ǰ��Ĳ���Ӧ�ñȺ���Ĳ�����
vector<int> randomVecInt(int b, int a){
	CV_Assert(b >= a);
	vector<int> result;
	for (int i = a; i <= b; i++){
		result.push_back(i);
	}
	srand((unsigned int)time(NULL));
	random_shuffle(result.begin(), result.end());
	return result;
}


// double* ת��Ϊ vector
vector<double> pdouble2vector(double* input, int n){
	vector<double> result;
	result.reserve(n);
	result.insert(result.begin(), &input[0], &input[n]);
	return result;
}

// ����������ʼ��
vector<double> initVectord(int len){
	vector<double> result(len);
	fill(result.begin(), result.end(), 0.0);
	return result;
}

// ����������ʼ��
double* initVector(int len){
	double* result;
	result = (double*)malloc(sizeof(double)*len);
	for (int i = 0; i < len; i++)
		result[i] = 0.0;
	return result;
}

// ����������ʼ��
float* initVectorf(int len){
	float* result;
	result = (float*)malloc(sizeof(float)*len);
	for (int i = 0; i < len; i++)
		result[i] = 0.0;
	return result;
}

// ����������ʼ��
Mat initVectoriMat(int minValue, int maxValue, int step /*= 1*/){
	int len = (maxValue - minValue) / step + 1;
	int *data = initAscendingVectorInt(len, minValue, step);
	return Mat(len, 1, CV_32SC1, data);
}

// ����������ʼ�� ���Կռ�
vector<double> linspace(double minV, double maxV, int N){
	vector<double> result;
	double step = (maxV - minV) / (double)(N-1);
	for (int i = 0; i < N; i++){
		result.push_back(minV + i*step);
	}
	return result;
}

//  ����������ʼ�� logspace(a, b, n); ���������� ��һ��Ԫ��Ϊ10^a, ���һ��Ԫ��Ϊ10^b, �γ�����Ϊn��Ԫ�صĵȱ�����
vector<double> logspace(double a, double b, int n){
	vector<double> result;
	CV_Assert(n >= 1);
	double scale = pow(pow(10, b) / pow(10, a), 1 / (double)(n-1));
	double tmp = pow(10, a);
	for (int i = 0; i < n; i++){
		result.push_back(tmp);
		tmp *= scale;
	}
	return result;
}

// ����������ʼ�� ����һ������delta�� ��n0��n2�ĵط�����n1ʱֵΪ1�����඼Ϊ0
vector<double> impseq(int n0, int n1, int n2){
	vector<double> result;
	CV_Assert(n1 >= n0 && n1 <= n2);
	for (int i = n0; i <= n2; i++){
		if (i == n1)
			result.push_back(1.0);
		else
			result.push_back(0.0);
	}
	return result;
}

// ����������ʼ��
double* initVec(double minV, double step, double maxV){
	int n = (int)((maxV - minV) / step) + 1;
	double* result = initVector(n);
	for (int i = 0;; i++){
		double tmp = minV + i*step;
		if (tmp > maxV)
			break;
		else
			result[i] = tmp;

	}
	return result;
}

// ���������������
vector<double> initRandomVector(int len){
	CV_Assert(len >= 1);
	static int seed = 0;
	if (!seed){
		srand((unsigned)time(NULL));
		seed++;
	}
	vector<double> result;
	rndDouble();
	for (int i = 0; i < len; i++){
		result.push_back(rndDouble());
	}
	return result;
}

// ����������ʼ��
vector<double> initVector(double minV, double step, double maxV){
	vector<double> result;
	for (int i = 0; ; i++){
		double tmp = minV + i*step;
		if (tmp >maxV)
			break;
		else
			result.push_back(tmp);

	}
	return result;
}

// ����������ʼ��
int* initVectorInt(int len){
	int* result = new int[len];
	for (int i = 0; i < len; i++)
		result[i] = 0;
	return result;
}

//// ��������ȫ�帳ֵ
//bool setAllTo(vector<double>& input, double value){
//	vector<double>::iterator it = input.begin();
//	vector<double>::iterator iend = input.end();
//	
//	while (it != iend){
//		*it = value;
//		it++;
//	}
//	return true;
//}

// ����������ʼ�� ��������
int* initAscendingVectorInt(int len, int initValue, int step){
	int* result = new int[len];
	for (int i = 0; i < len; i++){
		result[i] = initValue + i*step;
	}
	return result;
}

// ��������������
vector<double> downSampling(vector<double> src, int targetCount){
	double step = floor(src.size() / (double)targetCount);
	vector<double> result;
	int i = 0;
	while (i<targetCount){
		result.push_back((src[(int)(step*i)]));
		i++;
	}
	return result;
}

// ��������BOOL��
bool* initVectorBool(int len){
	bool* result = new bool[len];
	for (int i = 0; i < len; i++)
		result[i] = false;
	return result;
}

// ������������������Ԫ�ط��͸�
double getVecSquareAndRoot(double *v, int len){
	double result = 0;
	for (int i = 0; i < len; i++)
		result += v[i] * v[i];
	return sqrt(result);
}

// ��������2����
double norm2(double *v, int len){
	return getVecSquareAndRoot(v, len);
}

// ��������������1����		����ֵ��
double normV_L1(Mat vec){
	CV_Assert(vec.rows == 1 || vec.cols == 1);
	return norm(vec, NORM_L1);
}

// ����������λ��
void normalize(double *v, int len){
	double n = norm2(v, len);
	printf("n = %lf\n", n);
	for (int i = 0; i < len; i++){
		v[i] = v[i] / n;
	}
}

// ��������Ԫ�����
double sumV(double *v, int len){
	double result = 0;
	for (int i = 0; i < len; i++)
		result += v[i];
	return result;
}

// ��������Ԫ�����
double sumV(vector<double> input){	
	vector<double>::iterator it = input.begin();
	vector<double>::iterator iend = input.end();
	double result = 0;
	while (it != iend){
		result += *it;
		it++;
	}
	return result;
}

// �����˻���Ԫ�����˻�
double multV(vector<double> input){
	return accumulate(input.begin(), input.end(), 1, multiplies<double>());
}

// ����������ֵ
double meanV(vector<double> input){
	CV_Assert(input.size() > 0);	
	return (sumV(input) / (double)input.size());
}

// ���˻��Ӻ͡������� Ĭ���Ǵ�0Ԫ�ؿ�ʼ�����һ��Ԫ�� ��ӦԪ����˲����
double sumProduct(vector<double> A, vector<double> B, int len/* = -1*/, int startOffset/* = 0*/){
	if (len == -1){
		CV_Assert(A.size() == B.size());
		double sum = 0.0;
		for (int i = startOffset; i < (int)A.size(); i++){
			sum += A[i] * B[i];
		}
		return sum;
	}
	else {
		double sum = 0.0;
		for (int i = startOffset; i < startOffset + len; i++){
			sum += A[i] * B[i];
		}
		return sum;
	}
	return 0.0;
}

// ��������Э����
double covV(vector<double> input){
	CV_Assert(input.size() > 0);
	double result = 0;
	double mean = meanV(input);
	for (int i = 0; i < (int)input.size(); i++){
		result += pow((input[i] - mean), 2);
	}
	return (result /(input.size()-1));
}

// ��������Э������� ����Э�������
cv::Mat covMat(vector<double> X, vector<double> Y){
	CV_Assert(X.size() > 0 && Y.size()>0);
	CV_Assert(X.size() == Y.size());

	Mat result(2, 2, CV_64FC1);
	result.at<double>(0, 0) = covV(X);
	result.at<double>(0, 1) = covV(X, Y);
	result.at<double>(1, 0) = covV(Y, X);
	result.at<double>(1, 1) = covV(Y);
	return result;
}

// ��������Э���� X Y������������Ҳ������������ ע��
double covV(Mat X, Mat Y){
	CV_Assert((X.rows == 1 && X.cols > 0) || (X.cols == 1 && X.rows > 0));
	CV_Assert((Y.rows == 1 && Y.cols > 0) || (Y.cols == 1 && Y.rows > 0));
	CV_Assert(X.rows * X.cols == Y.rows * Y.cols);

	double result = 0;
	double meanX = mean(X)[0];
	double meanY = mean(Y)[0];
	double * ptrX = X.ptr<double>(0);
	double * ptrY = Y.ptr<double>(0);
	int N = X.rows * X.cols;
	for (int i = 0; i < N; i++){
		result += (*ptrX - meanX)*(*ptrY - meanY);
		++ptrX;
		++ptrY;
	}
	return result / (N-1);
}


// ��������Э����
double covV(vector<double> X, vector<double> Y){
	CV_Assert(X.size() > 0 && Y.size()>0);
	CV_Assert(X.size() == Y.size());

	double result = 0;
	double meanX = meanV(X);
	double meanY = meanV(Y);
	for (int i = 0; i < (int)X.size(); i++){
		result += (X[i] - meanX) * (Y[i] - meanY);
	}
	return (result / (X.size() - 1));
}

// ����������ֵ
double meanV(double *v, int len){
	return sumV(v,len)/len;
}

// ������������
double varV(Mat A){
	CV_Assert((A.rows == 1 && A.cols > 0) || (A.cols == 1 && A.rows > 0));
	double *ptrA = A.ptr<double>(0);
	double result = 0;
	double meanA = mean(A)[0];
	int N = A.cols * A.rows;
	for (int i = 0; i < N; i++){
		result += (*ptrA - meanA)*(*ptrA - meanA);
		++ptrA;
	}
	return result / (N - 1);
}

// ����������׼��
double stdVector(double *v, int len, int type){
	double mean = meanV(v, len);
	double tmp = 0;
	for (int i = 0; i < len; i++){
		tmp += (v[i] - mean)*(v[i] - mean);
	}
	return sqrt(tmp / (len - type));
}

// ����׼�
double stdVector(vector<double> data){
	double mean = meanV(data);
	double tmp = 0;
	for (int i = 0; i < (int)data.size(); i++){
		tmp += (data[i] - mean)*(data[i] - mean);
	}
	return sqrt(tmp / (data.size() - 1));
}

// ����׼�
double stdVector(vector<double> x0, vector<double> x1){
	CV_Assert(x0.size() == x1.size());
	return sqrt(quadSumV(subtractV(x0, x1)));
}

// �����ϵ��
double coefV(Mat X, Mat Y){
	CV_Assert((X.rows == 1 && X.cols > 0) || (X.cols == 1 && X.rows > 0));
	CV_Assert((Y.rows == 1 && Y.cols > 0) || (Y.cols == 1 && Y.rows > 0));
	CV_Assert(X.rows * X.cols == Y.rows * Y.cols);

	double sumX = sum(X)[0];
	double sumY = sum(Y)[0];
	double sumXY = X.dot(Y);
	double sumXX = X.dot(X);
	double sumYY = Y.dot(Y);

	int n = X.rows * X.cols;

	return ((n*sumXY - sumX*sumY) / sqrt(n*sumXX - sumX*sumX) / sqrt(n*sumYY - sumY*sumY));
}

// ��������scale�߶ȱ任 �õ�������
double* scaleV(double* vec, int len, double scale){
	double* result = initVector(len);
	for (int i = 0; i < len; i++){
		result[i] = vec[i] * scale;
	}
	return result;
}

// ���������ݱ任
double* powV(double* vec, int len, double index){
	double* result = initVector(len);
	for (int i = 0; i < len; i++){
		result[i] = pow(vec[i],index);
	}
	return result;
}

// ���������ݱ任
vector<double> powV(vector<double> input, double index){
	vector<double>::iterator  it = input.begin();
	vector<double>::iterator iend = input.end();
	vector<double> result;
	while (it != iend){
		result.push_back(pow(*it, index));
		it++;
	}
	return result;
}

// ����������Ԫ��ƽ����ƽ��ֵ
double meanQuadEle(double *vec, int len){
	double *tmp = powV(vec, len, 2);
	return meanV(tmp, len);
}

// ��������Ԫ��ƽ����
double quadSumV(double* vec, int len){
	return sumV(powV(vec, len, 2), len);
}

// ��������Ԫ��ƽ����
double quadSumV(vector<double> v){
	return quadSumV(v.data(), (int)v.size());
}

//  ����������������ӦԪ����� result = vec1 + vec2
double* sumV(double *vec1, double* vec2, int len){
	double *result = initVector(len);
	for (int i = 0; i < len; i++){
		result[i] = vec1[i] + vec2[i];
	}
	return result;
}

//  ����������������ӦԪ����� result = vec1 + vec2
vector<double> addV(vector<double> vec1, vector<double> vec2){
	Mat tmp = Mat(vec1) + Mat(vec2);
	return Mat2Vec(tmp);
}

//  ����������������ӦԪ����� result = vec1 - vec2
double* subV(double *vec1, double* vec2, int len){
	double *result = initVector(len);
	for (int i = 0; i < len; i++){
		result[i] = vec1[i] - vec2[i];
	}
	return result;
}

//  ��������ÿ��Ԫ�ؼ�ȥһ����ֵ
double* subV(double *vec, double delta, int len){
	double *result = initVector(len);
	for (int i = 0; i < len; i++){
		result[i] = vec[i] - delta;
	}
	return result;
}

//  ��������ÿ��Ԫ�ؼ�ȥvec��ƽ��ֵ
double* divV(double *vec, int len){
	double mean = meanV(vec, len);
	return subV(vec, mean, len);
}

//�������������˱���ϵ��
vector<double> scaleV(vector<double> input, double scale, double offset){
	CV_Assert(input.size() > 0);
	vector<double> result;
	vector<double>::iterator indexA = input.begin();
	vector<double>::iterator iend = input.end();
	while (indexA != iend){
		result.push_back((*indexA)*scale + offset);
		indexA++;
	}
	return result;
}

// ��������������Ԫ�ض�Ӧ������
vector<double> subtractV(vector<double> A, vector<double> B){
	CV_Assert(A.size() == B.size());
	vector<double> result;
	vector<double>::iterator indexA = A.begin();
	vector<double>::iterator indexB = B.begin();
	vector<double>::iterator iend = A.end();
	while (indexA != iend){
		result.push_back((*indexA) - (*indexB));
		indexA++;
		indexB++;
	}
	return result;
}

// ��������������ȥһ����ֵ
vector<double> subtractV(vector<double> x, double val){
	vector<double> result;
	vector<double>::iterator indexX = x.begin();
	vector<double>::iterator iend = x.end();
	while (indexX != iend){
		result.push_back((*indexX) - val);
		indexX++;
	}
	return result;
}

// ��������������Ԫ�ض�Ӧ������
vector<double> divisionV(vector<double> A, vector<double> B){
	CV_Assert(A.size() == B.size());
	vector<double> result;
	vector<double>::iterator indexA = A.begin();
	vector<double>::iterator indexB = B.begin();
	vector<double>::iterator iend = A.end();
	while (indexA != iend){
		result.push_back((*indexA)/(*indexB));
		indexA++;
		indexB++;
	}
	return result;
}

// ������������Ԫ�ط���
vector<double> signV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	while (index != iend){
		result.push_back(signValue(*index));
		index++;
	}
	return result;
}

//  ����������������ӦԪ����� result = vec1 * vec2
double* mulV(double *vec1, double* vec2, int len){
	double *result = initVector(len);
	for (int i = 0; i < len; i++){
		result[i] = vec1[i] * vec2[i];
	}
	return result;
}

// ���˷�����������ӦԪ����� �˷�
vector<double> mulV(vector<double> vec1, vector<double> vec2){
	CV_Assert(vec1.size() == vec2.size());
	vector<double> result;
	vector<double>::iterator indexA = vec1.begin();
	vector<double>::iterator indexB = vec2.begin();
	vector<double>::iterator iend = vec1.end();
	while (indexA != iend){
		result.push_back((*indexA) * (*indexB));
		indexA++;
		indexB++;
	}
	return result;
}

// �����������
double dotV(double* vec1, double *vec2, int len){
	double result = 0.0f;
	for (int i = 0; i < len; i++)
		result += vec1[i] * vec2[i];
	return result;
}

// ����������Ԫ��ȡģ
void absV(double* vec, int len){
	for (int i = 0; i < len; i++){
		vec[i] = fabs(vec[i]);
	}
}

// ����������Ԫ��ȡsinֵ
vector<double> sinV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	while (index != iend){
		result.push_back(sin(*index));
		index++;
	}
	return result;
}

// ��log����Ԫ��ȡlogֵ
vector<double> logV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	while (index != iend){
		result.push_back(log(*index));
		index++;
	}
	return result;
}

// ��sqrt����Ԫ��ȡƽ����
vector<double> sqrtV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();	
	while (index != iend){
		result.push_back(sqrt(*index));
		index++;
	}
	return result;
}

// ��ln����Ԫ��ȡlnֵ
vector<double> lnV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	double e = exp(1.0);
	while (index != iend){
		result.push_back(log(*index)/log(e));
		index++;
	}
	return result;
}

// ��exp�� ��Ԫ��ȡexpֵ
vector<double> expV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	while (index != iend){
		result.push_back(exp(*index));
		index++;
	}
	return result;
}

// ����������Ԫ��ȡsincֵ
vector<double> sincV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	while (index != iend){
		result.push_back((sin(*index) / (*index)));
		index++;
	}
	return result;
}

// ����������Ԫ��ȡcosֵ
vector<double> cosV(vector<double> input){
	vector<double> result;
	vector<double>::iterator index = input.begin();
	vector<double>::iterator iend = input.end();
	while (index != iend){
		result.push_back(cos(*index));
		index++;
	}
	return result;
}

// �����������������ŷ�Ͼ���
double distV(double* vec1, double *vec2, int len){
	double* tmp = subV(vec1, vec2, len);
	return sqrt(quadSumV(tmp, len));
}

// ���������෴��
void oppositeNumVec(float *vec, int len){
	for (int i = 0; i < len; i++)
		vec[i] *= -1;
}

// ���������������Ԫ��ֵ
double maxV(vector<double> vec){
	vector<double>::iterator where = max_element(vec.begin(), vec.end());
	return (*where);
}

// ��������������СԪ��ֵ
double minV(vector<double> vec){
	vector<double>::iterator where = min_element(vec.begin(), vec.end());
	return (*where);
}

// ���������������Ԫ��ֵ
double maxV(double* vec, int len){
	double result = vec[0];
	for (int i = 0; i < len; i++){
		if (result < vec[i]){
			result = vec[i];
		}
	}return result;
}

// ���������������Ԫ��ֵ
float maxV(float* vec, int len){
	float result = vec[0];
	for (int i = 0; i < len; i++){
		if (result < vec[i]){
			result = vec[i];
		}
	}return result;
}

// ��������������СԪ��ֵ
float minV(float* vec, int len){
	float result = vec[0];
	for (int i = 0; i < len; i++){
		if (result > vec[i]){
			result = vec[i];
		}
	}return result;
}

// ����������ӡ����double
void printVector(vector<double> vec){
	printf("   [\n");
	for (int i = 0; i < (int)vec.size(); i++){
		if (vec[i] >= 0) 	printf("      %.13lf ,\n", vec[i]);
		else	printf("     %.13lf ,\n", vec[i]);
	}printf("    ];\n");
}

// ����������ӡ���� double
void printVector(vector<double> vec, int len){
	printf("   [\n");
	for (int i = 0; i < len; i++){
		if (vec[i] > 0) 	printf("      %.13f ,\n", vec[i]);
		else	printf("     %.13f ,\n", vec[i]);
	}printf("    ];\n");
}

// ����������ӡ���� double
void printVector(double* vec, int len){
	printf("   [\n");
	for (int i = 0; i < len; i++){
		if (vec[i] > 0) 	printf("      %.13f ,\n", vec[i]);
		else	printf("     %.13f ,\n", vec[i]);
	}printf("    ];\n");
}

// ����������ӡ���� float
void printVector(float* vec, int len){
	printf("   [\n");
	for (int i = 0; i < len; i++){
		cout << setprecision(12) << vec[i] << endl;
		//if (vec[i] > 0) 	printf("      %.12f ,\n", vec[i]);
		//else	printf("     %.12f ,\n", vec[i]);
	}printf("    ];\n");
}

// ����������ӡ���� int
void printVector(int* vec, int len){
	printf("   [\n");
	for (int i = 0; i < len; i++){
		if (vec[i] > 0) 	printf("      %03d ,\n", vec[i]);
		else	printf("     %03d ,\n", vec[i]);
	}printf("    ];\n");
}

// �������Сֵ����vector �������Сֵ
void minMaxVector(vector<double> input, double& minV, double& maxV, int& minIndex, int& maxIndex, int N){
	minV = DBL_MAX;
	maxV = -DBL_MAX;
	minIndex = 0;
	maxIndex = 0;
	int size = min(N, (int)input.size());
	for (int i = 0; i < size; i++){
		if (minV > input[i]){
			minV = input[i];
			minIndex = i;
		}
		if (maxV < input[i]){
			maxV = input[i];
			maxIndex = i;
		}
	};
}

// �������Сֵ������ֵ[0]Ϊ��Сֵ [1]Ϊ���ֵ
double* minMaxVector(double* input, int len){
	double *value;
	value = (double*)malloc(sizeof(double)* 2);
	value[0] = DBL_MAX;
	value[1] = -DBL_MAX;
	int maxIndex = 0, minIndex = 0;
	for (int i = 0; i < len; i++){
		if (value[0] > input[i]){
			value[0] = input[i];
			minIndex = i;
		}
		if (value[1] < input[i]){
			value[1] = input[i];
			maxIndex = i;
		}
	}
	printf("min = %lf max = %lf\n", value[0], value[1]);
	printf("minIndex = %d maxIndex = %d\n", minIndex, maxIndex);
	return value;
}

// �����ֵ��
double maxValV(double* input, int len){
	double value = -DBL_MAX;
	for (int i = 0; i < len; i++){
		if (value < input[i]){
			value = input[i];
		}
	}
	return value;
}

// �����ֵ�����
int maxIndexV(double* input, int len){
	double value = -DBL_MAX;
	int result = -1;
	for (int i = 0; i < len; i++){
		if (value < input[i]){
			value = input[i];
			result = i;
		}
	}
	return result;
}

// ����Сֵ��
double minValV(double* input, int len){
	double value = DBL_MAX;
	for (int i = 0; i < len; i++){
		if (value > input[i]){
			value = input[i];
		}
	}
	return value;
}

// ����Сֵ�����
int minIndexV(double* input, int len){
	double value = DBL_MAX;
	int result = -1;
	for (int i = 0; i < len; i++){
		if (value > input[i]){
			value = input[i];
			result = i;
		}
	}
	return result;
}

// ������������ֵ
void swap2int(int& a, int& b){
	int tmp = a;
	a = b;
	b = tmp;
}

// ������������ֵ
void swap2float(float& a, float& b){
	float tmp = a;
	a = b;
	b = tmp;
}

// �����ص�һ��ĩβ��š�����һ������ һ��float ��m������С�������У� ǰn�������󣬴ӵ�n+1����ʼ�������ǰ��ı仯�ܶ࣬��n
int numGrp1st(float* vec, int len){
	float* div = new float[len-1]; // һ�ײ��
	for (int i = 0; i < len - 1; i++){
		div[i] = vec[i + 1] - vec[i];
		//printf("[%2d] %f\n", i, div[i]);
	}

	for (int i = 0; i < len - 2; i++){
		if (div[i + 1] - div[i] > 0.07)
			return i + 2;
	}
	return len;
}

// �������
double sumVector(vector<double> v){
	double sum = 0;
	vector<double>::iterator it = v.begin();
	for (; it < v.end(); it++){
		sum += *it;
	}
	return sum;
}

// ����ȡƽ��
double meanVector(vector<double> v){
	return (double)(sumVector(v) / v.size());
}

// �����ֵ��pair<int, int> useSecond = true ���жϵڶ���Ԫ�ص����ֵ,��ô����ֵ�ͷ��صڶ���Ԫ�ص����ֵ
int maxPair(vector<pair<int, int>>input, int& otherResult, bool useSecond){
	CV_Assert(input.size() > 0);
	int result = 0;
	if (useSecond){
		result = input[0].second;
		otherResult = input[0].first;
		for (int i = 1; i < (int)input.size(); i++){
			if (result < input[i].second){
				result = input[i].second;
				otherResult = input[i].first;
			}
		}
	}
	else {
		result = input[0].first;
		otherResult = input[0].second;
		for (int i = 1; i < (int)input.size(); i++){
			if (result < input[i].first){
				result = input[i].first;
				otherResult = input[i].second;
			}
		}
	}
	return result;
}

// ����Сֵ��pair<int, int> useSecond = true ���жϵڶ���Ԫ�ص���Сֵ,��ô����ֵ�ͷ��صڶ���Ԫ�ص���Сֵ
int minPair(vector<pair<int, int>>input, int& otherResult, bool useSecond){
	CV_Assert(input.size() > 0);
	int result = 0;
	if (useSecond){
		result = input[0].second;
		otherResult = input[0].first;
		//cout << "result = " << result << endl;
		//cout << "otherResult = "<<otherResult << endl;
		for (int i = 1; i < (int)input.size(); i++){
			//cout << "i =" << i << endl;
			if (result > input[i].second){
				result = input[i].second;
				otherResult = input[i].first;
			//	cout << "result = " << result << endl;
			//	cout << "otherResult = " << otherResult << endl;
			}
		}
	}
	else {
		result = input[0].first;
		otherResult = input[0].second;
		for (int i = 1; i < (int)input.size(); i++){
			if (result > input[i].first){
				result = input[i].first;
				otherResult = input[i].second;
			}
		}
	}
	return result;
}

// �����Сֵ
void maxmin(vector<double>a, double &maxv, double &minv){
	sort(a.begin(), a.end());
	maxv = a[a.size() - 1];
	minv = a[0];
}

// �����ơ�����
void copyV(double* src, double *dst, int len){
	for (int i = 0; i < len; i++)
		dst[i] = src[i];
}

// �����ơ�����
vector<double> copyV(vector<double> src){
	vector<double> result;
	result.assign(src.begin(), src.end());
	return result;
}

// �����桿 ���ɴ������������ź�
double* createNoiCos(double A, double freq, double phi, double C, double noiLev, int N){
	double *result = initVector(N);
	for (int i = 0; i < N; i++){
		result[i] = A*cos(2 * CV_PI*freq*i + phi) + C + noiLev*(rndDouble() - 0.5);
	}
	return result;
}

// ���������� �������
bool reverseArray(vector<double> &input){
	reverse(input.begin(), input.end());
	return true;
}

// ��ż���С����ż����
double* getEvenArray(double* input, int n){
	CV_Assert(n>0 && n % 2 == 0);
	double* result = initVector(n / 2);
	for (int i = 0; i < n / 2; i++){
		result[i] = input[i * 2 + 1];
	}
	return result;
}

// ��ż���С����ż����
vector<double> getEvenArray(vector<double> input){
	CV_Assert(input.size() > 0 && input.size() % 2 == 0);
	vector<double> result(input.size() / 2);
	for (int i = 0; i < (int)input.size() / 2; i++){
		result[i] = input[i * 2 + 1];
	}
	return result;
}

// �������С����������
double* getOddArray(double* input, int n){
	CV_Assert(n>0 && n % 2 == 0);
	double* result = initVector(n / 2);
	for (int i = 0; i < n / 2; i++){
		result[i] = input[i * 2];
	}
	return result;
}

// �������С������������
vector<double> getOddArray(vector<double> input){
	CV_Assert(input.size() > 0 && input.size() % 2 == 0);
	vector<double> result(input.size() / 2);
	for (int i = 0; i < (int)input.size() / 2; i++){
		result[i] = input[i * 2];
	}
	return result;
}

// �����㡿��Ϊ2���������ݸ�Ԫ��
bool adjustToIntegralPowerOfTwo(vector<double>& input){
	int n = (int)input.size();
	int r = roof2power(n);
	while (n < r){
		input.push_back(0.0);
		n++;
	}
	return true;
}

// ��FFT��
COMPLEX fftV(double* input, int n){
	if (n == 2)
	{
		COMPLEX result(2);/*
		result.real = new double[2];
		result.imag = new double[2];*/
		result.real[0] = input[0] + input[1];
		result.imag[0] = 0;
		result.real[1] = input[0] - input[1];
		result.imag[1] = 0;
		return result;
	} 
	else if(n > 2){
		double* oddArray = getOddArray(input, n);
		double* evenArray = getEvenArray(input, n);
		printf("oddArray = \n");
		printVector(oddArray, n / 2);
		printf("evenArray = \n");
		printVector(evenArray, n / 2);

		COMPLEX oddF = fftV(oddArray, n / 2);
		COMPLEX evenF = fftV(evenArray, n / 2);
		COMPLEX result(n);
		/*result.real = new double[n];
		result.imag = new double[n];*/
		for (int k = 0; k < n / 2; k++){
			double tmpReal = cos(PI_2*k / n)*evenF.real[k] + sin(PI_2*k / n)*evenF.imag[k];
			double tmpImag = -sin(PI_2*k / n)*evenF.real[k] + cos(PI_2*k / n)*evenF.imag[k];
			result.real[k] = oddF.real[k] + tmpReal;
			result.imag[k] = oddF.imag[k] + tmpImag;
			result.real[k + n / 2] = oddF.real[k] - tmpReal;
			result.imag[k + n / 2] = oddF.imag[k] - tmpImag;
		}
		return result;
	}
	return COMPLEX(0);
}

// ��FFT��
COMPLEX fftV(vector<double> input, int N){	
	//static int flag = 0;
	//printf("flag = %d\n", flag);

	double base = log((double)N) / log(2.0);
	CV_Assert(base - floor(base) < 1e-10);

	while ((int)input.size() < N)
		input.push_back(0.0);

	int n = N;


	if (input.size() == 2)
	{
		COMPLEX result(2);
		result.real[0] = input[0] + input[1];
		result.imag[0] = 0;
		
		result.real[1] = input[0] - input[1];
		result.imag[1] = 0;
		return result;
	}
	if (input.size() > 2){
		vector<double> oddArray = getOddArray(input);
		vector<double> evenArray = getEvenArray(input);

		COMPLEX oddF = fftV(oddArray, n/2);
		COMPLEX evenF = fftV(evenArray, n/2);
		COMPLEX result(n);

		for (int k = 0; k < n / 2; k++){
			double tmpReal = cos(PI_2*k / n)*evenF.real[k] + sin(PI_2*k / n)*evenF.imag[k];
			double tmpImag = -sin(PI_2*k / n)*evenF.real[k] + cos(PI_2*k / n)*evenF.imag[k];

			result.real[k] = oddF.real[k] + tmpReal;
			result.imag[k] =  oddF.imag[k] + tmpImag;

			result.real[k + n / 2] = oddF.real[k] - tmpReal;
			result.imag[k + n / 2] = oddF.imag[k] - tmpImag;
		}
		return result;
	}
	return COMPLEX(0);
}

// �����任Ϊ������
POLAR_COMPLEX convert2Polar(COMPLEX input){
	POLAR_COMPLEX result(input.n);
	for (int i = 0; i < input.n; i++){
		result.ampl[i] = getAmpl(input.real[i], input.imag[i]);
		result.angl[i] = getAngl(input.real[i], input.imag[i]);
	}
	return result;
}

// ʵ����ɢ��� x(n) �� h(n) �ľ�� ������Ԫ����M+N-1��
vector<double> discreteConv(vector<double> x, vector<double> h){
	int M = (int)x.size();
	int N = (int)h.size();
	vector<double> result(M + N - 1);

	for (int i = 0; i < M + N - 1; i++){
		cout << i << endl;
		result[i] = 0.0;
		for (int j = 0; j < M+N-1; j++){
			if (i - j >= 0 && i-j<N && j < M)
				result[i] += x[j] * h[i - j];
		}
	}
	return result;
}

// �����ҡ�������������ӽ���Ԫ�����
int getNearestEleIndex(float* data, float target, int len){
	CV_Assert(len > 0 || data[len - 1] > data[0]);
	if (target < data[0]){
		return 0;
	}
	else if (target > data[len - 1]){
		return len - 1;
	}
	float minDist = data[len - 1] - data[0];
	int index = -1;
	for (int i = 0; i < len; i++){
		float cur = fabs(data[i] - target);
		if (cur < minDist){
			minDist = cur;
			index = i;
		}
	}
	return index;
}

// �����滻
// vec�ǳ�ʼ����
// index�Ǵ�����ֵ����������
// data�Ǵ������ֵ
void substitute(Mat vec, vector<int> index, Mat data){
	CV_Assert(index.size() >= 1);
	CV_Assert(vec.type() == data.type());
	CV_Assert((data.cols == index.size() && data.rows == 1) || (data.rows == index.size() && data.cols == 1));
	CV_Assert((vec.rows >= (int)index.size() && vec.cols == 1) || (vec.cols >= (int)index.size() && vec.rows == 1));

	for (int i = 0; i < (int)index.size(); i++){
		//cout << "i" << i << endl;
		if (vec.type() == CV_8UC1){
			vec.ptr<uchar>(0)[index[i]] = data.ptr<uchar>(0)[i];
		}
		else if (vec.type() == CV_64FC1){
			vec.ptr<double>(0)[index[i]] = data.ptr<double>(0)[i];
		}
		else if (vec.type() == CV_32FC1){
			vec.ptr<float>(0)[index[i]] = data.ptr<float>(0)[i];
		}
		else if (vec.type() == CV_64FC2){
			vec.ptr<Point2d>(0)[index[i]] = data.ptr<Point2d>(0)[i];
		}
	}
}

// һά��ֵ 3��������ֵ
// ddy1 �ǵ�һ��Ķ��׵���
// ddyn �����һ��Ķ��׵���
vector<double> interpSpline(vector<double> x, vector<double> y, vector<double> x_new, double ddyStart, double ddyEnd){
	int n = (int)x.size();
	int m = (int)x_new.size();
	
	vector<double> s(n);
	vector<double> dy(n);
	vector<double> z(m);
	
	dy[0] = -0.5;
	double h0 = x[1] - x[0];
	s[0] = 3.0*(y[1] - y[0]) / (2.0*h0) - ddyStart*h0 / 4.0;
	
	double h1 = 0;

	for (int j = 1; j <= n - 2; j++)
	{
		h1 = x[j + 1] - x[j];
		double alpha = h0 / (h0 + h1);
		double beta = (1.0 - alpha)*(y[j] - y[j - 1]) / h0;
		beta = 3.0*(beta + alpha*(y[j + 1] - y[j]) / h1);
		dy[j] = -alpha / (2.0 + (1.0 - alpha)*dy[j - 1]);
		s[j] = (beta - (1.0 - alpha)*s[j - 1]);
		s[j] = s[j] / (2.0 + (1.0 - alpha)*dy[j - 1]);
		h0 = h1;
	}
	dy[n - 1] = (3.0*(y[n - 1] - y[n - 2]) / h1 + ddyEnd*h1 / 2.0 - s[n - 2]) / (2.0 + dy[n - 2]);
	for (int j = n - 2; j >= 0; j--)        
		dy[j] = dy[j] * dy[j + 1] + s[j];
	for (int j = 0; j <= n - 2; j++)        
		s[j] = x[j + 1] - x[j];

	int i = 0;
	for (int j = 0; j <= m - 1; j++)
	{
		if (x_new[j] >= x[n - 1]) i = n - 2;
		else
		{
			i = 0;
			while (x_new[j]>x[i + 1]) i = i + 1;
		}
		h1 = (x[i + 1] - x_new[j]) / s[i];
		h0 = h1*h1;
		h1 = (x_new[j] - x[i]) / s[i];
		h0 = h1*h1;
		z[j] = z[j] + (3.0*h0 - 2.0*h0*h1)*y[i + 1];
		z[j] = z[j] - s[i] * (h0 - h0*h1)*dy[i + 1];
	}
	return z;
}

// Lagrange��ֵ �������ղ�ֵ
vector<double> interpLagrange(vector<double> x, vector<double> y, vector<double> x0){
	CV_Assert(x.size() == y.size());
	int m = (int)x0.size();
	vector<double> s(m);
	for (int i = 0; i < m; i++){
		double t = 0;
		for (int j = 0; j < (int)x.size(); j++){
			double u = 1;
			for (int k = 0; k < (int)x.size(); k++){
				if (k != j){
					u = u*(x0[i] - x[k]) / (x[j] - x[k]);
				}
			}
			t = t + u*y[j];
		}
		s[i] = t;
	}
	return s;
}

// Newton��ֵ ţ�ٲ�ֵ
// x��yΪ��֪�Ĳ�ֵ�㼰�亯��ֵ
// x0ΪҪ��Ĳ�ֵ���x����ֵ��nnΪNewton��ֵ����ʽ�Ľ���
vector<double> interpNewton(vector<double> x, vector<double> y, vector<double> x0, int nn){
	CV_Assert(x.size() == y.size());
	int m = (int)x0.size();
	vector<double> s(m);
	for (int i = 1; i <= m; i++){
		double t = 0.0;
		int j = 1;
		vector<double> yy = copyV(y);
		int kk = j;
		// ���������
		while (kk <= nn){
			kk++;
			for (int k = kk; k <= (int)x.size(); k++){
				yy[k - 1] = (yy[k - 1] - yy[kk - 2]) / (x[k - 1] - x[kk - 2]);
			}
		}
		//printVector(yy);
		// ���ֵ���
		t = yy[0];
		
		for (int k = 2; k <= nn; k++){
			double u = 1.0;
			int jj = 1;
			while (jj < k){
				u = u*(x0[i - 1] - x[jj - 1]);
				//cout << "u = " << u << endl;
				jj++;
			}
			t = t + yy[k - 1] * u;
			//cout << "t = " << t << endl;
		}
		s[i - 1] = t;
	}
	return s;
}

// ����ʽ��ֵ
// coef Ϊ����ʽϵ�����Ӹߴ���ʹ�������
// x���Ա���
vector<double> polyval(vector<double> coef, vector<double> x){
	vector<double> y(x.size());
	for (int i = 0; i < (int)x.size(); i++){
		int n = (int)coef.size();
		double temp = 0.0;
		for (int j = 0; j < n; j++){
			//cout << "j = " << j << endl;
			temp += coef[j]*pow(x[i], (double)(n-j-1));
		}
		y[i] = temp;
	}
	return y;
}

double* FFT(double* x, int N){
	int K = N;
	double *X_real = initVector(K);
	double *X_imag = initVector(K);
	double *X_ampl = initVector(K);

	double t = 2 * CV_PI / N;

	for (int k = 0; k < K; k++){
		X_real[k] = 0;
		X_imag[k] = 0;
		for (int n = 0; n <= N - 1; n++){
			X_real[k] = X_real[k] + cos(t*k*n);
			X_imag[k] = X_imag[k] - sin(t*k*n);
		}
		X_ampl[k] = sqrt(X_real[k] * X_real[k] + X_imag[k] * X_imag[k]);
	}

	return X_ampl;
}

// ��ɢ�źŵ�����غ���
vector<double> selfCOV(vector<double> input){
	CV_Assert(input.size() >= 1);
	vector<double> result(input.size());
	for (int i = 0; i < (int)input.size(); i++){
		result[i] = 0;
		for (int j = i; j <= input.size() - 1; j++){
			result[i] += input[j] * input[j - i];
		}
	}
	return result;
}

// ��ɢ����źŵĻ���غ���
vector<double> xcorr(vector<double> x, vector<double> y){
	vector<double> result(x.size() + y.size() - 1);

	for (int n = 0; n < (int)result.size(); n++){
		result[n] = 0;
		for (int j = 0; j < (int)x.size(); j++){
			if (n-j >= 0 && n-j< y.size()){
				result[n] += x[j] * y[n-j];
			}
		}
	}
	return result;
}

// ��ɢ����źŵĻ�Э������
vector<double> xcov(vector<double> x, vector<double> y){
	vector<double> x_new = subtractV(x, meanVector(x));
	vector<double> y_new = subtractV(y, meanVector(y));
	
	return xcorr(x_new, y_new);
}

// X = F.*2
//[F, E] = log2(X) for each element of the real array X, returns an
//array F of real numbers, usually in the range 0.5 <= abs(F) < 1,
//and an array E of integers, so that X = F .* 2.^E.Any zeros in X
//produce F = 0 and E = 0.  This corresponds to the ANSI C function
//frexp() and the IEEE floating point standard function logb().
vector<pair<double, int>> log2(vector<double> input){
	vector<pair<double, int>> result;
	vector<double>::iterator i = input.begin();
	for (; i != input.end(); i++){
		result.push_back(log2Double(*i));
	}
	return result;
}

// �������
vector<double> convV(vector<double> a, vector<double> b){
	vector<double> result = initVectord(a.size()+b.size()-1);

	// ���a�ĳ��ȴ��ڻ����b�ĳ���
	if (a.size() >= b.size()){
		for (int i = 0; i < (int)b.size(); i++){
			for (int j = 0; j <= i; j++){
				double aa = a[i - j];
				double bb = b[j];
				result[i] = result[i] + aa*bb;
			}
		}

		for (int i = b.size(); i < (int)a.size(); i++){
			for (int j = 0; j < b.size(); j++){
				double aa = a[i - j];
				double bb = b[j];
				result[i] = result[i] + aa*bb;
			}
		}

		for (int i = a.size(); i < (int)a.size() + (int)b.size(); i++){
			for (int j = i - a.size() + 1; j < b.size(); j++){
				double aa = a[i - j];
				double bb = b[j];
				result[i] = result[i] + aa*bb;
			}
		}
	}
	else{	// ���b�ĳ��ȴ��ڻ����a�ĳ���
		for (int i = 0; i < (int)a.size(); i++){
			for (int j = 0; j < i; j++){
				double bb = b[i - j];
				double aa = a[j];
				result[i] = result[i] + aa*bb;
			}
		}

		for (int i = a.size(); i < (int)b.size(); i++){
			for (int j = 0; j < (int)a.size(); j++){
				double bb = b[i - j];
				double aa = a[j];
				result[i] = result[i] + aa*bb;
			}
		}

		for (int i = (int)b.size(); i < (int)b.size() + (int)a.size() - 1; i++){
			for (int j = i - (int)b.size() + 1; j < (int)a.size(); j++){
				double bb = b[i - j];
				double aa = a[j];
				result[i] = result[i] + aa*bb;
			}
		}
	}

	return result;
}

// ������һά����֮��Ĳ��� һά������ܳ��Ȳ�һ��
double dist2Vec(vector<double> a, vector<double> b){
	Mat A(a.size(), 1, CV_64FC1, a.data());
	Mat B(b.size(), 1, CV_64FC1, b.data());
	
	if (a.size() > b.size()){
		resize(B, B, A.size());
	}
	else if(a.size() < b.size()){
		resize(A, A, B.size());
	}

	return norm(A-B);
}

// ����������һ�� ���������ݽ��в���
void NormalizeV(vector<double>& vec, double minValue /* = 0*/, double maxValue /*= 1.0*/){
	double minVal = minV(vec);
	double maxVal = maxV(vec);
	double scale = (maxValue - minValue) / (maxVal - minVal);
	double alpha = minValue - minVal;

	vector<double> result;
	vector<double>::iterator it = vec.begin();
	for (; it != vec.end(); it++){
		*it = ((*it) - minVal) * scale + minValue;
	}
}

// ����������һ�� 
vector<double> normalizeV(vector<double> vec, double minValue /* = 0*/, double maxValue /*= 1.0*/){	
	double minVal = minV(vec);
	double maxVal = maxV(vec);
	double scale = (maxValue - minValue) / (maxVal - minVal);
	double alpha = minValue - minVal;

	vector<double> result;
	vector<double>::iterator it = vec.begin();
	for (; it != vec.end(); it++){
		result.push_back(((*it) - minVal) * scale + minValue);
	}
	return result; 
}

// ��ֵ������С��ĳ��ֵ�ľ�����
vector<double> thresholdV(vector<double> signal, double thresh){
	vector<double>::iterator it = signal.begin();
	vector<double> result;
	while (it != signal.end()){
		if (*it >= thresh){
			result.push_back(*it);
		}
		else{
			result.push_back(0);
		}
		it++;
	}
	return result;
}

// ָ��Ԫ�ظ�ֵ ����Ϊ���� Mat��ָ��Ԫ�ظ�ֵ
void assignByIndexs(Mat& input, Mat values, vector<int> idxs){
	CV_Assert(input.cols == 1 && values.total() == idxs.size());

	double* data = input.ptr<double>(0);
	for (int i = 0; i < (int)idxs.size(); i++){
		data[idxs[i]] = values.ptr<double>(0)[i];
	}
}


// ָ��Ԫ�ظ�ֵ ����Ϊ���� Mat��ָ��Ԫ�ظ�ֵ ĳЩԪ�ظ�ֵ
void assignByIndexs(cv::Mat& input, double val, vector<int> idxs){
	double* data = input.ptr<double>(0);
	for (int i = 0; i < (int)idxs.size(); i++){
		data[idxs[i]] = val;
	}
}

// ���Ҵ���a������Ԫ����� ���� ɸѡ ѡȡ
vector<int> findGTR(Mat input, double a){
	vector<int> idxs;
	double *data = input.ptr<double>(0);
	for (int i = 0; i < input.total(); i++){
		if (*data > a){
			idxs.push_back(i);
		}
	}
	return idxs;
}


// ����С��a������Ԫ����� ���� ɸѡ ѡȡ ĳЩ
vector<int> findLESS(cv::Mat input, double a){
	vector<int> idxs;
	double *data = input.ptr<double>(0);
	for (int i = 0; i < input.total(); i++){
		if (*data < a){
			idxs.push_back(i);
		}
	}
	return idxs;
}

// �ҷ�ֵ�� interval��������� ��ֵ���
vector<int> findPeaks(vector<double> data, int interval){
	vector<int> result;

	// ������ֵΪƽ��ֵ+���ֵ*5%
	double thresh = meanV(data) + maxV(data) * 0.05;

	int curId = 0;
	int lastId = curId;

	for (int i = 0; i<(int)data.size();i++){
		bool flag = true;
		for (int j = 1; j<interval / 2; j++){
			if (i - j >= 0){
				if (data[i] < data[i - j]) {
					flag = false;
					break;
				}
			}
			if (i + j < (int)data.size()){
				if (data[i] < data[i + j]) {
					flag = false;
					break;
				}
			}
		}

		if (flag && data[i]>data[i + 1] && data[i] > thresh){
			if (result.empty() || (!result.empty() && curId - lastId > interval)){
				result.push_back(curId);
				lastId = curId;
			}
			else if(curId - lastId <= interval && data[i] > data[result[result.size()-1]]){
				result[result.size() - 1] = curId;
				lastId = curId;
			}
		}
		curId++;
		
		cout << Mat(result) << endl;
	}
	return result;
}


// ���������ź�
// phi0Ϊ����ֵ b��ֱ������
vector<double> sinSignal(double T, double A, double phi0, double b, int len, bool useCos/* = false*/){
	double w = 2 * CV_PI / T;
	vector<double> result = initVectord(len);
	for (int i = 0; i < len; i++){
		if (useCos)
			result[i] = A*cos(i*w + phi0) + b;
		else
			result[i] = A*sin(i*w + phi0) + b;
	}
	return result;
}


// ������ת��Ϊvector<double>
vector<double> Mat2Vec(Mat input){
	vector<double> result;
	double * pinput = input.ptr<double>(0);
	for (int i = 0; i < input.rows*input.cols; i++){
		result.push_back(*pinput);
		pinput++;
	}
	return result;
}

// 1D����Ҷ�任 ����ֵΪMatͨ��1Ϊʵ�� ͨ��2Ϊ�鲿 һά����Ҷ�任
Mat fft(vector<double> signal){
	Mat input = Mat(signal);
	int m = getOptimalDFTSize((int)input.total());
	int n = 1;

	// Padding 0, result is @ padded ����Ĳ��ֲ���
	Mat padded;
	copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));

	// Create planes to storage REA	L part and IMAG part, IMAGE part init are 0
	Mat planes[] = { Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F) };

	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);
	return complexI;
}

// 1D����Ҷ�任 ȡ��ֵ FFT һά����Ҷ�任
vector<double> fft_Amplitude(vector<double> signal){	
	Mat complexI = fft(signal);
	
	vector<Mat> planes;
	// compute the magnitude and switch to logarithmic scale 
	split(complexI, planes);
	Mat mag;

	// ���ֵ����λ��Ϣ
	magnitude(planes[0], planes[1], mag);
	return Mat2Vec(mag);
}

// 1D����Ҷ�任 ȡ��λ FFT һά����Ҷ�任
vector<double> fft_Phase(vector<double> signal){
	Mat complexI = fft(signal);

	vector<Mat> planes;
	// compute the magnitude and switch to logarithmic scale 
	split(complexI, planes);

	Mat angle;

	// ���ֵ����λ��Ϣ
	phase(planes[0], planes[1], angle);
	return Mat2Vec(angle);
}