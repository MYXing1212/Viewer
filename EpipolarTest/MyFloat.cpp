//#include"stdafx.h"
#include"stdafx.h"
#include"MyFloat.h"

using namespace cv;

// ���ɸ�˹�ֲ�/��̬�ֲ��������
double GaussRand(double mu, double sigma){
	static double v1, v2, s;
	static int phase = 0;
	double x;

	if (0 == phase)
	{
		do
		{
			double u1 = (double)rand() / RAND_MAX;
			double u2 = (double)rand() / RAND_MAX;

			v1 = 2 * u1 - 1;
			v2 = 2 * u2 - 1;
			s = v1 * v1 + v2 * v2;
		} while (1 <= s || 0 == s);
		x = v1 * sqrt(-2 * log(s) / s);
	}
	else
	{
		x = v2 * sqrt(-2 * log(s) / s);
	}
	phase = 1 - phase;

	return ((x*sigma) + mu);
}

int RoundOff(float input)
{
	if (input - (int)input < 0.5)
		return (int)input;
	else
		return (int)input + 1;
}

// ���ź���
double signValue(double input){
	if (input > 0)
		return 1.0;
	else if (input < 0)
		return -1.0;
	else
		return 0.0;
}

// ���λ���� 1�����Ǹ�λ -1����С�����1λ
int topDigit(double input){

	input = abs(input);

	if (input < 1e-13)
		return 0;
	
	int i = 1;
	if (input<1) i = 0;
	while (1){
		if (input >= 1 && input<10)
		{
		
			return i;
		}
		if (floor(input) >= 10){
			i++;
			input /= 10;
		}
		else if (floor(input) < 1){
			i--;
			input *= 10;
		}
	}
}

// ��ȡ������ϱ߽�
double limitUp(double input, double ref){
	double tmp = input + pow(10.0, topDigit(ref) - 2);
	return floor(tmp / pow(10.0, topDigit(ref) - 2))*pow(10.0, topDigit(ref) - 2);
}

// ��ȡ������±߽�
double limitDown(double input, double ref){
	return input - pow(10.0, topDigit(ref) - 2);
}

// ��ȡĳ���Ͻ�����������Ϊref ��ȷ��ref�����λ
double calTopNum(double src, double ref){
	if (fabs(ref) < 1e-7)
		return ceil(src);
	int td = topDigit(ref);
	if (src > 0){
		if (td > 0){
			double tmp = src / pow((double)10, td - 1) + 0.000001;
			src = ceil(tmp);
			//CString note;
			//note.Format(_T("%lf"), tmp);
			//AfxMessageBox(note);
			src *= pow((double)10, td - 1);
		}
		else {
			double tmp = src * pow((double)10, -td) + 0.000001;
			src = ceil(tmp);
			src /= pow((double)10, -td);
		}
	}
	else {
		double tmpv = calFloorNum(-src, ref);
		src = -tmpv;
	}
	return src;
}

// ��ȡĳ���½�����������Ϊref����ȷ��ref�����λ
double calFloorNum(double src, double ref){
	if (fabs(ref) < 1e-7)
		return ceil(src);
	int td = topDigit(ref);
	//cout << "td = " << td << endl;
	if (src > 0){
		if (td > 0){
			double tmp = src / pow((double)10, td - 1) + 0.000001;
			src = floor(tmp);
			src *= pow((double)10, td - 1);
		}
		else {
			double tmp = src * pow((double)10, -td) + 0.000001;
			src = floor(tmp);
			src /= pow((double)10, -td);
		}
	}
	else if(fabs(src) < 1e-13){
		return -pow(10.0, td);
	}
	else {
		double tmpv = calTopNum(-src, ref);
		src = -tmpv;
	}

	return src;
}

double getAmpl(double real, double imag){
	return sqrt(real*real + imag*imag);
}

double getAngl(double real, double imag){
	double angl = 0;
	if (real>0)                angl = atan(imag / real);
	else if (real == 0 && imag>0)  angl = CV_PI / 2;
	else if (real<0 && imag>0)   angl = atan(imag / real) + CV_PI;
	else if (real<0 && imag<0)   angl = atan(imag / real) - CV_PI;
	else if (real == 0 && imag<0)  angl = -CV_PI / 2;
	else if (real<0 && imag == 0)  angl = CV_PI;
	return angl;
}

// ���ź���
double sign(double x){
	if (x>0)
		return 1.0;
	else if (x == 0)
		return 0.0;
	else
		return -1.0;
}

// the real array X, returns an
//array F of real numbers, usually in the range 0.5 <= abs(F) < 1,
//	and an array E of integers, so that X = F .* 2.^E.
// ��һ��Ԫ��ΪF�� �ڶ���Ԫ��ΪE
pair<double, int> log2Double(double x){
	if (fabs(x) < 1e-10)
		return make_pair(0, 0);
	else if (x < 0){
		pair<double, int> t = log2Double(-x);
		return make_pair(-t.first, t.second);
	}
	else {
		pair<double, int> result;

		double tmp = log(x) / log(2.0);
		result.second = (int)ceil(tmp);
		result.first = x / pow(2.0, result.second);

		if (fabs(result.first - 1) < 1e-10){
			result.second++;
			result.first = x / pow(2.0, result.second);
		}
		return result;
	}
}

// ���ظ���ĳ����һ����������λ������3�Ļ�����1�� 0.5�Ļ�����0.1
// -0.02�Ļ�����0.01
double resolution(double x){
	if (x < 0) x = -x;
	double y = log(x)/log(10);
	//cout << "y = " << y << endl;
	double m = pow(10, floor(y));
	//cout << "m = " << m << endl;
	return m;
}

//����ָ����Χ�ڵ����������
// �������������������[dbLow, dbUpper)
double rnd(double dbLow, double dbUpper)
{
	double dbTemp = rand() / ((double)RAND_MAX + 1.0);
	return dbLow + dbTemp*(dbUpper - dbLow);
}

//���ظ�������������ȡ����ĸ�����
double ROUND(double dbA)
{
	return (double)((int)(dbA + 0.5));
}

//�õ�ǰʱ����ʼ��������ӣ���ֹÿ�����еĽ������ͬ
void setRandomSeed(){
	time_t tm;
	time(&tm);
	unsigned int nSeed = (unsigned int)tm;
	srand(nSeed);
}
