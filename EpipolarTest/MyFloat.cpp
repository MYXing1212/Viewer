//#include"stdafx.h"
#include"stdafx.h"
#include"MyFloat.h"

using namespace cv;

// 生成高斯分布/正态分布的随机数
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

// 符号函数
double signValue(double input){
	if (input > 0)
		return 1.0;
	else if (input < 0)
		return -1.0;
	else
		return 0.0;
}

// 最高位数字 1代表是个位 -1代表小数点后1位
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

// 算取区间段上边界
double limitUp(double input, double ref){
	double tmp = input + pow(10.0, topDigit(ref) - 2);
	return floor(tmp / pow(10.0, topDigit(ref) - 2))*pow(10.0, topDigit(ref) - 2);
}

// 算取区间段下边界
double limitDown(double input, double ref){
	return input - pow(10.0, topDigit(ref) - 2);
}

// 算取某数上界数，参照数为ref 精确到ref的最高位
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

// 算取某数下界数，参照数为ref，精确到ref的最高位
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

// 符号函数
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
// 第一个元素为F， 第二个元素为E
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

// 返回高于某数的一个数量级单位，比如3的话返回1， 0.5的话返回0.1
// -0.02的话返回0.01
double resolution(double x){
	if (x < 0) x = -x;
	double y = log(x)/log(10);
	//cout << "y = " << y << endl;
	double m = pow(10, floor(y));
	//cout << "m = " << m << endl;
	return m;
}

//返回指定范围内的随机浮点数
// 产生的随机数在区间内[dbLow, dbUpper)
double rnd(double dbLow, double dbUpper)
{
	double dbTemp = rand() / ((double)RAND_MAX + 1.0);
	return dbLow + dbTemp*(dbUpper - dbLow);
}

//返回浮点数四舍五入取整后的浮点数
double ROUND(double dbA)
{
	return (double)((int)(dbA + 0.5));
}

//用当前时间点初始化随机种子，防止每次运行的结果都相同
void setRandomSeed(){
	time_t tm;
	time(&tm);
	unsigned int nSeed = (unsigned int)tm;
	srand(nSeed);
}
