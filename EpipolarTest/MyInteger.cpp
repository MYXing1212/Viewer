#include"stdafx.h"
#include"MyInteger.h"

MyInteger::MyInteger()
{}

MyInteger::~MyInteger()
{}

// 找到不大于这个数的4的倍数
int get4DivisableNumLess(int src){
	return src &~3;
}

// 找到不小于这个数的4的倍数
int get4DivisableNumGreater(int src){
	return (src % 4) ? (get4DivisableNumLess(src) + 4) : (src);
}

// 返回≥input的最小的2的整数次幂
int roof2power(int input){
	double tmp = log((double)input) / log(2.0);
	if ((tmp - floor(tmp)) < 1e-10)
		return (int)pow(2, tmp);
	else 
		return (int)pow(2, floor(tmp)+1);
}

// 返回<=input的最大的2的整数次幂
int floor2power(int input){
	double tmp = log((double)input) / log(2.0);
	if ((tmp - floor(tmp)) < 1e-10)
		return (int)pow(2, tmp);
	else
		return (int)pow(2, floor(tmp));
}

// vector 码位倒置后的结果
void bitreverseVec(double* input, int len, int N){
	double* result = (double*)malloc(sizeof(double)*len);
	for (int i = 0; i < len; i++){
		if (i % 5000 == 0) printf("i = %d\n", i);
		result[i] = input[bitreverse(i,N)];
	}
	for (int i = 0; i < len; i++){
		input[i] = result[i];
	}
}

// 码位倒置  input的二进制为10010 倒置后为 01001
// ref为参考位数，比如3二进制 = 11，如果给出参考位数后，假如ref=5 那么3对应的二进制为00011
int bitreverse(int input, int ref){
	vector<int> bitseq;
	while(input){
		int tmp = input % 2;
		bitseq.push_back(tmp);
		input /= 2;
	}
	for (;ref > (int)bitseq.size();){
		bitseq.push_back(0);
	}
	int result = 0;
	for (int i = 0; i < (int)bitseq.size(); i++){
		//printf("%d", bitseq[i]);
		result = result * 2 + bitseq[i];
	}	
	//printf("\nresult = %d\n", result);
	return result;
}

// 判断是否是2的整数次幂
bool is2ofIntPow(int input){
	return (input == roof2power(input));
}

// 阶乘
int Fact(int n){
	int r = 1;
	while (n > 1){
		r = r*n;
		n--;
	}
	return r;
}

// 检测两个数是否相等
bool MyInteger::isEqual(int a,int b)
{
	return !(a^b);
}

// 检测一个数是否为对称数 
bool MyInteger::CheckSymmetryNum(int num)
{
	int tmp = getInvertedNum(num);
	if(tmp == num) return true;
	else           return false;
}

// 得到一个数的反序数
int MyInteger::getInvertedNum(int num)
{
	vector<int> figures;
	getAllFigures(num,figures,Inverted_Order);
	vector<int>::const_iterator it = figures.begin();
	int result = 0;
	for(;it!=figures.end();++it)
	{
		if(it == figures.begin())
			result = *it;
		else result = result*10 + *it;
	}
	return result;
}

// 得到一个数的位数
int MyInteger::getFiguresNum(int num)
{
	int n = 1;
	if(num<0) 
	{
		printf("该数为一负数!"); return num;
	}
	for(;num>9;num/=10) n++;
	return n;
}

// 得到一个数字的各位上的数 type = Positive_Order 从左到右 高位到低位
//                          type = Inverted_Order 从右到左 低位到高位
void MyInteger::getAllFigures(int num,vector<int>& figures,
	                 int type)
{
	if(num<0) 
	{
		printf("该数为一负数!"); return ;
	}
	for(;num>0;)
	{
		if(type == Positive_Order)
		{
			figures.insert(figures.begin(),num - num/10*10);
			num/=10;
		}
		else if(type == Inverted_Order)
		{
			figures.push_back(num-num/10*10);
			num/=10;
		}
	}
}

// 检测一个数是不是水仙花数
bool MyInteger::CheckNarcissisticNum(int num)
{
	if(num<0) 
	{
		printf("该数为一负数!"); return false;
	}
	vector<int> figures;
	getAllFigures(num,figures,Positive_Order);
	vector<int>::const_iterator it = figures.begin();
	int sum = 0;
	for(;it!=figures.end();++it)
		sum+=(*it)*(*it)*(*it);
	if(sum == num) return true;
	else           return false;
}

// 判断一个数是不是完全数
bool MyInteger::CheckPerfectNum(int num)
{
	vector<int> divisors;
	getDivisors(num,divisors,true);
	vector<int>::const_iterator it = divisors.begin();

	int sum;
	for(sum = 0;it!=divisors.end();++it)
		sum += *it;
	if(sum == 2*num)	return true;
	else        		return false;
}

// 找一个数的约数
void MyInteger::getDivisors(int num,vector<int>& divisors,bool IfAll )
{
	for(int i=1;i<=num;i++)
	{
        if(!IfAll&&(i==1||i==num))// 是否加上1和本身
			continue;
		if(num%i==0)
		{
			divisors.push_back(i);
		}
	}
}

// 计算最大公约数 greatest common divisor(gsd)
int  MyInteger::SolveGCD(int x, int y)
{
	if(x<=0||y<=0)
		printf("Error！！！\n");
	for(int i=min(x,y);i>0;i--)
		if(x%i==0&&y%i==0) return i;
	return 0;
}

// 计算最小公倍数 lowest common multiple(lcm)
int MyInteger::SolveLCM(int x, int y)
{
	if(x<=0||y<=0)
		printf("Error！！！\n");
	for(int i=max(x,y);i<=x*y;i++)
		if(i%x==0&&i%y==0) return i;
	return 0;
}

// 检测一个数是不是素数
bool MyInteger::CheckPrime(int x)
{
	if(x<=1) printf("Error！！！\n");
	for(int i = (int)sqrt((double)x);i>1;i--)
		if(x%i==0) return false;
	return true;
}

// 打印杨辉三角 r是行数
void MyInteger::PrintYangHuiTriangle(int r)
{
	int m,cnm,k;
	for(k=1;k<=40;k++) printf(" ");
	printf("%6d\n",1);               //输出第一行的1
	for(m=1;m<r;m++)
	{
		for(k=1;k<=40-3*m;k++)
			printf(" ");
		cnm=1;
		printf("%6d",cnm);
		for(k=1;k<=m;k++)
		{
			cnm=cnm*(m-k+1)/k;       //计算第m行的第k个数
			printf("%6d",cnm);
		}
		printf("\n");
	}
}

// 插入排序算法 ，M是数组的长度
void MyInteger::InsertSort(int a[], int M)    
{
	int temp;            // 临时变量
	int i, j;            // 循环变量
	for (i = 1; i < M; i++)
	{
		temp = a[i];            // 获取比较值
		for (j = i; j>0 && a[j - 1] > temp; j--)   // 前i个元素，如果有大元素交换
			a[j] = a[j - 1];                       // 移到当前位置
		a[j] = temp;                               // 将最后一个交换的j位置元素赋值temp
		/*for (int k = 0; k < M; k++)
			cout << a[k] << " ";
		cout << endl;*/
	}
}

// 冒泡排序算法，M是数组的长度
void MyInteger::Bubble(int a[], int M)
{
	int temp;    // 临时变量
	for (int i = 0; i < M - 1; i++)    // 元素比较界限
	{
		for (int j = 0; j < M - 1; j++)     // 该元素与加1位置元素进行比较
		{
			if (a[j]>a[j + 1])     // 把大元素放到位置右
			{
				temp = a[j];
				a[j] = a[j+1];
				a[j + 1] = temp;
			}
		}
		for (int k = 0; k < M; k++)
			cout << a[k] << " ";   // 输出每次排序的结果
		cout << endl;
	}
}

// 选择排序法，M是数组的长度
void MyInteger::SelectSort(int a[], int M)
{
	int pos;    // 目前最小的数字的位置
	int temp;    // temp存最小数字
	for (int i = 0; i < M; i++)
	{
		pos = i;
		temp = a[i];
		for (int j = i + 1; j < M; j++)
		{
			if (a[j] < temp)
			{
				pos = j;
				temp = a[j];
			}
		}
		a[pos] = a[i];
		a[i] = temp;
		for (int k = 0; k < M; k++)
			cout << a[k] << " ";
		cout << endl;
	}
}


// 打印九九乘法表
void MyInteger::PrintMultiplicationTable(void)
{
	for(int i=1;i<=9;i++)
	{
		for(int j=1;j<=i;j++)
			printf("%d×%d=%2d  ",i,j,i*j);
		printf("\n");
	}
}

// 分解质因数
void MyInteger::getPrimeFactors(int num,vector<int>& factors)
{
	if(num<=1) 
	{
		printf("该数无法进行质因数分解!"); return;
	}
	int k = num;
	for(int i=0;i<=999;i++)
	{
		if(num==1) break;
		for(int j=2;j<=k;j++)
		{
			if(num%j==0) 
			{
				factors.push_back(j);	
				num=num/j;break;
			}
			else continue;
		}
	}
}

// 数字小写变大写
string MyInteger::GetChineseNums(int n)
{
	int BIAO = 11;

	string str[19] = { " ", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", " ",
		"拾", "佰", "仟", "万", "拾", "佰", "仟", "亿" };
	long int  i = 0;      // 目标数字和索引值
	int temp[20] = { 0 };
	while (n)
	{
		temp[i] = n % 10;     // 取最低值
		n = n / 10;
		temp[i + 1] = BIAO + i / 2;   // 上一位的单位
		i += 2;
	}
	int j = i - 2;
	string result = "";
	for (; j >= 0; j--)
		result += str[temp[j]].c_str();
	return result;
}




/*
	#include<string>
	#include<d:\Programme\我的类 VS samples\数学类\整数运算类\MyInteger.h>
	#include<iostream>

	using namespace std;
	#define NUM 4

	void main()
	{
		MyInteger myInteger;
		double a[NUM];
		string eq[NUM];
		cout << "--------巧算24点--------" << endl;
		cout << "请输入4个数: " << endl;
		for (int i = 0; i < NUM; i++)
		{
			char buffer[20];
			int x;
			cin >> x;
			a[i] = x;
			_itoa_s(x, buffer, 10);
			eq[i] = buffer;
		}
		if (myInteger.Cal(NUM, a, eq))
			cout << "计算过程: " << eq[0] << endl;
		else
			cout << "该4个数构不成24" << endl;
	}

*/

/*
	equation是算式字符串
*/
bool MyInteger::Cal(int n, double *number, string *equation)
{
	/*cout << "n = " << n << endl;
	for (int t = 0; t < 4; t++)
		printf("%.2lf  ", number[t]);*/
	if (n == 1)
	{
		if (fabs(number[0] - RESULT) < PRECISION)   // 小于精度
			return true;
		else 
			return false;
	}
	for (int i = 0; i < n - 1; i++)                 // 没结束，继续循环
	{
		for (int j = i + 1; j < n; j++)
		{
			double a, b;
			string expa, expb;
			a = number[i];
			b = number[j];
			//cout << "a = " << a << ", b = " << b << endl;
			// 挪动后面的有效数字
			number[j] = number[n - 1];
			/*for (int t = 0; t < 4; t++)
				printf("%.2lf  ", number[t]);*/
			expa = equation[i];
			expb = equation[j];
			equation[j] = equation[n - 1];
			// a+b
			equation[i] = '(' + expa + '+' + expb + ')';
			number[i] = a + b;
			if (Cal(n - 1, number, equation))
				return true;
			// a-b
			equation[i] = '(' + expa + '-' + expb + ')';
			number[i] = a - b;
			if (Cal(n - 1, number, equation))
				return true;
			// b - a
			equation[i] = '(' + expb + '-' + expa + ')';
			number[i] = b-a;
			if (Cal(n - 1, number, equation))
				return true;
			// a*b
			equation[i] = '(' + expa + '*' + expb + ')';
			number[i] = a*b;
			if (Cal(n - 1, number, equation))
				return true;
			// a/b
			if (b != 0)     // 初始不为0
			{
				equation[i] = '(' + expa + '/' + expb + ')';
				number[i] = a / b;
				if (Cal(n - 1, number, equation))
					return true;
			}
			// b/a
			if (a != 0)     // 初始不为0
			{
				equation[i] = '(' + expb + '/' + expa + ')';
				number[i] = b / a;
				if (Cal(n - 1, number, equation))
					return true;
			}
			// 恢复
			number[i] = a;
			number[j] = b;
			equation[i] = expa;
			equation[j] = expb;
			// 恢复后
			/*cout << "\n恢复后" << endl;
			for (int t = 0; t < 4; t++)
				printf("%.2lf  ",number[t]);*/
		}
	}
	return false;
}