#include"stdafx.h"
#include"MyInteger.h"

MyInteger::MyInteger()
{}

MyInteger::~MyInteger()
{}

// �ҵ��������������4�ı���
int get4DivisableNumLess(int src){
	return src &~3;
}

// �ҵ���С���������4�ı���
int get4DivisableNumGreater(int src){
	return (src % 4) ? (get4DivisableNumLess(src) + 4) : (src);
}

// ���ء�input����С��2����������
int roof2power(int input){
	double tmp = log((double)input) / log(2.0);
	if ((tmp - floor(tmp)) < 1e-10)
		return (int)pow(2, tmp);
	else 
		return (int)pow(2, floor(tmp)+1);
}

// ����<=input������2����������
int floor2power(int input){
	double tmp = log((double)input) / log(2.0);
	if ((tmp - floor(tmp)) < 1e-10)
		return (int)pow(2, tmp);
	else
		return (int)pow(2, floor(tmp));
}

// vector ��λ���ú�Ľ��
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

// ��λ����  input�Ķ�����Ϊ10010 ���ú�Ϊ 01001
// refΪ�ο�λ��������3������ = 11����������ο�λ���󣬼���ref=5 ��ô3��Ӧ�Ķ�����Ϊ00011
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

// �ж��Ƿ���2����������
bool is2ofIntPow(int input){
	return (input == roof2power(input));
}

// �׳�
int Fact(int n){
	int r = 1;
	while (n > 1){
		r = r*n;
		n--;
	}
	return r;
}

// ����������Ƿ����
bool MyInteger::isEqual(int a,int b)
{
	return !(a^b);
}

// ���һ�����Ƿ�Ϊ�Գ��� 
bool MyInteger::CheckSymmetryNum(int num)
{
	int tmp = getInvertedNum(num);
	if(tmp == num) return true;
	else           return false;
}

// �õ�һ�����ķ�����
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

// �õ�һ������λ��
int MyInteger::getFiguresNum(int num)
{
	int n = 1;
	if(num<0) 
	{
		printf("����Ϊһ����!"); return num;
	}
	for(;num>9;num/=10) n++;
	return n;
}

// �õ�һ�����ֵĸ�λ�ϵ��� type = Positive_Order ������ ��λ����λ
//                          type = Inverted_Order ���ҵ��� ��λ����λ
void MyInteger::getAllFigures(int num,vector<int>& figures,
	                 int type)
{
	if(num<0) 
	{
		printf("����Ϊһ����!"); return ;
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

// ���һ�����ǲ���ˮ�ɻ���
bool MyInteger::CheckNarcissisticNum(int num)
{
	if(num<0) 
	{
		printf("����Ϊһ����!"); return false;
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

// �ж�һ�����ǲ�����ȫ��
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

// ��һ������Լ��
void MyInteger::getDivisors(int num,vector<int>& divisors,bool IfAll )
{
	for(int i=1;i<=num;i++)
	{
        if(!IfAll&&(i==1||i==num))// �Ƿ����1�ͱ���
			continue;
		if(num%i==0)
		{
			divisors.push_back(i);
		}
	}
}

// �������Լ�� greatest common divisor(gsd)
int  MyInteger::SolveGCD(int x, int y)
{
	if(x<=0||y<=0)
		printf("Error������\n");
	for(int i=min(x,y);i>0;i--)
		if(x%i==0&&y%i==0) return i;
	return 0;
}

// ������С������ lowest common multiple(lcm)
int MyInteger::SolveLCM(int x, int y)
{
	if(x<=0||y<=0)
		printf("Error������\n");
	for(int i=max(x,y);i<=x*y;i++)
		if(i%x==0&&i%y==0) return i;
	return 0;
}

// ���һ�����ǲ�������
bool MyInteger::CheckPrime(int x)
{
	if(x<=1) printf("Error������\n");
	for(int i = (int)sqrt((double)x);i>1;i--)
		if(x%i==0) return false;
	return true;
}

// ��ӡ������� r������
void MyInteger::PrintYangHuiTriangle(int r)
{
	int m,cnm,k;
	for(k=1;k<=40;k++) printf(" ");
	printf("%6d\n",1);               //�����һ�е�1
	for(m=1;m<r;m++)
	{
		for(k=1;k<=40-3*m;k++)
			printf(" ");
		cnm=1;
		printf("%6d",cnm);
		for(k=1;k<=m;k++)
		{
			cnm=cnm*(m-k+1)/k;       //�����m�еĵ�k����
			printf("%6d",cnm);
		}
		printf("\n");
	}
}

// ���������㷨 ��M������ĳ���
void MyInteger::InsertSort(int a[], int M)    
{
	int temp;            // ��ʱ����
	int i, j;            // ѭ������
	for (i = 1; i < M; i++)
	{
		temp = a[i];            // ��ȡ�Ƚ�ֵ
		for (j = i; j>0 && a[j - 1] > temp; j--)   // ǰi��Ԫ�أ�����д�Ԫ�ؽ���
			a[j] = a[j - 1];                       // �Ƶ���ǰλ��
		a[j] = temp;                               // �����һ��������jλ��Ԫ�ظ�ֵtemp
		/*for (int k = 0; k < M; k++)
			cout << a[k] << " ";
		cout << endl;*/
	}
}

// ð�������㷨��M������ĳ���
void MyInteger::Bubble(int a[], int M)
{
	int temp;    // ��ʱ����
	for (int i = 0; i < M - 1; i++)    // Ԫ�رȽϽ���
	{
		for (int j = 0; j < M - 1; j++)     // ��Ԫ�����1λ��Ԫ�ؽ��бȽ�
		{
			if (a[j]>a[j + 1])     // �Ѵ�Ԫ�طŵ�λ����
			{
				temp = a[j];
				a[j] = a[j+1];
				a[j + 1] = temp;
			}
		}
		for (int k = 0; k < M; k++)
			cout << a[k] << " ";   // ���ÿ������Ľ��
		cout << endl;
	}
}

// ѡ�����򷨣�M������ĳ���
void MyInteger::SelectSort(int a[], int M)
{
	int pos;    // Ŀǰ��С�����ֵ�λ��
	int temp;    // temp����С����
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


// ��ӡ�žų˷���
void MyInteger::PrintMultiplicationTable(void)
{
	for(int i=1;i<=9;i++)
	{
		for(int j=1;j<=i;j++)
			printf("%d��%d=%2d  ",i,j,i*j);
		printf("\n");
	}
}

// �ֽ�������
void MyInteger::getPrimeFactors(int num,vector<int>& factors)
{
	if(num<=1) 
	{
		printf("�����޷������������ֽ�!"); return;
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

// ����Сд���д
string MyInteger::GetChineseNums(int n)
{
	int BIAO = 11;

	string str[19] = { " ", "Ҽ", "��", "��", "��", "��", "½", "��", "��", "��", " ",
		"ʰ", "��", "Ǫ", "��", "ʰ", "��", "Ǫ", "��" };
	long int  i = 0;      // Ŀ�����ֺ�����ֵ
	int temp[20] = { 0 };
	while (n)
	{
		temp[i] = n % 10;     // ȡ���ֵ
		n = n / 10;
		temp[i + 1] = BIAO + i / 2;   // ��һλ�ĵ�λ
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
	#include<d:\Programme\�ҵ��� VS samples\��ѧ��\����������\MyInteger.h>
	#include<iostream>

	using namespace std;
	#define NUM 4

	void main()
	{
		MyInteger myInteger;
		double a[NUM];
		string eq[NUM];
		cout << "--------����24��--------" << endl;
		cout << "������4����: " << endl;
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
			cout << "�������: " << eq[0] << endl;
		else
			cout << "��4����������24" << endl;
	}

*/

/*
	equation����ʽ�ַ���
*/
bool MyInteger::Cal(int n, double *number, string *equation)
{
	/*cout << "n = " << n << endl;
	for (int t = 0; t < 4; t++)
		printf("%.2lf  ", number[t]);*/
	if (n == 1)
	{
		if (fabs(number[0] - RESULT) < PRECISION)   // С�ھ���
			return true;
		else 
			return false;
	}
	for (int i = 0; i < n - 1; i++)                 // û����������ѭ��
	{
		for (int j = i + 1; j < n; j++)
		{
			double a, b;
			string expa, expb;
			a = number[i];
			b = number[j];
			//cout << "a = " << a << ", b = " << b << endl;
			// Ų���������Ч����
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
			if (b != 0)     // ��ʼ��Ϊ0
			{
				equation[i] = '(' + expa + '/' + expb + ')';
				number[i] = a / b;
				if (Cal(n - 1, number, equation))
					return true;
			}
			// b/a
			if (a != 0)     // ��ʼ��Ϊ0
			{
				equation[i] = '(' + expb + '/' + expa + ')';
				number[i] = b / a;
				if (Cal(n - 1, number, equation))
					return true;
			}
			// �ָ�
			number[i] = a;
			number[j] = b;
			equation[i] = expa;
			equation[j] = expb;
			// �ָ���
			/*cout << "\n�ָ���" << endl;
			for (int t = 0; t < 4; t++)
				printf("%.2lf  ",number[t]);*/
		}
	}
	return false;
}