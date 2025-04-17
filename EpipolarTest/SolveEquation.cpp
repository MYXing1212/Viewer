#include"stdafx.h"
#include"SolveEquation.h"

using namespace cv;

// ���Է��� ��С���˽� AX = b   ��С���˽�Ϊx0 = A.inv() * b
double* LeastSquareSoluForLiEq(double* A, int m, int n, double* b){
	//double* A_inv = InvHighMat(A, m, n);
	Mat Amat(m, n, CV_64FC1, A);
	Mat bmat(m, 1, CV_64FC1, b);

	Mat x0 = InvMat(Amat)*bmat;

	double* result = initVector(m);

	memcpy(result, x0.ptr<double>(0), sizeof(double)*m);
	return  result;
}

// ������Է����� AX = b ;A��n*n����, b��n*1����
double* solveLiEq(double *A, double *b, int n){
	Mat Amat(n, n, CV_64FC1, A);
	Mat bmat(n, 1, CV_64FC1, b);
	
	Mat solu;
	solve(Amat, bmat, solu);
	double *result = initVector(n);
	memcpy(result, solu.ptr<double>(0), sizeof(double)*n);

	return result;
}

// ���һԪ���η���
vector<double> solveQuadraticEqu(double a, double b, double c){
	double delta = b*b - 4 * a*c;
	vector<double> result;
	if (delta < 0)
		return result;
	else if (delta == 0){
		result.push_back(-b / 2.0 / a);
	}
	else if (delta > 0){
		result.push_back((-b + sqrt(delta)) / 2.0 / a);
		result.push_back((-b - sqrt(delta)) / 2.0 / a);
	}
	return result;
}

// ׷�Ϸ�������Խ����Է����� Ax = b
// A��n*n�ľ���
// b��n*1������
// c�����Խ�������һ���Խ����ϵ�����
// d�����Խ�������һ���Խ����ϵ�����
double* solve3Diag(double* a, double* c, double* d, double* b, int n){
	double* x = initVector(n);
	double* y = initVector(n);
	double *p = initVector(n);
	double *q = initVector(n - 1);
	p[0] = a[0];
	for (int i = 0; i < n - 1; i++){
		q[i] = c[i] / p[i];
		p[i + 1] = a[i + 1] - d[i] * q[i];
	}
	y[0] = b[0] / p[0];
	for (int i = 1; i < n; i++){
		y[i] = (b[i] - d[i-1] * y[i - 1]) / p[i];
	}
	printVector(y, 5);
	x[n - 1] = y[n - 1];
	for (int i = n - 2; i >= 0; i--){
		x[i] = y[i] - q[i] * x[i + 1];
	}
	return x;
}

// �����Խ��߷��������ⷽ��
// ϵ������Ϊ���µ������ԽǾ���
//		��												��
//		�� a[1]    c[1]							   d[1] ��
//		�� d[2]    a[2]    c[2]							��
//		��           �v       �v       �v					��
//		��					�v       �v	    �v			��
//	A = ��						  d[n-1]  a[n-1]  c[n-1]��
//		�� c[n]					           d[n]    a[n]	��
//		��												��
// �����Է�����Ax=b��Ϊ�����Խ����Է����飬��༼���������������Ϊ����������Է�����
// ע������ʱ��c1��d1��λ�ã�����
vector<double> solveQuasi3Diag(vector<double> a, vector<double> c, vector<double> d, vector<double> b){
	int n = (int)a.size();
	vector<double> p(n - 1), q(n-2);
	p[0] = a[0];
	for (int i = 1; i < n - 1; i++){
		q[i - 1] = c[i - 1] / p[i - 1];
		p[i] = a[i] - d[i] * q[i - 1];
	}
	vector<double> s(n - 1);
	s[0] = d[0] / p[0];
	for (int i = 1; i < n - 2; i++){
		s[i] = -d[i] * s[i - 1] / p[i];
	}
	s[n - 2] = (c[n - 2] - d[n - 2] * s[n - 3]) / p[n - 2];
	vector<double> r(n);
	r[0] = c[n - 1];
	for (int i = 1; i < n - 1; i++){
		r[i] = -r[i - 1] * q[i - 1];
	}
	r[n - 2] = d[n - 1] - r[n - 3] * q[n - 3];
	r[n - 1] = a[n - 1] - sumProduct(r, s, n - 1);
	// ��Ly=b��Ux=y�ֱ�����y��xΪ
	vector<double> y(n);
	y[0] = b[0] / p[0];
	for (int i = 1; i < n - 1; i++){
		y[i] = (b[i] - d[i] * y[i - 1]) / p[i];
	}
	y[n - 1] = (b[n - 1] - sumProduct(r, y, n - 1)) / r[n - 1];
	vector<double> x(n);
	x[n - 1] = y[n - 1];
	x[n - 2] = y[n - 2] - s[n - 2] * x[n - 1];
	for (int i = n - 3; i >= 0; i--){
		x[i] = y[i] - q[i] * x[i + 1] - s[i] * x[n - 1];
	}
	return x;
}