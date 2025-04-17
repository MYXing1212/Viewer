#include"stdafx.h"
#include"MyMatrix.h"
/* Functions to compute the integral, and the 0th and 1st derivative of the
Gaussian function 1/(sqrt(2*PI)*sigma)*exp(-0.5*x^2/sigma^2) */

using namespace cv;


// �����󡿳�ʼ�� ��ʼֵ��Ϊ0
double** initMatrix(int row, int col){
	double** result;
	result = (double**)malloc(sizeof(double*)*row);
	for (int i = 0; i < row; i++){
		*(result + i) = (double*)malloc(sizeof(double)*col);
		for (int j = 0; j < col; j++){
			result[i][j] = 0;
		}
	}
	return result;
}

// �����󡿳�ʼ�� ��ʼֵ��Ϊ0
int** initMatrixInt(int row, int col){
	int** result;
	result = (int**)malloc(sizeof(int*)*row);
	for (int i = 0; i < row; i++){
		*(result + i) = (int*)malloc(sizeof(int)*col);
		for (int j = 0; j < col; j++){
			result[i][j] = 0;
		}
	}
	return result;
}

// �����󡿳�ʼ��
double* initMatrix(Mat A){
	if (A.type() != CV_64FC1 && A.type() != CV_64F){
		printf("double** initMatrix(Mat)  ��������������Ͳ�ƥ�䣬ӦΪdouble��\n");
		return initVector(1);
	}
	double* result = initVector(A.rows*A.cols);
	memcpy(result, A.ptr<double>(0), A.rows*A.cols*sizeof(double));
	return result;
}

// �����󡿳�ʼ����λ��
double** eye(int len){
	double **result = initMatrix(len, len);
	for (int i = 0; i < len; i++){
		result[i][i] = 1;
	}
	return result;
}

// ��������󡿾��ȷֲ� ����a��b
Mat randnMatUniform(Size size, double a, double b){
	Mat result(size, CV_64FC1);
	Mat A = (Mat_<double>(1, 1) << a);
	Mat B = (Mat_<double>(1, 1) << b);
	unsigned int optional_seed = (unsigned int)time(NULL);
	cv::RNG rng(optional_seed);
	rng.fill(result, cv::RNG::UNIFORM, A, B);
	return result;
}

// �����������ֵ�ʸ�˹�ֲ�������Size��mean����std
Mat randnMat(Size size, double mean, double std){
	Mat result(size, CV_64FC1);
	srand((unsigned)time(NULL));
	GaussRand();

	double* p = result.ptr<double>(0);

	for (int i = 0; i < result.rows*result.cols; i++){
		*p++ = GaussRand(mean, std);
	}
	return result;
}

// ����ʼ���������ʼ�� ԭʼ����src �е�ĳЩ �±�ΪIdxs��Ԫ�ع��ɵľ���
Mat initMatIdxs(Mat src, Mat idxs){
	Mat result = Mat::zeros(idxs.size().area(), 1, src.type());
	for (int i = 0; i < result.size().area(); i++){
		if (src.type() == CV_32SC1){
			result.ptr<int>(0)[i] = src.ptr<int>(0)[idxs.ptr<int>(0)[i]];
		}
		else if (src.type() == CV_64FC1){
			result.ptr<double>(0)[i] = src.ptr<double>(0)[idxs.ptr<int>(0)[i]];
		}
	}
	return result;
}

// ����ʼ���������ʼ�� ԭʼ����src �е�ĳЩ�к�ĳЩ��Ԫ�ع����µľ���
Mat initMatRowColIdxs(Mat src, Mat row_idxs, Mat col_idxs){
	int r = (row_idxs.data) ? row_idxs.size().area() : src.rows;
	int c = (col_idxs.data) ? col_idxs.size().area() : src.cols;

	Mat result = Mat::zeros(r, c, src.type());
	for (int i = 0; i < r; i++){
		for (int j = 0; j < c; j++){
			if (src.type() == CV_32SC1){
				if (row_idxs.data && col_idxs.data)
					result.at<int>(i, j) = src.at<int>(row_idxs.ptr<int>(0)[i], col_idxs.ptr<int>(0)[j]);
				else if (row_idxs.data && !col_idxs.data)
					result.at<int>(i, j) = src.at<int>(row_idxs.ptr<int>(0)[i], j);
				else if (!row_idxs.data && col_idxs.data)
					result.at<int>(i, j) = src.at<int>(i, col_idxs.ptr<int>(0)[j]);
			}
			else if (src.type() == CV_64FC1){
				if (row_idxs.data && col_idxs.data)
					result.at<double>(i, j) = src.at<double>(row_idxs.ptr<int>(0)[i], col_idxs.ptr<int>(0)[j]);
				else if (row_idxs.data && !col_idxs.data)
					result.at<double>(i, j) = src.at<double>(row_idxs.ptr<int>(0)[i], j);
				else if (!row_idxs.data && col_idxs.data)
					result.at<double>(i, j) = src.at<double>(i, col_idxs.ptr<int>(0)[j]);
			}
		}
	}
	return result;
}

// // ��ĳЩ�� ĳ�� �з�Χ ��Χ�С�����ѡȡĳЩ�й����µľ���
Mat colSelect(Mat src, int startIdx, int endIdx){
	CV_Assert(startIdx >= 0 && startIdx < src.cols);
	CV_Assert(endIdx >= 0 && endIdx < src.cols);
	CV_Assert(endIdx >= startIdx);
	return src.colRange(startIdx, endIdx + 1);
}

// // ��ĳЩ�� ĳ�� �з�Χ ��Χ�С�����ѡȡĳЩ�й����µľ���
Mat rowSelect(Mat src, int startIdx, int endIdx){
	CV_Assert(startIdx >= 0 && startIdx < src.rows);
	CV_Assert(endIdx >= 0 && endIdx < src.rows);
	CV_Assert(endIdx >= startIdx);
	return src.rowRange(startIdx, endIdx + 1);
}

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
Mat colSelect(Mat src, set<int> colIndexs){
	set<int>::iterator k = colIndexs.begin();
	int c = (int)colIndexs.size();
	Mat result = Mat::zeros(src.rows, c, src.type());
	for (int i = 0; k != colIndexs.end(); k++, i++){
		src.col(*k).copyTo(result.col(i));
	}
	return result;
}

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
Mat colSelect(Mat src, vector<int> colIndexs){
	vector<int>::iterator k = colIndexs.begin();
	int c = (int)colIndexs.size();
	Mat result = Mat::zeros(src.rows, c, src.type());
	for (int i = 0; k != colIndexs.end(); k++, i++){
		src.col(*k).copyTo(result.col(i));
	}
	return result;
}

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
Mat rowSelect(Mat src, vector<int> rowIndexs){
	vector<int>::iterator k = rowIndexs.begin();
	int r = (int)rowIndexs.size();
	Mat result = Mat::zeros(r, src.cols, src.type());
	for (int i = 0; k != rowIndexs.end(); k++, i++){
		src.row(*k).copyTo(result.row(i));
	}
	return result;
}

// // ��ĳЩ�� ĳ�С�����ѡȡĳЩ�й����µľ���
Mat rowSelect(Mat src, set<int> rowIndexs){
	set<int>::iterator k = rowIndexs.begin();
	int c = (int)rowIndexs.size();
	Mat result = Mat::zeros(c, src.cols, src.type());
	for (int i = 0; k != rowIndexs.end(); k++, i++){
		src.row(*k).copyTo(result.row(i));
	}
	return result;
}

// ����ʼ�����ԽǾ��� data��һά����
Mat diagMat(Mat data){
	int n = (int)data.total();
	Mat result = Mat::zeros(n, n, data.type());
	for (int i = 0; i < n; i++){
		if (data.type() == CV_32SC1)
			result.at<int>(i, i) = data.ptr<int>(0)[i];
		if (data.type() == CV_64FC1)
			result.at<double>(i, i) = data.ptr<double>(0)[i];
	}
	return result;
}

// ��������� ��ʼֵ���0��1֮��
Mat randMat(int row, int col){
	//srand((unsigned)time(NULL));
	rndDouble();
	Mat result = Mat::zeros(row, col, CV_64FC1);
	double *p = result.ptr<double>(0);
	for (int i = 0; i < result.rows * result.cols; i++){
		*p++ = rndDouble();
	}
	return result;
}

// ��������� ��ʼֵ���0��1֮��
Mat randMatFloat(int row, int col)
{
	Mat result = Mat::zeros(row, col, CV_32FC1);
	float *p = result.ptr<float>(0);
	for (int i = 0; i < result.rows * result.cols; i++)
	{
		*p++ = rndDouble();
	}
	return result;
}

// �����󡿳�ʼ������ʼֵ��� 0��1֮��
double** randMatrix(int row, int col){
	double **result = initMatrix(row, col);
	srand((unsigned)time(NULL));
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = rndDouble();
		}
	}
	return result;
}

// �����󡿶Խ�������Ϊ�Խ�Ԫ��Vec������Ԫ��Ϊ0
double** diagMatrix(double*vec, int len){
	double** result = initMatrix(len, len);
	for (int i = 0; i < len; i++){
		result[i][i] = vec[i];
	}
	return result;
}

// �����󡿸���
double* copyMat(double* input, int row, int col){
	double* result = initVector(row*col);
	for (int i = 0; i < row*col; i++)
		result[i] = input[i];
	return result;
}

// ������ ����
void copyMatInt(int**src, int ** dst, int row, int col)
{
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			dst[i][j] = src[i][j];
		}
	}
}

// �����󡿸���
int** copyMatInt(int** src, int row, int col){
	int **result = initMatrixInt(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = src[i][j];
		}
	}
	return result;
}

// �����󡿸���
double** copyMat(double** src, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = src[i][j];
		}
	}
	return result;
}

// �����󡿷�����ת���� ��z��
double** Rot_z(double angle){
	double **result = eye(3);
	result[0][0] = cos(angle);
	result[0][1] = -sin(angle);
	result[1][0] = sin(angle);
	result[1][1] = cos(angle);
	return result;
}

// �����󡿷�����ת���� ��x��
double** Rot_x(double angle){
	double**result = eye(3);
	result[1][1] = cos(angle);
	result[1][2] = -sin(angle);
	result[2][1] = sin(angle);
	result[2][2] = cos(angle);
	return result;
}

// �����󡿳�ʼ�� ���������г�ʼ��
double** initMatrix(double* vec, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			int tmp = col * i + j;
			*(*(result + i) + j) = *(vec + tmp);
		}
	}
	return result;
}

// �����󡿵õ�ĳһ�� ע��ӵ�0�п�ʼ
double* rowM(double* mat, int row, int col, int rowNo){
	double *result = initVector(col);
	if (rowNo > row){
		printf("rowM(double*, int, int ,int) ��ȡ�����ĳһ�У� ERROR ��������Χ������");
		return result;
	}


	for (int i = 0; i < col; i++){
		result[i] = mat[rowNo*col + i];
	}
	return result;
}

// �����󡿵õ�ĳһ��
double* rowM(double** mat, int rowOffset){
	return mat[rowOffset - 1];
}

// �������������
void exchange2rows(double* mat, int row, int col, int rowNo1, int rowNo2){
	if (rowNo1 >= row || rowNo2 >= row){
		printf("exchange2rows(double*, int, int ,int, int) ���������ĳһ�У� ERROR ��������Χ������");
		return;
	}

	// Ҫ������������ͬһ��
	if (rowNo1 == rowNo2){
		return;
	}

	for (int i = 0; i < col; i++){
		double tmp = mat[rowNo1*col + i];
		mat[rowNo1*col + i] = mat[rowNo2*col + i];
		mat[rowNo2*col + i] = tmp;
	}
}

// �����󡿵õ�����ĳһ��
double* colM(double** mat, int row, int colOffset){
	double *result = initVector(row);
	for (int i = 0; i < row; i++)
		result[i] = mat[i][colOffset - 1];
	return result;
}

// ������ת��
double** T_Mat(double **src, int row, int col){
	double **result = initMatrix(col, row);
	for (int i = 0; i < col; i++){
		for (int j = 0; j < row; j++){
			result[i][j] = src[j][i];
		}
	}
	return result;
}

// ������ת��
double* T_Mat(double *src, int row, int col){
	double *result = initVector(row*col);
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			result[i*row + j] = src[j*col + i];
		}
	}
	return result;
}

// ����������
double* InvMat(double *mat, int n){
	Mat input(n, n, CV_64FC1, mat);
	Mat invMat = input.inv();

	double *result = initVector(n * n);
	double *pt = invMat.ptr<double>(0);
	for (int i = 0; i < n*n; i++){
		result[i] = *pt++;
	}
	return result;

}

// �����������Ĺ����棨M*N�ľ���r(A) = N, �����ȣ� ʵ���� 
double* InvHighMat(double* A, int m, int n){
	double* result = initVector(m*n);
	if (n > m)
	{
		printf("InvHighMat(double*, int, int) �������Ǹ��󣡣���\n");
		return result;
	}
	double *A_T = T_Mat(A, m, n);			// A ��ת��
	double *ATA = MmulM(A_T, A, n, m, n);	// A.t() * A
	//printMatrix(ATA, 2,2);				
	double *ATA_inv = InvMat(ATA, n);		// (A.t()*A).inv()
	//printMatrix(ATA_inv, 2, 2);
	//result = MmulM(ATA_inv, A_T, n, n, m);	// ��A.t()*A��.inv()��*A.t()
	return result;
}

// �����������Ĺ����棨M*N�ľ���r(A) = M, �����ȣ�ʵ����
double* InvLowMat(double* A, int m, int n){
	double* result = initVector(m*n);
	if (n < m)
	{
		printf("InvLowMat(double*, int, int) �������ǵ��󣡣���\n");
		return result;
	}
	double *A_T = T_Mat(A, m, n);			// A.t()
	double *AAT = MmulM(A, A_T, m, n, m);	// A*A.t()
	double *AAT_inv = InvMat(AAT, m);		// (A*A.t()).inv()
	//printMatrix(A_T, n, m);
	result = MmulM(A_T, AAT_inv, n, m, m);	// A.t()*[(A*A.t()).inv()]
	return result;
}

// ����������
double** InvMat(double** ppDbMat, int nLen)
{
	double *pDbSrc = new double[nLen*nLen];

	int *is, *js, i, j, k;
	// ����Ҫ������������
	int nCnt = 0;
	for (i = 0; i < nLen; i++)
	{
		for (j = 0; j < nLen; j++)
		{
			pDbSrc[nCnt++] = ppDbMat[i][j];
		}
	}


	double d, p;
	is = new int[nLen];
	js = new int[nLen];

	for (k = 0; k < nLen; k++)	{
		d = 0.0;
		for (i = k; i < nLen; i++){
			for (j = k; j < nLen; j++){

				p = fabs(pDbSrc[i*nLen + j]);		// �ҵ�����ֵ����ϵ��
				if (p>d)	{
					d = p;

					// ��¼����ֵ����ϵ�����С�������
					is[k] = i;
					js[k] = j;
				}
			}
		}
		if (d + 1.0 == 1.0)	{					// ϵ��ȫ��0��ϵ������Ϊ0�󣬴�ʱΪ�������
			delete is;
			delete js;
			printf("��  Error!  ��������󣬲������棡����\n");
			return NULL;
		}
		if (is[k] != k)		{					//	��ǰ�в��������Ԫ��
			for (j = 0; j < nLen; j++)	{
				// ��������Ԫ��
				p = pDbSrc[k*nLen + j];
				pDbSrc[k*nLen + j] = pDbSrc[(is[k] * nLen) + j];
				pDbSrc[(is[k])*nLen + j] = p;
			}
		}

		if (js[k] != k)	{						// ��ǰ�в��������Ԫ��
			for (i = 0; i < nLen; i++){
				// ��������Ԫ��
				p = pDbSrc[i*nLen + k];
				pDbSrc[i*nLen + k] = pDbSrc[i*nLen + (js[k])];
				pDbSrc[i*nLen + (js[k])] = p;
			}
		}

		pDbSrc[k*nLen + k] = 1.0 / pDbSrc[k*nLen + k];		// ����Ԫ�ĵ���

		// a[k,j] a[k,k]  -> a[k][j]
		for (j = 0; j < nLen; j++)	{
			if (j != k)	{
				pDbSrc[k*nLen + j] *= pDbSrc[k*nLen + k];
			}
		}

		// a[i,j] - a[i,k]a[k,j]  -> a[i,j]
		for (i = 0; i < nLen; i++)	{
			if (i != k)	{
				for (j = 0; j < nLen; j++){
					if (j != k){
						pDbSrc[i*nLen + j] -= pDbSrc[i*nLen + k] * pDbSrc[k*nLen + j];
					}
				}
			}
		}

		// -a[i,k]a[k,k]  -> a[i,k]
		for (i = 0; i < nLen; i++)
		{
			if (i != k){
				pDbSrc[i*nLen + k] *= -pDbSrc[k*nLen + k];
			}
		}

	}

	for (k = nLen - 1; k >= 0; k--){
		// �ָ���
		if (js[k] != k){
			for (j = 0; j < nLen; j++){
				p = pDbSrc[k*nLen + j];
				pDbSrc[k*nLen + j] = pDbSrc[(js[k])*nLen + j];
				pDbSrc[(js[k])*nLen + j] = p;
			}
		}
		// �ָ���
		if (is[k] != k){
			for (i = 0; i < nLen; i++){
				p = pDbSrc[i*nLen + k];
				pDbSrc[i*nLen + k] = pDbSrc[i*nLen + (is[k])];
				pDbSrc[i*nLen + (is[k])] = p;
			}
		}

	}


	// ��������ƻ�ϵ������ppDbMat
	nCnt = 0;
	for (i = 0; i < nLen; i++){
		for (j = 0; j < nLen; j++){
			ppDbMat[i][j] = pDbSrc[nCnt++];
		}
	}

	double** result = initMatrix(pDbSrc, nLen, nLen);

	// �ͷſռ�
	delete is;
	delete js;
	delete pDbSrc;

	return result;
}

// ���������
double** sumM(double** mat1, double** mat2, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = mat1[i][j] + mat2[i][j];
		}
	}
	return result;
}

// ���������
double *sumM(double *a, double *b, int row, int col)
{
	double* result = initVector(row*col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			result[i*col + j] = a[i*col + j] + b[i*col + j];
		}
	}
	return result;
}

// ��������� ��֤input��ͨ����Ϊ1
double sumM(Mat input){
	CV_Assert(input.channels() == 1);
	return sum(input)[0];
}

// �����󡿱����任 ÿ��Ԫ�س���һϵ��
double** scaleM(double** mat, double scale, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = mat[i][j] * scale;
		}
	}
	return result;
}

// ���������
double** subM(double** mat, double offset, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = mat[i][j] - offset;
		}
	}
	return result;
}



// ���������
double *subM(double *a, double *b, int row, int col)
{
	double* result = initVector(row*col);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			result[i*col + j] = a[i*col + j] - b[i*col + j];
		}
	}
	return result;
}

// ��������
double traceM(double** mat, int size) {
	double sum = 0;
	for (int i = 0; i < size; i++){
		sum += mat[i][i];
	}
	return sum;
}

// ��������
double traceM(double *m, int n)
{
	double sum = 0.0;

	for (int i = 0; i < n; i++)
		sum += m[i*n + i];

	return sum;
}

// ��������ȡ�
int rankM(Mat src){
	Mat _eig, _vec;
	cv::eigen(src.t()*src, _eig, _vec);

	//cout <<"_eig = "<< _eig << endl;
	for (int i = 0; i < _eig.rows; i++){
		//cout << _eig.at<double>(i, 0) << endl;
		if (fabs(_eig.at<double>(i, 0)) < 1e-7)
			return i;
	}
	return _eig.rows;
}

// ��������ȡ�
int rankM(double *mat, int m, int n){
	Mat src(m, n, CV_64FC1, mat);
	Mat _eig, _vec;
	cv::eigen(src.t()*src, _eig, _vec);

	//cout << _eig << endl;
	//cout << _vec << endl;

	for (int i = 0; i < _eig.rows; i++){
		//cout << _eig.at<double>(i, 0) << endl;
		if (_eig.at<double>(i, 0) < 1e-7)
			return i;
	}
	return _eig.cols;
}

// ������ʽ��ֵ
double detM(Mat src){
	if (src.cols != src.rows){
		printf("ERROR���� --> double detM(Mat), �������ʽ��ֵ �������Ƿ���!!");
		return 0;
	}
	return determinant(src);
}

// ������ʽ��ֵ
double detM(double* mat, int n){
	Mat src(n, n, CV_64FC1, mat);
	return detM(src);
}

// ��Э�������������Э�������
Mat covMat(Mat A){
	CV_Assert(A.rows > 0 && A.cols > 0);
	vector<Mat> colArray;
	Mat result = Mat::zeros(A.size(), A.type());
	for (int i = 0; i < A.cols; i++){
		colArray.push_back(A.col(i).clone());
		//cout << varV(colArray[i]) << endl;
	}
	for (int i = 0; i < A.rows; i++){
		for (int j = i; j < A.cols; j++){
			//if (i == j)
			//	result.at<double>(i, i) = covV(colArray[i]);
			//else
			result.at<double>(i, j) = covV(colArray[i], colArray[j]);
			if (i != j)
				result.at<double>(j, i) = result.at<double>(i, j);
		}
	}

	return result;
}

// ������ ���ϵ������
Mat corrcoef(Mat A){
	CV_Assert(A.rows > 0 && A.cols > 0);
	vector<Mat> colArray;
	Mat result = Mat::zeros(A.size(), A.type());
	for (int i = 0; i < A.cols; i++){
		colArray.push_back(A.col(i).clone());
		//cout << varV(colArray[i]) << endl;
	}
	for (int i = 0; i < A.rows; i++){
		for (int j = i; j < A.cols; j++){
			result.at<double>(i, j) = coefV(colArray[i], colArray[j]);
			if (i != j)
				result.at<double>(j, i) = result.at<double>(i, j);
		}
	}

	return result;
}

// ��������������� mat1������Ӧ����mat2������
double** MmulM(double** mat1, double** mat2, int row1, int col1, int col2){
	double** result = initMatrix(row1, col2);
	for (int i = 0; i < row1; i++){
		for (int j = 0; j < col2; j++){
			result[i][j] = dotV(rowM(mat1, i + 1), colM(mat2, col1, j + 1), col1);
		}
	}
	return result;
}

// ��������������� mat1������nӦ����mat2������n   mat1 m*n    mat2 n*k
float* MmulM(float *p0, float *p1, int m, int n, int k){
	float *p2 = initVectorf(m*k);
	for (int i = 0; i <= m - 1; i++){
		for (int j = 0; j <= k - 1; j++)
		{
			*(p2 + i*k + j) = 0;
			for (int l = 0; l <= n - 1; l++)
				*(p2 + i*k + j) = (*(p2 + i*k + j)) + (*(p0 + i*n + l))*(*(p1 + l*k + j));
		}
	}
	return p2;
}

// ��������������� mat1������Ӧ����mat2������
double* MmulM(double *p, double *y, int row1, int col1, int col2)
{
	Mat _p(row1, col1, CV_64F, p);
	Mat _y(col1, col2, CV_64F, y);
	Mat mul = _p*_y;

	double *result = initVector(row1 * col2);
	double *pt = mul.ptr<double>(0);
	for (int i = 0; i < row1*col2; i++){
		result[i] = *pt++;
	}
	return result;
}

// ������ �������ӦԪ����ˣ�mat1�ĳߴ�Ӧ�õ���mat2�ĳߴ�
double* MmulMEle(double* m1, double* m2, int row, int col){
	Mat _m1(row, col, CV_64F, m1);
	Mat _m2(row, col, CV_64F, m2);
	Mat mul = _m1.mul(_m2);

	double *result = initVector(row * col);
	double *pt = mul.ptr<double>(0);
	for (int i = 0; i < row*col; i++){
		result[i] = *pt++;
	}
	return result;
}

// ������ �������ӦԪ����ˣ�mat1�ĳߴ�Ӧ�õ���mat2�ĳߴ�
float* MmulMEle(float* m1, float* m2, int row, int col){
	float *result = initVectorf(row * col);
	for (int i = 0; i < row*col; i++){
		result[i] = m1[i] * m2[i];
	}
	return result;
}

// �����󡿴�ӡ����
void printMatrix(int **m, int row, int col){
	printf("\n  [\n");
	for (int i = 0; i < row; i++){
		printf("     ");
		for (int j = 0; j < col; j++){
			//if (abs(m[i][j]) < 1e-6) m[i][j] = 0;
			if (m[i][j] >= 0) printf(" ");
			printf("%d ", m[i][j]);
		}printf("\n");
	}
	printf("   ];\n");
}

// �����󡿴�ӡ����
void printMatrix(double **m, int row, int col){
	printf("\n  [\n");
	for (int i = 0; i < row; i++){
		printf("     ");
		for (int j = 0; j < col; j++){
			if (fabs(m[i][j]) < 1e-6) m[i][j] = 0;
			if (m[i][j] >= 0) printf(" ");
			printf("%.7lf ", m[i][j]);
		}printf("\n");
	}
	printf("   ];\n");
}

// �����󡿴�ӡ����
void printMatrix(double *m, int row, int col){
	printf("\n  [\n");
	for (int i = 0; i < row; i++){
		printf("     ");
		for (int j = 0; j < col; j++){
			if (fabs(m[i*col + j]) < 1e-6) m[i*col + j] = 0.0f;
			if (m[i*col + j] >= 0) printf(" ");
			printf("%.7f ", m[i*col + j]);
		}printf("\n");
	}
	printf("   ];\n");
}



// �����󡿼���Գƾ�����������ֵ�Ͷ�Ӧ���������� �����ǶԳƾ��󣡣���
void getMaxMatEigen(double* m, double& eigen, vector<double> &q, int n){
	double *vec = new double[n*n];
	double *eig = new double[n];

	Mat _m(n, n, CV_64F, m);
	Mat _vec(n, n, CV_64F, vec);
	Mat _eig(n, 1, CV_64F, eig);

	cv::eigen(_m, _eig, _vec);
	eigen = eig[0];
	q.resize(n);
	for (int i = 0; i < n; i++)
		q[i] = vec[i];

	delete[] vec;
	delete[] eig;
}

// �����󡿼���Գƾ������ֵ��С������ֵ�Ͷ�Ӧ���������� �����ǶԳƾ��󣡣���
double* getAbsMinMatEigen(double* a, double& eigen, int n){
	double *vec = new double[n*n];
	double *eig = new double[n];
	double *result = new double[n];

	Mat _m(n, n, CV_64F, a);
	Mat _vec(n, n, CV_64F, vec);
	Mat _eig(n, 1, CV_64F, eig);

	cv::eigen(_m, _eig, _vec);

	int minNo = 0;
	eigen = fabs(eig[0]);
	for (int i = 0; i < n; i++){
		if (eigen > fabs(eig[i])){
			minNo = i;
			eigen = fabs(eig[i]);
		}
	}

	for (int i = 0; i < n; i++){
		result[i] = vec[minNo*n + i];
	}
	return result;
}

// �����󡿼��� ����ֵ����������
int getEigensAndVecs(Mat input, Mat& eigens, Mat& vectors){
	cv::eigen(input, eigens, vectors);
	return rankM(input);
}

// �����󡿼��� ����ֵ���������� ��ʹ�����������ʱ�� 
// eigens �� vectors Ӧ�����Ѿ�������ڴ��
int getEigensAndVecs(double *input, int n, double*eigens, double* vectors){
	Mat mat(n, n, CV_64FC1, input);

	Mat _eig, _vec;
	int r = getEigensAndVecs(mat, _eig, _vec);

	for (int i = 0; i < n; i++){
		eigens[i] = _eig.at<double>(i, 0);
		for (int j = 0; j < n; j++){
			vectors[i*n + j] = _vec.at<double>(i, j);
		}
	}
	return r;
}

Mat householder(Mat x, Mat y){
	int sign = 1;
	if (x.at<double>(0, 0) < 0)
		sign = -1;
	double nx = norm(x);
	double ny = norm(y);


	double rho = 1;

	if (fabs(ny) >1e-7)
		rho = -1 * sign*nx / ny;

	Mat tmpy = y*rho;

	Mat diff_xy = x - tmpy;

	normalize(diff_xy, diff_xy);
	Mat I = Mat::eye(x.rows, x.rows, x.type());

	Mat diff_xy_mul = -2 * diff_xy*diff_xy.t();

	return (diff_xy_mul + I);
}

// �ֿ���� A, B�ڶԽ��ߣ�����Ϊ0
Mat blkdiag(Mat A, Mat B){
	Mat result = Mat::zeros(A.rows + B.rows, A.cols + B.cols, A.type());
	A.copyTo(result(Rect(0, 0, A.cols, A.rows)));
	B.copyTo(result(Rect(A.cols, A.rows, B.cols, B.rows)));
	return result;
}

Mat blkdiag(Mat& Ht, Size& rect){
	int H_width = rect.width;
	int H_height = rect.height;

	int Ht_width = Ht.cols;
	int Ht_height = Ht.rows;
	if (H_width < Ht_width || H_height < Ht_height)
		return Mat();
	else if (H_width == Ht_width && H_height == Ht_height)
		return Ht;
	else {
		Mat H = Mat::eye(rect.height, rect.width, CV_64FC1);
		int start_i = H_height - Ht_height;
		int start_j = H_width - Ht_width;

		for (int i = start_i; i < H_height; i++){
			for (int j = start_j; j < H_width; j++){
				H.at<double>(i, j) = Ht.at<double>(i - start_i, j - start_j);
			}
		}
		return H;
	}
}

// ����Householder�任��������A�ֽ�ΪA = QR������QΪ��������RΪ��������
// ����˵��
// A����Ҫ����QR�ֽ�ķ���
// Q���ֽ�õ�����������
// R���ֽ�õ�����������
void QR(Mat A, Mat& Q, Mat& R){
	int n = A.cols;
	R = A.clone();
	Q = Mat::eye(n, n, CV_64FC1);
	for (int i = 0; i < n - 1; i++){
		Mat x = Mat::zeros(n - i, 1, CV_64FC1);
		Mat y = Mat::zeros(n - i, 1, CV_64FC1);
		for (int j = 0; j < n - i; j++){
			x.ptr<double>(0)[j] = R.at<double>(j + i, i);
		}
		//	cout << "x = " << x << endl;
		y.at<double>(0, 0) = 1.0;
		Mat Ht = householder(x, y);
		//	cout << "Ht = " << Ht << endl;
		Mat H;
		if (i>0)
			H = blkdiag(Mat::eye(i, i, CV_64FC1), Ht);
		else
			H = Ht.clone();
		//	cout << "H = " << H << endl;
		Q = Q*H;
		R = H*R;
	}
}


// QR�ֽ� R�����Խ��߶�����ֵ ��ʱ�õ��Ľ����MATLAB�в�һ��
bool QR_householder(Mat A, Mat &Q, Mat& R){
	//cout << "A = " << A << endl;

	A.copyTo(Q);
	vector<Mat> beta;
	Mat temp = A.col(0).clone();
	beta.push_back(temp);

	for (int i = 1; i < A.cols; i++){
		Mat temp = A.col(i).clone();
		for (int j = i - 1; j >= 0; j--){
			temp -= beta[j].dot(A.col(i)) / (beta[j].dot(beta[j])) * beta[j];
		}
		beta.push_back(temp);
	}

	for (size_t i = 0; i < beta.size(); i++){
		normalize(beta[i], beta[i]);
		beta[(int)i].copyTo(Q.col((int)i));
	}

	R = Q.inv()*A;
	return true;
}

// ������ �ֽ⡿�Գ����������Cholesky�ֽ�
// ��һ���Գ������ľ����ʾ��һ�������Ǿ���L����ת�õĳ˻��ķֽ� A = L*L.t()
double* cholesky(double* A, int n){
	double *L = initVector(n*n);

	L[0] = sqrt(A[0]);
	for (int i = 1; i < n; i++){
		L[i*n] = A[i*n] / L[0];
	}

	for (int k = 1; k < n; k++){
		double sum = 0;
		for (int j = 0; j < k; j++){
			sum += L[k*n + j] * L[k*n + j];
			L[k*n + k] = sqrt(A[k*n + k] - sum);
			for (int i = k + 1; i < n; i++){
				double sum2 = 0;
				for (int q = 0; q < k; q++){
					sum2 += L[i*n + q] * L[k*n + q];
					L[i*n + k] = (A[i*n + k] - sum2) / L[k*n + k];
				}
			}
		}
	}
	return L;
}

// ������ �ֽ⡿�Գ����������Cholesky�ֽ�
// ��һ���Գ������ľ����ʾ��һ�������Ǿ���L����ת�õĳ˻��ķֽ� A = L*L.t()
Mat cholesky(Mat A){
	CV_Assert(isPositiveDefinite(A) == POSITIVE_DEFINITE);
	CV_Assert(isSymmetry(A));
	double *data = A.ptr<double>(0);
	double *result = cholesky(data, A.rows);
	Mat L(A.size(), CV_64FC1, result);
	return L;
}

// ������ �ֽ⡿�Գ����������Cholesky�ֽ�
// ��һ���Գ������ľ����ʾ��һ�������Ǿ���L����ת�õĳ˻��ķֽ� A = C.t()*L
Mat cholesky2(Mat A){
	CV_Assert(isPositiveDefinite(A) == POSITIVE_DEFINITE);
	CV_Assert(isSymmetry(A));
	Mat L, U;
	decompDoolittle(A, L, U);

	Mat C = Mat::diag(U.diag()).clone();// * L.t();
	pow(C, 0.5, C);

	C = C*L.t();

	return C;
}

// �����󡿽�����Լ��Ϊ��ɭ�������
double* hessenberg(double* a, int n){
	double*A = copyMat(a, n, n);
	int i, u, v;

	double t;
	for (int k = 1; k <= n - 2; k++){
		double d = 0.0;
		for (int j = k; j <= n - 1; j++){
			u = j*n + k - 1;
			t = A[u];
			if (fabs(t) > fabs(d)){
				d = t;
				i = j;
			}
		}
		if (fabs(d) + 1.0 != 1.0){
			if (i != k){
				for (int j = k - 1; j <= n - 1; j++){
					u = i*n + j;
					v = k*n + j;
					t = A[u];
					A[u] = A[v];
					A[v] = t;
				}
				for (int j = 0; j <= n - 1; j++){
					u = j*n + i;
					v = j*n + k;
					t = A[u];
					A[u] = A[v];
					A[v] = t;
				}
			}
			for (int i = k + 1; i <= n - 1; i++){
				u = i*n + k - 1;
				t = A[u] / d;
				A[u] = 0.0;
				for (int j = k; j <= n - 1; j++){
					v = i*n + j;
					A[v] = A[v] - t*A[k*n + j];
				}
				for (int j = 0; j <= n - 1; j++){
					v = j*n + k;
					A[v] = A[v] + t*A[j*n + i];
				}
			}
		}
	}
	return A;
}

// ������ �����桿
double* InvMat(double* input, int row, int col){
	Mat src(row, col, CV_64FC1, input);
	Mat P, delta, Q;
	svdDecomp(src, P, delta, Q);

	/*cout << P << endl;
	cout << delta << endl;
	cout << Q << endl;*/

	Mat tmp = Q*delta.inv()*P.t();
	double *result = initVector(row*col);
	memcpy(result, tmp.ptr<double>(0), row*col*sizeof(double));
	return result;
}

// ������ �����桿 α�� �Ӻ���
Mat InvMat(Mat src){
	Mat P, delta, Q;
	svdDecomp(src, P, delta, Q);

	/*cout << P << endl;
	cout << delta << endl;
	cout << Q << endl;*/

	return (Q*delta.inv()*P.t());
}

/************************************************************************/
/* input:
/* a:���m*nʵ����A,����ʱ�����������
/* m:���� n������
/* u:���m*m����������, v:���n*n����������
/* eps:��������Ҫ��,  ka: max(m,n)+1
/* output:
/* ����ֵ���Ϊ��������ʾ������60�Σ���δ�������ֵ������ֵΪ�Ǹ���������
/************************************************************************/
int svdDecomp(double *a, int m, int n, double *u, double *v, int ka, double eps)
{
	int i, j, k, l, it, ll, kk, ix, iy, mm, nn, iz, m1, ks;
	double d, dd, t, sm, sm1, em1, sk, ek, b, c, shh, fg[2], cs[2];
	double *s, *e, *w;

	s = (double *)malloc(ka*sizeof(double));
	e = (double *)malloc(ka*sizeof(double));
	w = (double *)malloc(ka*sizeof(double));
	it = 60;
	k = n;

	if (m - 1 < n)
		k = m - 1;

	l = m;

	if (n - 2 < m)
		l = n - 2;

	if (l<0)
		l = 0;
	ll = k;

	if (l>k)
		ll = l;

	if (ll >= 1)
	{
		for (kk = 1; kk <= ll; kk++)
		{
			if (kk <= k)
			{
				d = 0.0;
				for (i = kk; i <= m; i++)
				{
					ix = (i - 1)*n + kk - 1; d = d + a[ix] * a[ix];
				}
				s[kk - 1] = sqrt(d);
				if (s[kk - 1] != 0.0)
				{
					ix = (kk - 1)*n + kk - 1;
					if (a[ix] != 0.0)
					{
						s[kk - 1] = fabs(s[kk - 1]);
						if (a[ix] < 0.0)
							s[kk - 1] = -s[kk - 1];
					}
					for (i = kk; i <= m; i++)
					{
						iy = (i - 1)*n + kk - 1;
						a[iy] = a[iy] / s[kk - 1];
					}
					a[ix] = 1.0 + a[ix];
				}
				s[kk - 1] = -s[kk - 1];
			}
			if (n >= kk + 1)
			{
				for (j = kk + 1; j <= n; j++)
				{
					if ((kk <= k) && (s[kk - 1] != 0.0))
					{
						d = 0.0;
						for (i = kk; i <= m; i++)
						{
							ix = (i - 1)*n + kk - 1;
							iy = (i - 1)*n + j - 1;
							d = d + a[ix] * a[iy];
						}
						d = -d / a[(kk - 1)*n + kk - 1];
						for (i = kk; i <= m; i++)
						{
							ix = (i - 1)*n + j - 1;
							iy = (i - 1)*n + kk - 1;
							a[ix] = a[ix] + d*a[iy];
						}
					}
					e[j - 1] = a[(kk - 1)*n + j - 1];
				}
			}
			if (kk <= k)
			{
				for (i = kk; i <= m; i++)
				{
					ix = (i - 1)*m + kk - 1; iy = (i - 1)*n + kk - 1;
					u[ix] = a[iy];
				}
			}
			if (kk <= l)
			{
				d = 0.0;
				for (i = kk + 1; i <= n; i++)
					d = d + e[i - 1] * e[i - 1];
				e[kk - 1] = sqrt(d);
				if (e[kk - 1] != 0.0)
				{
					if (e[kk] != 0.0)
					{
						e[kk - 1] = fabs(e[kk - 1]);
						if (e[kk] < 0.0)
							e[kk - 1] = -e[kk - 1];
					}
					for (i = kk + 1; i <= n; i++)
						e[i - 1] = e[i - 1] / e[kk - 1];
					e[kk] = 1.0 + e[kk];
				}
				e[kk - 1] = -e[kk - 1];
				if ((kk + 1 <= m) && (e[kk - 1] != 0.0))
				{
					for (i = kk + 1; i <= m; i++) w[i - 1] = 0.0;
					for (j = kk + 1; j <= n; j++)
					for (i = kk + 1; i <= m; i++)
						w[i - 1] = w[i - 1] + e[j - 1] * a[(i - 1)*n + j - 1];
					for (j = kk + 1; j <= n; j++)
					for (i = kk + 1; i <= m; i++)
					{
						ix = (i - 1)*n + j - 1;
						a[ix] = a[ix] - w[i - 1] * e[j - 1] / e[kk];
					}
				}
				for (i = kk + 1; i <= n; i++)
					v[(i - 1)*n + kk - 1] = e[i - 1];
			}
		}
	}
	mm = n;
	if (m + 1 < n)
		mm = m + 1;
	if (k < n)
		s[k] = a[k*n + k];
	if (m < mm)
		s[mm - 1] = 0.0;
	if (l + 1 < mm)
		e[l] = a[l*n + mm - 1];
	e[mm - 1] = 0.0;
	nn = m;
	if (m > n)
		nn = n;
	if (nn >= k + 1)
	{
		for (j = k + 1; j <= nn; j++)
		{
			for (i = 1; i <= m; i++)
				u[(i - 1)*m + j - 1] = 0.0;
			u[(j - 1)*m + j - 1] = 1.0;
		}
	}
	if (k >= 1)
	{
		for (ll = 1; ll <= k; ll++)
		{
			kk = k - ll + 1; iz = (kk - 1)*m + kk - 1;
			if (s[kk - 1] != 0.0)
			{
				if (nn >= kk + 1)
				for (j = kk + 1; j <= nn; j++)
				{
					d = 0.0;
					for (i = kk; i <= m; i++)
					{
						ix = (i - 1)*m + kk - 1;
						iy = (i - 1)*m + j - 1;
						d = d + u[ix] * u[iy] / u[iz];
					}
					d = -d;
					for (i = kk; i <= m; i++)
					{
						ix = (i - 1)*m + j - 1;
						iy = (i - 1)*m + kk - 1;
						u[ix] = u[ix] + d*u[iy];
					}
				}
				for (i = kk; i <= m; i++)
				{
					ix = (i - 1)*m + kk - 1;
					u[ix] = -u[ix];
				}
				u[iz] = 1.0 + u[iz];
				if (kk - 1 >= 1)
				for (i = 1; i <= kk - 1; i++)
					u[(i - 1)*m + kk - 1] = 0.0;
			}
			else
			{
				for (i = 1; i <= m; i++)
					u[(i - 1)*m + kk - 1] = 0.0;
				u[(kk - 1)*m + kk - 1] = 1.0;
			}
		}
	}
	for (ll = 1; ll <= n; ll++)
	{
		kk = n - ll + 1; iz = kk*n + kk - 1;
		if ((kk <= l) && (e[kk - 1] != 0.0))
		{
			for (j = kk + 1; j <= n; j++)
			{
				d = 0.0;
				for (i = kk + 1; i <= n; i++)
				{
					ix = (i - 1)*n + kk - 1; iy = (i - 1)*n + j - 1;
					d = d + v[ix] * v[iy] / v[iz];
				}
				d = -d;
				for (i = kk + 1; i <= n; i++)
				{
					ix = (i - 1)*n + j - 1;
					iy = (i - 1)*n + kk - 1;
					v[ix] = v[ix] + d*v[iy];
				}
			}
		}
		for (i = 1; i <= n; i++)
			v[(i - 1)*n + kk - 1] = 0.0;
		v[iz - n] = 1.0;
	}
	for (i = 1; i <= m; i++)
	for (j = 1; j <= n; j++)
		a[(i - 1)*n + j - 1] = 0.0;
	m1 = mm;
	it = 60;

	while (1 == 1)
	{
		if (mm == 0)
		{
			ppp(a, e, s, v, m, n);
			free(s); free(e); free(w);
			return(1);
		}
		if (it == 0)
		{
			ppp(a, e, s, v, m, n);
			free(s); free(e); free(w);
			return(-1);
		}
		kk = mm - 1;
		while ((kk != 0) && (fabs(e[kk - 1]) != 0.0))
		{
			d = fabs(s[kk - 1]) + fabs(s[kk]);
			dd = fabs(e[kk - 1]);
			if (dd > eps*d)
				kk = kk - 1;
			else
				e[kk - 1] = 0.0;
		}
		if (kk == mm - 1)
		{
			kk = kk + 1;
			if (s[kk - 1] < 0.0)
			{
				s[kk - 1] = -s[kk - 1];
				for (i = 1; i <= n; i++)
				{
					ix = (i - 1)*n + kk - 1; v[ix] = -v[ix];
				}
			}
			while ((kk != m1) && (s[kk - 1] < s[kk]))
			{
				d = s[kk - 1]; s[kk - 1] = s[kk]; s[kk] = d;
				if (kk < n)
				for (i = 1; i <= n; i++)
				{
					ix = (i - 1)*n + kk - 1; iy = (i - 1)*n + kk;
					d = v[ix]; v[ix] = v[iy]; v[iy] = d;
				}
				if (kk < m)
				for (i = 1; i <= m; i++)
				{
					ix = (i - 1)*m + kk - 1; iy = (i - 1)*m + kk;
					d = u[ix]; u[ix] = u[iy]; u[iy] = d;
				}
				kk = kk + 1;
			}
			it = 60;
			mm = mm - 1;
		}
		else
		{
			ks = mm;
			while ((ks > kk) && (fabs(s[ks - 1]) != 0.0))
			{
				d = 0.0;
				if (ks != mm)
					d = d + fabs(e[ks - 1]);
				if (ks != kk + 1)
					d = d + fabs(e[ks - 2]);
				dd = fabs(s[ks - 1]);
				if (dd > eps*d)
					ks = ks - 1;
				else
					s[ks - 1] = 0.0;
			}
			if (ks == kk)
			{
				kk = kk + 1;
				d = fabs(s[mm - 1]);
				t = fabs(s[mm - 2]);
				if (t > d)
					d = t;
				t = fabs(e[mm - 2]);
				if (t > d)
					d = t;
				t = fabs(s[kk - 1]);
				if (t > d)
					d = t;
				t = fabs(e[kk - 1]);
				if (t > d)
					d = t;
				sm = s[mm - 1] / d;
				sm1 = s[mm - 2] / d;
				em1 = e[mm - 2] / d;
				sk = s[kk - 1] / d;
				ek = e[kk - 1] / d;
				b = ((sm1 + sm)*(sm1 - sm) + em1*em1) / 2.0;
				c = sm*em1; c = c*c; shh = 0.0;
				if ((b != 0.0) || (c != 0.0))
				{
					shh = sqrt(b*b + c);
					if (b < 0.0)
						shh = -shh;
					shh = c / (b + shh);
				}
				fg[0] = (sk + sm)*(sk - sm) - shh;
				fg[1] = sk*ek;
				for (i = kk; i <= mm - 1; i++)
				{
					sss(fg, cs);
					if (i != kk)
						e[i - 2] = fg[0];
					fg[0] = cs[0] * s[i - 1] + cs[1] * e[i - 1];
					e[i - 1] = cs[0] * e[i - 1] - cs[1] * s[i - 1];
					fg[1] = cs[1] * s[i];
					s[i] = cs[0] * s[i];
					if ((cs[0] != 1.0) || (cs[1] != 0.0))
					for (j = 1; j <= n; j++)
					{
						ix = (j - 1)*n + i - 1;
						iy = (j - 1)*n + i;
						d = cs[0] * v[ix] + cs[1] * v[iy];
						v[iy] = -cs[1] * v[ix] + cs[0] * v[iy];
						v[ix] = d;
					}
					sss(fg, cs);
					s[i - 1] = fg[0];
					fg[0] = cs[0] * e[i - 1] + cs[1] * s[i];
					s[i] = -cs[1] * e[i - 1] + cs[0] * s[i];
					fg[1] = cs[1] * e[i];
					e[i] = cs[0] * e[i];
					if (i < m)
					if ((cs[0] != 1.0) || (cs[1] != 0.0))
					for (j = 1; j <= m; j++)
					{
						ix = (j - 1)*m + i - 1;
						iy = (j - 1)*m + i;
						d = cs[0] * u[ix] + cs[1] * u[iy];
						u[iy] = -cs[1] * u[ix] + cs[0] * u[iy];
						u[ix] = d;
					}
				}
				e[mm - 2] = fg[0];
				it = it - 1;
			}
			else
			{
				if (ks == mm)
				{
					kk = kk + 1;
					fg[1] = e[mm - 2]; e[mm - 2] = 0.0;
					for (ll = kk; ll <= mm - 1; ll++)
					{
						i = mm + kk - ll - 1;
						fg[0] = s[i - 1];
						sss(fg, cs);
						s[i - 1] = fg[0];
						if (i != kk)
						{
							fg[1] = -cs[1] * e[i - 2];
							e[i - 2] = cs[0] * e[i - 2];
						}
						if ((cs[0] != 1.0) || (cs[1] != 0.0))
						for (j = 1; j <= n; j++)
						{
							ix = (j - 1)*n + i - 1;
							iy = (j - 1)*n + mm - 1;
							d = cs[0] * v[ix] + cs[1] * v[iy];
							v[iy] = -cs[1] * v[ix] + cs[0] * v[iy];
							v[ix] = d;
						}
					}
				}
				else
				{
					kk = ks + 1;
					fg[1] = e[kk - 2];
					e[kk - 2] = 0.0;
					for (i = kk; i <= mm; i++)
					{
						fg[0] = s[i - 1];
						sss(fg, cs);
						s[i - 1] = fg[0];
						fg[1] = -cs[1] * e[i - 1];
						e[i - 1] = cs[0] * e[i - 1];
						if ((cs[0] != 1.0) || (cs[1] != 0.0))
						for (j = 1; j <= m; j++)
						{
							ix = (j - 1)*m + i - 1;
							iy = (j - 1)*m + kk - 2;
							d = cs[0] * u[ix] + cs[1] * u[iy];
							u[iy] = -cs[1] * u[ix] + cs[0] * u[iy];
							u[ix] = d;
						}
					}
				}
			}
		}
	}
	return(1);
}


// ����ֵ�ֽ� SVD ��������
static void ppp(double a[], double e[], double s[], double v[], int m, int n)
{
	int i, j, p, q;
	double d;
	if (m >= n) i = n;
	else i = m;
	for (j = 1; j <= i - 1; j++)
	{
		a[(j - 1)*n + j - 1] = s[j - 1];
		a[(j - 1)*n + j] = e[j - 1];
	}
	a[(i - 1)*n + i - 1] = s[i - 1];
	if (m < n)
		a[(i - 1)*n + i] = e[i - 1];
	for (i = 1; i <= n - 1; i++)
	for (j = i + 1; j <= n; j++)
	{
		p = (i - 1)*n + j - 1; q = (j - 1)*n + i - 1;
		d = v[p]; v[p] = v[q]; v[q] = d;
	}
	return;
}

// ����ֵ�ֽ� SVD  ��������
static void sss(double fg[], double cs[])
{
	double r, d;
	if ((fabs(fg[0]) + fabs(fg[1])) == 0.0)
	{
		cs[0] = 1.0; cs[1] = 0.0; d = 0.0;
	}
	else
	{
		d = sqrt(fg[0] * fg[0] + fg[1] * fg[1]);
		if (fabs(fg[0]) > fabs(fg[1]))
		{
			d = fabs(d);
			if (fg[0] < 0.0) d = -d;
		}
		if (fabs(fg[1]) >= fabs(fg[0]))
		{
			d = fabs(d);
			if (fg[1]<0.0) d = -d;
		}
		cs[0] = fg[0] / d; cs[1] = fg[1] / d;
	}
	r = 1.0;
	if (fabs(fg[0])>fabs(fg[1]))
		r = cs[1];
	else
	if (cs[0] != 0.0) r = 1.0 / cs[0];
	fg[0] = d;
	fg[1] = r;
	return;
}

// ������ֽ⡿SVD�ֽ� A = U*W*V.t(); U��m*m���Ͼ��� V��n*n���Ͼ��� W��m*n�ľ���
void svdDecomp(double* input, int m, int n, double *P, double* delta, double *Q){
	Mat src(m, n, CV_64FC1, input);

	Mat AHA = src.t()*src;
	Mat _vec(n, n, CV_64F);
	Mat _eig(n, 1, CV_64F);

	cv::eigen(AHA, _eig, _vec);

	int r = rankM(input, m, n);

	// A = P * �� * Q.t()
	// A  m*n
	// P  m*r
	// ��  r*r
	// Q  n*r
	Mat _Q(n, r, CV_64FC1);
	Mat _P(m, r, CV_64FC1);
	Mat _delta = Mat::zeros(r, r, CV_64FC1);


	// �μ��������ۿαʼ�P61
	for (int i = 0; i < r; i++){
		Mat tmp = _vec.row(i).t();
		Mat tmp2 = src*tmp;
		tmp = tmp / norm(tmp);
		tmp.copyTo(_Q.col(i));

		tmp2 = tmp2 / norm(tmp2);
		tmp2.copyTo(_P.col(i));

		_delta.at<double>(i, i) = sqrt(_eig.at<double>(i, 0));
	}

	memcpy(P, _P.ptr<double>(0), m*r*sizeof(double));
	memcpy(delta, _delta.ptr<double>(0), r*r*sizeof(double));
	memcpy(Q, _Q.ptr<double>(0), n*r*sizeof(double));
}
// ������ֽ⡿SVD�ֽ� A = P*delta*Q.t() P��m*r�ľ��� delta��r*r�ľ��������Խ���Ԫ��ΪA������ֵ�� QΪn*r�ľ���
void svdDecomp(Mat src, Mat& P, Mat& delta, Mat& Q){
	Mat AHA = src.t()*src;
	Mat _vec(src.cols, src.cols, CV_64F);
	Mat _eig;

	cv::eigen(AHA, _eig, _vec);

	int r = rankM(src);

	// A = P * �� * Q.t()
	// A  m*n
	// P  m*r
	// ��  r*r
	// Q  n*r
	Q.create(src.cols, r, CV_64FC1);
	P.create(src.rows, r, CV_64FC1);
	delta = Mat::zeros(r, r, CV_64FC1);

	// �μ��������ۿαʼ�P61
	for (int i = 0; i < r; i++){
		Mat tmp = _vec.row(i).t();
		Mat tmp2 = src*tmp;
		tmp = tmp / norm(tmp);
		tmp.copyTo(Q.col(i));

		tmp2 = tmp2 / norm(tmp2);
		tmp2.copyTo(P.col(i));

		delta.at<double>(i, i) = sqrt(_eig.at<double>(i, 0));
	}
}


// �����������������������ȫ1��
void AddOnesRow(Mat& input){
	Mat result(input.rows + 1, input.cols, input.type());
	input.copyTo(result(Rect(0, 0, input.cols, input.rows)));
	result.row(result.rows - 1).setTo(Scalar::all(1));
	result.copyTo(input);
}

// �����󡿵�� ��������ĵ����Դ����mat1��mat2��ά������һ�£���������result��
bool dotarray(Mat mat1, Mat mat2, Mat& result){
	if (mat1.rows != mat2.rows || mat1.cols != mat2.cols)
		return false;

	Mat temp = mat1.mul(mat2);
	result = Mat::zeros(1, mat1.cols, CV_64FC1);

	for (int i = 0; i < mat1.cols; i++){
		double dResult = 0;
		for (int j = 0; j < mat1.rows; j++){
			dResult = dResult + *(temp.ptr<double>(0) + i + j*mat1.cols);
		}
		*(result.ptr<double>(0) + i) = dResult;
	}
	return true;
}

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
Mat repmat(Mat src, int rows, int cols){
	Mat dst = Mat::zeros(src.rows*rows, src.cols*cols, src.type());
	int r = src.rows;
	int c = src.cols;
	for (int i = 0; i < rows;i++){
		for (int j = 0; j < cols; j++){
				src.copyTo(dst(Rect(c*j, r*i, c, r)));
		}
	}
	return dst;
}

// ��ʵ�Գ����ȫ������ֵ����������
int cjcbi(double*a, int n, double*v, double eps, int jt){
	int i, j, p, q, u, w, t, s, l;
	double fm, cn, sn, omega, x, y, d;
	l = 1;
	for (i = 0; i <= n - 1; i++)
	{
		v[i*n + i] = 1.0;
		for (j = 0; j <= n - 1; j++)
		if (i != j) v[i*n + j] = 0.0;
	}
	while (1 == 1)
	{
		fm = 0.0;
		for (i = 1; i <= n - 1; i++)
		for (j = 0; j <= i - 1; j++)
		{
			d = fabs(a[i*n + j]);
			if ((i != j) && (d > fm))
			{
				fm = d; p = i; q = j;
			}
		}
		if (fm<eps)  return(1);
		if (l>jt)  return(-1);
		l = l + 1;
		u = p*n + q; w = p*n + p; t = q*n + p; s = q*n + q;
		x = -a[u]; y = (a[s] - a[w]) / 2.0;
		omega = x / sqrt(x*x + y*y);
		if (y < 0.0) omega = -omega;
		sn = 1.0 + sqrt(1.0 - omega*omega);
		sn = omega / sqrt(2.0*sn);
		cn = sqrt(1.0 - sn*sn);
		fm = a[w];
		a[w] = fm*cn*cn + a[s] * sn*sn + a[u] * omega;
		a[s] = fm*sn*sn + a[s] * cn*cn - a[u] * omega;
		a[u] = 0.0; a[t] = 0.0;
		for (j = 0; j <= n - 1; j++)
		if ((j != p) && (j != q))
		{
			u = p*n + j; w = q*n + j;
			fm = a[u];
			a[u] = fm*cn + a[w] * sn;
			a[w] = -fm*sn + a[w] * cn;
		}
		for (i = 0; i <= n - 1; i++)
		if ((i != p) && (i != q))
		{
			u = i*n + p; w = i*n + q;
			fm = a[u];
			a[u] = fm*cn + a[w] * sn;
			a[w] = -fm*sn + a[w] * cn;
		}
		for (i = 0; i <= n - 1; i++)
		{
			u = i*n + p; w = i*n + q;
			fm = v[u];
			v[u] = fm*cn + v[w] * sn;
			v[w] = -fm*sn + v[w] * cn;
		}
	}
	return(1);
}

// ������������Сֵ
void maxAndmin(double &bmax, double &bmin, double* data, int col, int row){
	double max = data[0];
	double min = data[0];
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			double kk = data[i*col + j];
			if (kk>max){
				max = kk;
			}
			if (kk < min){
				min = kk;
			}
		}
	}
	bmax = max;
	bmin = min;
}

// �ж��Ƿ��ǶԳƾ���
bool isSymmetry(Mat input){
	CV_Assert(input.rows == input.cols);

	Mat T = abs(input - input.t());
	double result = fabs(sum(T)[0]);
	if (result < 1e-7)
		return true;
	else
		return false;
}

// �ж��Ƿ�����������
int isPositiveDefinite(Mat input){
	Mat eigens, vectors;
	cv::eigen(input, eigens, vectors);

	//cout << eigens << endl;

	// �Ƿ��ǰ�����
	double* p = eigens.ptr<double>(0);

	// �Ƿ������
	for (int i = 0; i < eigens.rows*eigens.cols; i++){
		if (p[i] < 0.0 - 1e-12)
			return NEGITIVE_DEFINITE;
	}


	for (int i = 0; i < eigens.rows*eigens.cols; i++){
		if (fabs(p[i]) < 1e-12)
			return POSITIVE_SEMIDEFINITE;
	}

	return POSITIVE_DEFINITE;
}

// ����������cond
double cond(Mat input){
	CV_Assert(isSymmetry(input));
	Mat eigens, vectors;
	cv::eigen(input, eigens, vectors);

	eigens = abs(eigens);
	cout << eigens << endl;
	double maxEg, minEg;
	minMaxIdx(eigens, &minEg, &maxEg);

	if (minEg < 1e-12)
		return RAND_MAX;

	return (maxEg / minEg);
}

// Doolittle �ֽ� LU �ֽ�
bool decompDoolittle(Mat input, Mat& L, Mat& U){
	CV_Assert(input.rows == input.cols);
	CV_Assert(fabs(determinant(input)) > 1e-12);

	L = Mat::zeros(input.size(), CV_64FC1);
	U = Mat::zeros(input.size(), CV_64FC1);

	int n = input.rows;

	for (int k = 0; k <= n - 1; k++){
		for (int j = k; j <= n - 1; j++){
			double tmp = 0.0;
			for (int t = 0; t <= k - 1; t++){
				tmp += L.at<double>(k, t)*U.at<double>(t, j);
			}
			U.at<double>(k, j) = input.at<double>(k, j) - tmp;
		}
		for (int i = k + 1; i <= n - 1; i++){
			if (k < n - 1){
				double tmp = 0.0;
				for (int t = 0; t <= k - 1; t++){
					tmp += L.at<double>(i, t)*U.at<double>(t, k);
				}
				L.at<double>(i, k) = input.at<double>(i, k) - tmp;
				L.at<double>(i, k) /= U.at<double>(k, k);
			}
		}
	}

	L.diag().setTo(1.0);

	//cout << L << endl;
	//cout << U << endl;

	return true;
}

// Crout �ֽ� LU �ֽ�
bool decompCrout(Mat input, Mat& L_, Mat& U_){
	CV_Assert(input.rows == input.cols);
	CV_Assert(fabs(determinant(input)) > 1e-12);

	L_ = Mat::zeros(input.size(), CV_64FC1);
	U_ = Mat::zeros(input.size(), CV_64FC1);

	int n = input.rows;

	for (int k = 0; k <= n - 1; k++){
		for (int i = k; i <= n - 1; i++){
			double tmp = 0.0;
			for (int t = 0; t <= k - 1; t++){
				tmp += L_.at<double>(i, t)*U_.at<double>(t, k);
			}
			L_.at<double>(i, k) = input.at<double>(i, k) - tmp;
		}
		for (int j = k + 1; j <= n - 1; j++){
			if (k < n - 1){
				double tmp = 0.0;
				for (int t = 0; t <= k - 1; t++){
					tmp += L_.at<double>(k, t)*U_.at<double>(t, j);
				}
				U_.at<double>(k, j) = input.at<double>(k, j) - tmp;
				U_.at<double>(k, j) /= L_.at<double>(k, k);
			}
		}
	}

	U_.diag().setTo(1.0);

	cout << L_ << endl;
	cout << U_ << endl;
	return true;
}

// ���������жϾ����Ƿ�Ϊ������
bool isSingular(Mat A){
	CV_Assert(isSquare(A));
	if (fabs(determinant(A)) < 1e-12)
		return true;
	else
		return false;
}

// �����ȡ��жϾ����Ƿ�����
bool isFullRank(Mat A){
	CV_Assert(isSquare(A));

	if (rankM(A) == A.cols)
		return true;
	else
		return false;
}

// �������жϾ����Ƿ�Ϊ����
bool isSquare(Mat A){
	return (A.rows == A.cols);
}

// ��������������Ԫ�ؾ���ֵ���ֵ
double infiniteNorm(Mat A){
	return norm(A, NORM_INF);
}

// ��1������������1���� 1-����
double norm_L1(cv::Mat A){
	return norm(A, NORM_L1);
}


// ���Խ���Ԫ�����ֵ��
double maxValueInDiag(Mat input){
	Mat d = input.diag();
	double minV, maxV;
	minMaxLoc(d, &minV, &maxV);
	return maxV;
}

// svd�ֽ� ��MATLAB����һ�� �����V��MATLAB�е�v��ת�ù�ϵ
bool svd(Mat A, Mat& U, Mat &S, Mat &V, bool useDouble/* = true*/){
	Mat tmp;
	cv::SVDecomp(A, tmp, U, V, SVD::FULL_UV);

	//SVD::compute(A,tmp,U,V,SVD::FULL_UV);
	//cout << "U = " << U << endl;
	////cout << "S = " << S << endl;
	//cout << "V = " << V << endl;
	if (useDouble)
		S = Mat::zeros(min(A.rows, A.cols), 1, CV_64FC1);
	else
		S = Mat::zeros(min(A.rows, A.cols), 1, CV_32FC1);

	for (int i = 0; i < tmp.rows*tmp.cols; i++){
		if (useDouble)
			S.at<double>(i, i) = tmp.ptr<double>(0)[i];
		else
			S.at<float>(i, i) = tmp.ptr<float>(0)[i];
	}
	//cout << "S = " << S << endl;

	//cout << "-- svd ok!" << endl;
	return true;
}

// ���ά����Ҷ�任 double�� ����Ϊ��ͨ�� ʵ��; ����ֵΪ����
Mat fft(Mat input, Mat& mag, Mat& angle){
	if (input.cols == 1){
		int m = getOptimalDFTSize(input.rows);
		int n = 1;

		// Padding 0, result is @ padded
		Mat padded;
		copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));

		// Create planes to storage REA	L part and IMAG part, IMAGE part init are 0
		Mat planes[] = { Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F) };

		Mat complexI;
		merge(planes, 2, complexI);

		dft(complexI, complexI);

		/*for (int i = 0; i < complexI.rows*complexI.cols; i++){
			cout << complexI.ptr<double>(0)[i * 2] << " " << complexI.ptr<double>(0)[i * 2 + 1] << endl;
			}*/

		// compute the magnitude and switch to logarithmic scale 
		split(complexI, planes);

		// ���ֵ����λ��Ϣ
		magnitude(planes[0], planes[1], mag);
		phase(planes[0], planes[1], angle);

		return complexI;
	}
	else if (input.cols > 1){						// ÿ�н��и���Ҷ�任
		Mat c(input.size(), CV_64FC2);
		for (int i = 0; i < input.cols; i++){
			Mat m, a;
			Mat tmp = fft(input.col(i), m, a);
			tmp.copyTo(c.col(i));
		}
		return c;
	}
}

//// ������չ��1ά
//Mat Mat2Vec(Mat input){
//	Mat result = Mat::zeros(input.cols*input.rows, 1, CV_64FC1);
//	double * presult = result.ptr<double>(0);
//	double * pinput = input.ptr<double>(0);
//	for (int i = 0; i < input.rows*input.cols; i++){
//		*presult++ = *pinput++;
//	}
//	return result;
//}


// �ֿ���������һ�� ���� 
Mat combine2MatV(Mat Up, Mat Down){
	if (Up.size() == Size(0, 0))
		return Down;
	else if (Down.rows == 0 || Down.cols == 0)
		return Up;
	
	CV_Assert(Up.cols == Down.cols && Up.type() == Down.type());
	Mat result(Up.rows + Down.rows, Up.cols, Up.type());
	Up.copyTo(result(Rect(0, 0, Up.cols, Up.rows)));
	Down.copyTo(result(Rect(0, Up.rows, Down.cols, Down.rows)));
	return result;
}

// �ֿ���������һ�� ����
Mat combine2MatH(Mat Left, Mat Right){
	if (Left.size() == Size(0, 0))
		return Right;
	else if (Right.size() == Size(0, 0))
		return Left;

	CV_Assert(Left.rows == Right.rows && Left.type() == Right.type());
	Mat result(Left.rows, Left.cols + Right.cols, Left.type());
	Left.copyTo(result(Rect(0, 0, Left.cols, Left.rows)));
	Right.copyTo(result(Rect(Left.cols, 0, Right.cols, Right.rows)));
	return result;
}

// Hermiteת��
Mat hermiteT(Mat input){
	CV_Assert(input.channels() == 2);
	Mat result = input.t();
	for (int i = 0; i < result.rows*result.cols; i++){
		result.ptr<Point2d>(0)[i].y *= -1.0;
	}
	return result;
}

// �����˷� ������
Mat complexMulEle(Mat X, Mat Y){
	CV_Assert(X.channels() == 2 && Y.channels() == 2);
	CV_Assert(X.rows == Y.rows && X.cols == Y.cols);
	Mat result = X.clone();
	Point2d *pResult = result.ptr<Point2d>(0);
	Point2d *pDataX = X.ptr<Point2d>(0);
	Point2d *pDataY = Y.ptr<Point2d>(0);
	for (int i = 0; i < X.rows*X.cols; i++){
		(*pResult++) = complexMul(*pDataX++, *pDataY++);
	}
	return result;
}

// ������Ȼָ�� 
Mat complexExp(Mat X){
	CV_Assert(X.channels() == 2);
	Mat result = X.clone();
	Point2d *pResult = result.ptr<Point2d>(0);
	Point2d *pData = X.ptr<Point2d>(0);
	for (int i = 0; i < X.rows*X.cols; i++){
		double real = (*pData).x;
		double imag = (*pData++).y;
		(*pResult).x = exp(real)*cos(imag);
		(*pResult++).y = exp(real)*sin(imag);
	}
	return result;
}

// �����˷� X,Y ����˫ͨ����  ����ֵҲ��˫ͨ����
Mat complexMul(Mat X, Mat Y){
	CV_Assert(X.cols == Y.rows && X.channels() == 2 && Y.channels() == 2);
	Mat result;

	vector<Mat> a;
	vector<Mat> b;
	split(X, a);
	split(Y, b);

	vector<Mat> c(2);
	c[0] = a[0] * b[0] - a[1] * b[1];
	c[1] = a[0] * b[1] + a[1] * b[0];
	merge(c, result);

	return result;
}

// �����˷� ����ϵ�� Point2d X * Point2d Y
Point2d complexMul(Point2d X, Point2d Y){
	return Point2d(X.x*Y.x - X.y*Y.y, X.x*Y.y + X.y*Y.x);
}

// �����˷� ����ϵ�� Point2d X * Mat Y
Mat complexScale(Mat Y, Point2d X){
	CV_Assert(Y.channels() == 2);
	Mat result = Y.clone();
	Point2d *pResult = result.ptr<Point2d>(0);
	Point2d *pData = Y.ptr<Point2d>(0);
	for (int i = 0; i < Y.rows*Y.cols; i++){
		*pResult++ = complexMul(X, *pData++);
	}
	return result;
}

// Matȡʵ�� input˫ͨ��
Mat real(Mat input){
	CV_Assert(input.channels() == 2);
	vector<Mat> r;
	split(input, r);
	return r[0];
}

// Matȡ�鲿 input˫ͨ��
Mat imag(Mat input){
	CV_Assert(input.channels() == 2);
	vector<Mat> r;
	split(input, r);
	return r[1];
}

// ���������ֵ
Mat complexAbsMat(Mat input){
	CV_Assert(input.channels() == 2);
	vector<Mat> r;
	split(input, r);
	Mat result;
	magnitude(r[0], r[1], result);
	return result;
}

// ʵ�����鲿�ϲ�Ϊ2ͨ������
Mat merge(Mat real, Mat imag){
	vector<Mat> t;
	t.push_back(real);
	t.push_back(imag);
	Mat result;
	merge(t, result);
	return result;
}

// ʵ������ת�������� �൱�ڲ���һ��ȫ����
Mat real2complex(Mat input){
	CV_Assert(input.channels() == 1);
	Mat t = input.clone();
	Mat z = Mat::zeros(t.size(), input.type());
	return merge(t, z);
}

// ���ɷַ���
// ���룺 A		---	��������ÿ��Ϊһ������
//		 k		--- ��ά��kά
// ����� pcaA	--- ��ά���kά��������������ɵľ���ÿ��һ������������kΪ��ά�������������
//		 V		--- ���ɷַ���
void PCATrans(Mat input, int k, Mat& pcaA, Mat& V){
	input.convertTo(input, CV_64FC1);

	int r = input.rows;
	int c = input.cols;

	// ������ֵ ÿһ�еľ�ֵ
	Mat meanVec = input.row(0).clone();
	for (int i = 0; i < input.cols; i++){
		meanVec.ptr<double>(0)[i] = mean(input.col(i))[0];
	}
	//cout << meanVec << endl;

	// ����Э��������ת��covMatT
	Mat Z = (input - repmat(meanVec, r, 1));
	//cout << "Z = " << Z << endl;
	Mat covMatT = Z * Z.t();
	//cout << "covMatT = " << covMatT << endl;

	//cout << "ok" << endl;
	// ����covMatT��ǰk������ֵ�ͱ�������
	// �����󡿼��� ����ֵ����������
	Mat eigens;
	Mat vectors;
	cv::eigen(covMatT, eigens, vectors);

	//cout << "����ֵ: " << eigens << endl;
	//cout << "��������: " << vectors << endl;
	//cout << "nihao" << endl;
	vectors(Rect(0, 0, vectors.cols, k)).copyTo(V);
	Mat D = eigens(Rect(0, 0, eigens.cols, k)).clone();

	//cout << V << endl;

	// �õ�Э�������(covMatT)'�ı�������
	V = Z.t()*V.t();


	// ����������һ��Ϊ��λ��������
	for (int i = 0; i < V.cols; i++){
		double temp = norm(V.col(i));
		V.col(i) /= temp;
	}

	//cout << "V = " << V << endl;
	//cout << "V.size = " << V.size() << endl;

	// ���Ա任��ͶӰ����ά��kά
	pcaA = Z * V;
	//cout << "pcaA = " << pcaA << endl;
}

// չ�������� ����չ��
Mat convert2Vec(Mat input){
	Mat result = Mat::zeros(input.size().area(), 1, input.type());
	if (input.type() == CV_8UC1){
		memcpy(result.ptr<uchar>(0), input.ptr<char>(0), sizeof(uchar)*input.size().area());
	}
	else if (input.type() == CV_64FC1){
		memcpy(result.ptr<double>(0), input.ptr<double>(0), sizeof(double)*input.size().area());
	}
	return result;
}

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
Mat hadamard(int N){
	CV_Assert(is2ofIntPow(N) || (is2ofIntPow(N / 12) && N % 12 == 0) || (is2ofIntPow(N / 20) && N % 20 == 0));
	pair<double, int> fe;
	int k = 0, e = 0;
	fe = log2Double(N);
	if (fe.first == 0.5 && fe.second > 0){
		k = 0;
	}
	else {
		fe = log2Double(N / 12);
		if (N % 12 == 0 && fe.first == 0.5 && fe.second > 0){
			k = 1;
		}
		else{
			k = 2;
			fe = log2Double(N / 20);
		}
	}
	e = fe.second - 1;

	Mat H;
	Mat m1 = (Mat_<double>(1, 11) << -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1);
	Mat m2 = (Mat_<double>(1, 11) << -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1);

	Mat m3 = (Mat_<double>(1, 19) << -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1);
	Mat m4 = (Mat_<double>(1, 19) << 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1);
	if (k == 0)			// N = 1*2^e;
		H = (Mat_<double>(1, 1) << 1.0);
	else if (k == 1)	// N = 12*2^e;
	{
		H = combine2MatH(Mat::ones(11, 1, CV_64FC1), toeplitz(m1, m2));
		H = combine2MatV(Mat::ones(1, 12, CV_64FC1), H);
		//cout << "H = " << H*H.t() << endl;
	}
	else if (k == 2){	// N = 20*2^e
		H = combine2MatH(Mat::ones(19, 1, CV_64FC1), hankel(m3, m4));
		H = combine2MatV(Mat::ones(1, 20, CV_64FC1), H);
	}

	for (int i = 0; i < e; i++){
		Mat H1 = combine2MatH(H, H);
		Mat H2 = combine2MatH(H, -H);
		H = combine2MatV(H1, H2);
	}
	return H;
}

// toeplitz�����Ĺ����������������ȣ�toeplitz������
//	�������Ⱦ�����ص��ǣ�����һ�С���һ���⣬����ÿ��Ԫ�ض��������Ͻǵ�Ԫ����ͬ��
//	���ø�ʽ��
//	A = toeplitz(��1��Ԫ�����飬��1��Ԫ�����飩
// ע�⣺��1�еĵ�1��Ԫ��Ӧ���1�еĵ�1��Ԫ����ͬ�������1�еĵ�һ��Ԫ�ؽ��Զ���Ϊ1�еĵ�1��Ԫ�ء�
Mat toeplitz(Mat c, Mat r){
	CV_Assert((c.data != NULL) && (c.rows == 1 || c.cols == 1));
	CV_Assert((r.data != NULL) && (r.rows == 1 || r.cols == 1));
	int rows = c.rows*c.cols;
	int cols = r.rows*r.cols;
	cout << c << endl;

	Mat result = Mat::zeros(rows, cols, CV_64FC1);
	memcpy(result.row(0).ptr<double>(0), r.ptr<double>(0), sizeof(double)*cols);
	for (int i = 0; i < rows; i++){
		result.ptr<double>(i)[0] = c.ptr<double>(0)[i];
	}


	for (int i = 1; i < rows; i++){
		for (int j = 1; j < cols; j++){
			result.at<double>(i, j) = result.at<double>(i - 1, j - 1);
		}
	}
	return result;
}

// hankel�����Ĺ���������Hankel����
//	Hankel������ص��ǣ�����һ�С����һ���⣬����ÿ��Ԫ�ض��������½ǵ�Ԫ����ͬ��
//	���ø�ʽ��
//	A = hankel(��1��Ԫ������,���һ��Ԫ�����飩
// ע�⣺���һ�еĵ�1��Ԫ��Ӧ���1�еĵ�1��Ԫ����ͬ���������һ�еĵ�һ��Ԫ�ؽ��Զ���Ϊ1�еĵ�1��Ԫ�ء�
Mat hankel(Mat c, Mat r){
	CV_Assert((c.data != NULL) && (c.rows == 1 || c.cols == 1));
	CV_Assert((r.data != NULL) && (r.rows == 1 || r.cols == 1));
	int rows = c.rows*c.cols;
	int cols = r.rows*r.cols;
	cout << c << endl;

	Mat result = Mat::zeros(rows, cols, CV_64FC1);
	memcpy(result.row(rows - 1).ptr<double>(0), r.ptr<double>(0), sizeof(double)*cols);
	for (int i = 0; i < rows; i++){
		result.ptr<double>(i)[0] = c.ptr<double>(0)[i];
	}

	for (int i = rows - 2; i >= 0; i--){
		for (int j = 1; j < cols; j++){
			result.at<double>(i, j) = result.at<double>(i + 1, j - 1);
		}
	}
	return result;
}

// �ϲ������� Ĭ��Ϊ����ϲ� ����ƴ�� ���Һϲ� ����ƴ�� ���ºϲ�
Mat combine2Mat(Mat A, Mat B, bool CmbHor){
	CV_Assert(A.type() == B.type());
	if (CmbHor){
		Mat result = Mat::zeros(max(A.rows, B.rows), A.cols + B.cols, A.type());
		A.copyTo(result(Rect(0, 0, A.cols, A.rows)));
		B.copyTo(result(Rect(A.cols, 0, B.cols, B.rows)));
		return result;
	}
	else {
		Mat result = Mat::zeros(A.rows + B.rows, max(A.cols, B.cols), A.type());
		A.copyTo(result(Rect(0, 0, A.cols, A.rows)));
		B.copyTo(result(Rect(0, A.rows, B.cols, B.rows)));
		return result;
	}
}


// �������о���ֵС��val��Ԫ������ ���ص�����Ĥͼ�� ����Ԫ��Ϊ0 ��������Ϊ255
Mat set2zeroAbsBelowThresh(Mat& input, double val){
	CV_Assert(input.type() == CV_64FC1 && input.data != NULL);
	Mat mask = Mat::zeros(input.size(), CV_8UC1);
	mask.setTo(255);
	for (int i = 0; i < input.rows; i++){
		double * data = input.ptr<double>(i);
		for (int j = 0; j < input.cols; j++){
			if (fabs(*data) < val){
				*data = 0;
				mask.at<uchar>(i, j) = 0;
			}
			data++;
		}
	}
	return mask;
}

// �ж����������Ƿ���ͬ ����ͼ���Ƿ���ͬ
bool isTwoMatEqual(Mat A, Mat B){
	if (A.size() != B.size() || A.type() != B.type() || A.channels() != B.channels()){
		return false;
	}
	Scalar r = sum(A - B);
	Mat C = Mat::zeros(A.size(), CV_8UC1);
	for (int i = 0; i < A.rows; i++){
		for (int j = 0; j < A.cols; j++){
			if (A.at<uchar>(i, j) == B.at<uchar>(i, j)){
				C.at<uchar>(i, j) = 0;
			}
			else{
				C.at<uchar>(i, j) = 255;
			}
		}
	}
	namedWindow("hello", 0);
	imshow("hello", C);
	waitKey(0);
	return (r == Scalar::all(0));
}


// ��� ��˹��
void convolve_gauss(Mat src, Mat& dst, double sigma, long deriv_type){
	Mat orders1;
	switch (deriv_type){
	case DERIV_R:
		orders1 = (Mat_<double>(1, 2) << 0, 1); break;
	case DERIV_C:
		orders1 = (Mat_<double>(1, 2) << 1, 0); break;
	case DERIV_RR:
		orders1 = (Mat_<double>(1, 2) << 0, 2); break;
	case DERIV_RC:
		orders1 = (Mat_<double>(1, 2) << 1, 1); break;
	case DERIV_CC:
		orders1 = (Mat_<double>(1, 2) << 2, 0); break;
	}

	dst = src.clone();
	gfilter(src, dst, sigma, orders1);
}

// Gaussian filtering and Gaussian derivative filters
void gfilter(Mat src, Mat& dst, double sigma, Mat orders){
	Mat LL = Mat::zeros(src.size(), CV_32FC1);
	int dims = 2;
	Mat kk;
	for (int i = 0; i < dims; i++){
		// ����kΪ������
		gaussiankernel((float)sigma, (int)orders.ptr<double>(0)[i], kk);

		// shift the dimension of the kernel
		if (i == 0){
			filter2D(src, LL, CV_32FC1, kk);
		}
		else if (i == 1){
			Mat kk1 = kk.clone();
			filter2D(LL, dst, CV_32FC1, kk1);
		}
	}
	dst.convertTo(dst, CV_64FC1);
}

// GAUSSIAN DERIVATIVE KERNEL - creates a gaussian deivative kernel.
void gaussiankernel(float sigma, int order, Mat& outputArray){
	float sigma2 = sigma*sigma;
	float sigma2szeRatio = (float)(3.0f + 0.25*order - 2.5 / (pow((double)(order - 6), 2.0) + pow((double)(order - 9), 2.0)));

	// calculate kernel size
	int	sz = int(ceil(float(sigma2szeRatio * sigma)));
	outputArray = Mat::zeros(2 * sz + 1, 1, CV_64FC1);

	double *temp = new double[2 * sz + 1];
	double *temp1 = new double[2 * sz + 1];
	double *temp2 = new double[2 * sz + 1];
	double *tempk = outputArray.ptr<double>(0);
	double *temppart = new double[2 * sz + 1];
	for (int i = 0; i < (2 * sz + 1); i++){
		*(temp + i) = -sz + i;	//	t[i] = -sze + i, t�ķ�Χ��-sze��sze
	}

	// CALCULATE GAUSSIAN
	for (int i = 0; i < (2 * sz + 1); i++){
		*(temp1 + i) = exp(-(*(temp + i))*(*(temp + i)) / (2 * sigma2));
		*(temp2 + i) = *(temp + i) / (sigma*sqrt(2.0f));
	}

	switch (order){
	case 0:
		memset(temppart, 1, sizeof(double)*(2 * sz + 1)); break;
	case 1:
		for (int i = 0; i < (2 * sz + 1); i++){
			*(temppart + i) = (*(temp2 + i)) * 2;
		}
		break;
	case 2:
		for (int i = 0; i < (2 * sz + 1); i++){
			*(temppart + i) = (*(temp2 + i))*(*(temp2 + i)) * 4 - 2;
		}
		break;
	default:
		cout << "There is a problem!" << endl;
		break;
	}

	// apply Hermite polynomial to gauss
	for (int i = 0; i < (2 * sz + 1); i++){
		*(tempk + i) = (pow(-1.0, (double)order))*(*(temppart + i))*(*(temp1 + i));
	}

	// Normalize
	double Sum = accumulate(&temp1[0], &temp1[2 * sz + 1], 0);
	double norm_default = 1.0 / Sum;
	double norm_hermite = 1.0 / (pow((sigma * sqrt(2.0)), order));
	for (int i = 0; i < (2 * sz + 1); i++){
		*(tempk + i) = (*(tempk + i))*(norm_default*norm_hermite);
	}

	delete[] temp;
	delete[] temp1;
	delete[] temp2;
	tempk = NULL;
	delete[] temppart;
}

//// �����ֵ�����ؾ�������ֵ
//double maxM(Mat input){
//	double maxValue;
//	minMaxLoc(input, NULL, &maxValue);
//	return maxValue;
//}

// �����ֵ�����ؾ�������ֵ����ƫ��ֵ
int maxIndex(Mat input){
	double maxValue;
	int max_index;
	minMaxIdx(convert2Vec(input), NULL, &maxValue, NULL, &max_index);
	return max_index;
}

// ����Сֵ�����ؾ������Сֵ����ƫ��ֵ
int minIndex(Mat input){
	double minValue;
	int min_index;
	minMaxIdx(convert2Vec(input), &minValue, NULL, &min_index, NULL);
	return min_index;
}

// ����Сֵ�����ؾ������Сֵ
double minM(Mat input){
	double minValue;
	minMaxLoc(input, &minValue, NULL);
	return minValue;
}

// ���Ƚϡ������뵥����ֵ ���ؾ���������inputͬ�ߴ磬����Mat��������ΪCV_32S1 
// ����cmpFlag = trueʱ����ĳλ�ó�MatԪ��С��value�򣬽��Mat�ô���Ϊ0��������Ϊ1
// ����cmpFlag = falseʱ����ĳλ�ó�MatԪ�ش���value�򣬽��Mat�ô���Ϊ0��������Ϊ1
Mat cmpMatVSVal(Mat input, double value, int cmpFlag /*= CMP_EQU*/){
	Mat result = Mat::zeros(input.size(), CV_32SC1);
	for (int i = 0; i < input.rows; i++){
		double *data = input.ptr<double>(i);
		int *r = result.ptr<int>(i);
		for (int j = 0; j < input.cols; j++){
			switch (cmpFlag)
			{
			case CMP_EQU:
				if (data[j] == value) r[j] = 1;
				break;
			case CMP_EQU_OR_GREATER:
				if (data[j] >= value) r[j] = 1;
				break;
			case CMP_GREATER:
				if (data[j] > value) r[j] = 1;
				break;
			case CMP_EQU_OR_LESS:
				if (data[j] <= value) r[j] = 1;
				break;
			case CMP_LESS:
				if (data[j] < value) r[j] = 1;
				break;
			default:
				break;
			}
		}
	}
	return result;
}

// ���Ƚϡ����������Ƚ� ���ؾ����������������ߴ磬����Mat��������ΪCV_32S1 
// ����cmpFlag = CMP_EQU				ʱ����ĳλ�ó�AԪ�ص���BԪ���򣬽��Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_EQU_OR_GREATER	ʱ����ĳλ�ó�AԪ�ش��ڵ���BԪ���򣬽��Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_GREATER			ʱ����ĳλ�ó�AԪ�ش���BԪ���򣬽��Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_EQU_OR_LESS		ʱ����ĳλ�ó�AԪ��С�ڵ���BԪ���򣬽��Mat�ô���Ϊ1��������Ϊ0
// ����cmpFlag = CMP_LESS			ʱ����ĳλ�ó�AԪ��С��BԪ���򣬽��Mat�ô���Ϊ1��������Ϊ0
Mat cmpMatVSMat(Mat A, Mat B, int cmpFlag  /*= CMP_EQU*/){
	CV_Assert(A.size() == B.size());
	Mat result = Mat::zeros(A.size(), CV_32SC1);
	for (int i = 0; i < A.rows; i++){
		double *dataA = A.ptr<double>(i);
		double *dataB = B.ptr<double>(i);
		int *r = result.ptr<int>(i);
		for (int j = 0; j < A.cols; j++){
			switch (cmpFlag)
			{
			case CMP_EQU:
				if (dataA[j] == dataB[j]) r[j] = 1;
				break;
			case CMP_EQU_OR_GREATER:
				if (dataA[j] >= dataB[j]) r[j] = 1;
				break;
			case CMP_GREATER:
				if (dataA[j] > dataB[j]) r[j] = 1;
				break;
			case CMP_EQU_OR_LESS:
				if (dataA[j] <= dataB[j]) r[j] = 1;
				break;
			case CMP_LESS:
				if (dataA[j] < dataB[j]) r[j] = 1;
				break;
			default:
				break;
			}
		}
	}
	return result;
}

// ����Ӿ��� get submatrix
Mat getSubMat(Mat src, int r1, int c1, int r2/*=-1*/, int c2/*=-1*/){
	if (r2 == -1) r2 = src.rows - 1;
	if (c2 == -1) c2 = src.cols - 1;
	if (r1 < 0 || r2 >= src.rows || c1 < 0 || c2 >= src.cols || r2 < r1 || c2 < c1){
		cout << "ERROR: Cannot get submatrix [" << r1 << ".." << r2 <<
			"] x [" << c1 << ".." << c2 << "]" <<
			" of a (" << src.rows << "x" << src.cols << ") matrix." << endl;
		return Mat();
	}
	return (src(Rect(c1, r1, c2 - c1 + 1, r2 - r1 + 1)).clone());
}

// �����󡿸��Ƶ� ��data���Ƶ�src��(c,r)λ�ô� ��r�У���c��
void setMat(const Mat& src, Mat data, int r, int c){
	if (r<0 || c<0 || r + data.rows > src.rows || c + data.cols>src.cols) {
		cout << "ERROR: Cannot set submatrix [" << r << ".." << r + data.rows - 1 <<
			"] x [" << c << ".." << c + data.cols - 1 << "]" <<
			" of a (" << src.rows << "x" << src.cols << ") matrix." << endl;
		return;
	}
	data.copyTo(src(Rect(c, r, data.cols, data.rows)));
}

// ���������öԽ���Ԫ��
void setDiagf(Mat& src, float s, int i1, int i2/*=-1*/){
	if (i2 == -1) i2 = min(src.rows - 1, src.cols - 1);
	for (int i = i1; i <= i2; i++)
		src.at<float>(i, i) = s;
}

// ����N�׸���Ҷ����Mat ˫ͨ��  ��j�е�k�е�Ԫ�ر��ʽΪexp��2��ijk/n��
Mat FourierMat(int n){
	Mat result = Mat::zeros(n, n, CV_64FC2);
	double tmp = 2 * CV_PI / n;
	for (int j = 0; j < n; j++){
		for (int k = 0; k < n; k++){
			result.at<Point2d>(j, k).x = cos(tmp*j*k);
			result.at<Point2d>(j, k).y = sin(tmp*j*k);
		}
	}
	return result;
}

// ������Ļ���� 
// ����A�Ļ������A�в�ͬ�еĹ�һ���ڻ��ľ���ֵ�е����ֵ
double CrossCorrelationMat(Mat A){
	int k = A.cols;
	double result = 0;
	for (int i = 0; i < k; i++){
		for (int j = i + 1; j < k; j++){
			Mat a1 = A.col(i).clone();
			Mat a2 = A.col(j).clone();
			double temp = fabs(a1.dot(a2)) / norm(a1) / norm(a2);
			if (temp > result){
				result = temp;
			}
		}
	}
	return result;
}

// ������ֵ��Mat�и�Ԫ��ȡ����ֵ
Mat absMat(Mat input){
	if (input.channels() == 1){
		input.convertTo(input, CV_64FC1);
		return abs(input);
	}
	else if (input.channels() == 2){
		input.convertTo(input, CV_64FC2);
		cout <<"input = "<< input << endl;
		Mat result = Mat::zeros(input.size(), CV_64FC1);
		for (int i = 0; i < input.rows; i++){
			for (int j = 0; j < input.cols; j++){
				double x = input.at<Point2d>(i, j).x;
				//double y = input.at<Point2d>(i, j).y;
				result.at<double>(i, j) = x;
			}
		}
		return result;
	}
}

// �е�λ��
void colNormalized(Mat& input){
	Mat W = Mat::zeros(input.cols, input.cols, CV_64FC1);

	for (int i = 0; i < input.cols; i++){
		W.at<double>(i, i) = 1 / norm(input.col(i));
	}
	input = input*W;
}

// vector<int> ���L��Ԫ�ض�Ӧ���±�
vector<int> maxValuesIndex(Mat input, int L){
	CV_Assert(input.cols == 1);
	Mat c2(input);
	int n = input.rows;
	sortIdx(input, c2, SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
	//cout << "c1: \n" << input << endl;
	//cout << "c2: \n" << c2 << endl;
	vector<int> idxs;
	for (int i = 0; i < L; i++){
		idxs.push_back(c2.ptr<int>(0)[i]);
	}
	
	return idxs;
}

// ˳ʱ��ת90��
Mat turnRight(Mat input){
	Mat result;
	flip(input.t(), result, 1);	// ˮƽ��ת
	return result;
}

// ��ʱ��ת90��
Mat turnLeft(Mat input){
	Mat result;
	flip(input.t(), result, 0);	// ��ֱ��ת
	return result;
}

// �����ֵ ����������Щһ��һ�еľ���
double valM_double(Mat input){
	CV_Assert(input.total() == 1);
	CV_Assert(input.type() == CV_64FC1 || input.type() == CV_64F);
	return input.ptr<double>(0)[0];
}

// �����ֵ ����������Щһ��һ�еľ���
double valM_int(Mat input){
	CV_Assert(input.total() == 1);
	CV_Assert(input.type() == CV_32SC1 || input.type() == CV_32S);
	return input.ptr<int>(0)[0];
}

// ���ž��� �Ծ������sign���㣬�����ͬ����С�ľ��� ֵֻ��+1 ��-1
Mat signMat(Mat input){
	Mat result = Mat::ones(input.size(), CV_64FC1);
	for (int i = 0; i < (int)input.total(); i++){
		result.ptr<double>(0)[i] = signValue(input.ptr<double>(0)[i]);
	}
	return result;
}

// ��ȡĳԪ�ص�����Ԫ��
// type ��Ϊ4���� �� 8����
vector<cv::Point> adjacentPixels(cv::Mat data, cv::Point p, int type/* = 8*/){
	vector<Point> result;
	for (int i = -1; i <= 1; i++){
		if (p.y + i < 0 || p.y + i >= data.rows)
			continue;
		for (int j = -1; j <= 1; j++){
			if (p.x+j<0 || p.x+j>=data.cols)
				continue;
			if ((type == 8 && !(i==0&&j==0)) || (type == 4 && (j == 0 || i == 0) && i!=j))
				result.push_back(Point(p.x + j, p.y + i));
		}
	}
	return result;
}

// �ж����������Ƿ����
bool isEqual(cv::Mat A, cv::Mat B){
	CV_Assert(A.size() == B.size());
	return (norm(A - B) < 1e-12);
}

// ���󳤶� ��Ϊ��������Ϊ�������ȣ��������� �� ����Ϊ����
int length(cv::Mat m){
	if (m.cols == 1){
		return m.rows;
	}
	return m.cols;
}

// �󷴶Գƾ���
cv::Mat skewSymMat(cv::Mat input){
	CV_Assert((input.rows == 3 || input.cols == 3) && input.size().area() == 3);
	Mat result = Mat::zeros(3, 3, input.type());
	result.at<double>(0, 1) = -input.ptr<double>(0)[2];
	result.at<double>(0, 2) = input.ptr<double>(0)[1];
	result.at<double>(1, 0) = input.ptr<double>(0)[2];
	result.at<double>(1, 2) = -input.ptr<double>(0)[0];
	result.at<double>(2, 0) = -input.ptr<double>(0)[1];
	result.at<double>(2, 1) = input.ptr<double>(0)[0];
	return result;
}