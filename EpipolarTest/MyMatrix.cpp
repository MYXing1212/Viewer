#include"stdafx.h"
#include"MyMatrix.h"
/* Functions to compute the integral, and the 0th and 1st derivative of the
Gaussian function 1/(sqrt(2*PI)*sigma)*exp(-0.5*x^2/sigma^2) */

using namespace cv;


// 【矩阵】初始化 初始值皆为0
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

// 【矩阵】初始化 初始值皆为0
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

// 【矩阵】初始化
double* initMatrix(Mat A){
	if (A.type() != CV_64FC1 && A.type() != CV_64F){
		printf("double** initMatrix(Mat)  输入矩阵数据类型不匹配，应为double型\n");
		return initVector(1);
	}
	double* result = initVector(A.rows*A.cols);
	memcpy(result, A.ptr<double>(0), A.rows*A.cols*sizeof(double));
	return result;
}

// 【矩阵】初始化单位阵
double** eye(int len){
	double **result = initMatrix(len, len);
	for (int i = 0; i < len; i++){
		result[i][i] = 1;
	}
	return result;
}

// 【随机矩阵】均匀分布 给出a和b
Mat randnMatUniform(Size size, double a, double b){
	Mat result(size, CV_64FC1);
	Mat A = (Mat_<double>(1, 1) << a);
	Mat B = (Mat_<double>(1, 1) << b);
	unsigned int optional_seed = (unsigned int)time(NULL);
	cv::RNG rng(optional_seed);
	rng.fill(result, cv::RNG::UNIFORM, A, B);
	return result;
}

// 【随机矩阵】数值呈高斯分布，给出Size，mean，和std
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

// 【初始化】矩阵初始化 原始矩阵src 中的某些 下标为Idxs的元素构成的矩阵
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

// 【初始化】矩阵初始化 原始矩阵src 中的某些行和某些列元素构成新的矩阵
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

// // 【某些列 某列 列范围 范围列】矩阵选取某些列构成新的矩阵
Mat colSelect(Mat src, int startIdx, int endIdx){
	CV_Assert(startIdx >= 0 && startIdx < src.cols);
	CV_Assert(endIdx >= 0 && endIdx < src.cols);
	CV_Assert(endIdx >= startIdx);
	return src.colRange(startIdx, endIdx + 1);
}

// // 【某些行 某行 行范围 范围行】矩阵选取某些行构成新的矩阵
Mat rowSelect(Mat src, int startIdx, int endIdx){
	CV_Assert(startIdx >= 0 && startIdx < src.rows);
	CV_Assert(endIdx >= 0 && endIdx < src.rows);
	CV_Assert(endIdx >= startIdx);
	return src.rowRange(startIdx, endIdx + 1);
}

// // 【某些列 某列】矩阵选取某些列构成新的矩阵
Mat colSelect(Mat src, set<int> colIndexs){
	set<int>::iterator k = colIndexs.begin();
	int c = (int)colIndexs.size();
	Mat result = Mat::zeros(src.rows, c, src.type());
	for (int i = 0; k != colIndexs.end(); k++, i++){
		src.col(*k).copyTo(result.col(i));
	}
	return result;
}

// // 【某些列 某列】矩阵选取某些列构成新的矩阵
Mat colSelect(Mat src, vector<int> colIndexs){
	vector<int>::iterator k = colIndexs.begin();
	int c = (int)colIndexs.size();
	Mat result = Mat::zeros(src.rows, c, src.type());
	for (int i = 0; k != colIndexs.end(); k++, i++){
		src.col(*k).copyTo(result.col(i));
	}
	return result;
}

// // 【某些行 某行】矩阵选取某些行构成新的矩阵
Mat rowSelect(Mat src, vector<int> rowIndexs){
	vector<int>::iterator k = rowIndexs.begin();
	int r = (int)rowIndexs.size();
	Mat result = Mat::zeros(r, src.cols, src.type());
	for (int i = 0; k != rowIndexs.end(); k++, i++){
		src.row(*k).copyTo(result.row(i));
	}
	return result;
}

// // 【某些行 某行】矩阵选取某些行构成新的矩阵
Mat rowSelect(Mat src, set<int> rowIndexs){
	set<int>::iterator k = rowIndexs.begin();
	int c = (int)rowIndexs.size();
	Mat result = Mat::zeros(c, src.cols, src.type());
	for (int i = 0; k != rowIndexs.end(); k++, i++){
		src.row(*k).copyTo(result.row(i));
	}
	return result;
}

// 【初始化】对角矩阵 data是一维向量
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

// 【随机矩阵】 初始值随机0到1之间
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

// 【随机矩阵】 初始值随机0到1之间
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

// 【矩阵】初始化，初始值随机 0到1之间
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

// 【矩阵】对角阵，输入为对角元素Vec，其他元素为0
double** diagMatrix(double*vec, int len){
	double** result = initMatrix(len, len);
	for (int i = 0; i < len; i++){
		result[i][i] = vec[i];
	}
	return result;
}

// 【矩阵】复制
double* copyMat(double* input, int row, int col){
	double* result = initVector(row*col);
	for (int i = 0; i < row*col; i++)
		result[i] = input[i];
	return result;
}

// 【矩阵】 复制
void copyMatInt(int**src, int ** dst, int row, int col)
{
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			dst[i][j] = src[i][j];
		}
	}
}

// 【矩阵】复制
int** copyMatInt(int** src, int row, int col){
	int **result = initMatrixInt(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = src[i][j];
		}
	}
	return result;
}

// 【矩阵】复制
double** copyMat(double** src, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = src[i][j];
		}
	}
	return result;
}

// 【矩阵】返回旋转矩阵 绕z轴
double** Rot_z(double angle){
	double **result = eye(3);
	result[0][0] = cos(angle);
	result[0][1] = -sin(angle);
	result[1][0] = sin(angle);
	result[1][1] = cos(angle);
	return result;
}

// 【矩阵】返回旋转矩阵 绕x轴
double** Rot_x(double angle){
	double**result = eye(3);
	result[1][1] = cos(angle);
	result[1][2] = -sin(angle);
	result[2][1] = sin(angle);
	result[2][2] = cos(angle);
	return result;
}

// 【矩阵】初始化 用向量进行初始化
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

// 【矩阵】得到某一行 注意从第0行开始
double* rowM(double* mat, int row, int col, int rowNo){
	double *result = initVector(col);
	if (rowNo > row){
		printf("rowM(double*, int, int ,int) 获取矩阵的某一行： ERROR 超出矩阵范围！！！");
		return result;
	}


	for (int i = 0; i < col; i++){
		result[i] = mat[rowNo*col + i];
	}
	return result;
}

// 【矩阵】得到某一行
double* rowM(double** mat, int rowOffset){
	return mat[rowOffset - 1];
}

// 交换矩阵的两行
void exchange2rows(double* mat, int row, int col, int rowNo1, int rowNo2){
	if (rowNo1 >= row || rowNo2 >= row){
		printf("exchange2rows(double*, int, int ,int, int) 交换矩阵的某一行： ERROR 超出矩阵范围！！！");
		return;
	}

	// 要交换的两行是同一行
	if (rowNo1 == rowNo2){
		return;
	}

	for (int i = 0; i < col; i++){
		double tmp = mat[rowNo1*col + i];
		mat[rowNo1*col + i] = mat[rowNo2*col + i];
		mat[rowNo2*col + i] = tmp;
	}
}

// 【矩阵】得到矩阵某一列
double* colM(double** mat, int row, int colOffset){
	double *result = initVector(row);
	for (int i = 0; i < row; i++)
		result[i] = mat[i][colOffset - 1];
	return result;
}

// 【矩阵】转置
double** T_Mat(double **src, int row, int col){
	double **result = initMatrix(col, row);
	for (int i = 0; i < col; i++){
		for (int j = 0; j < row; j++){
			result[i][j] = src[j][i];
		}
	}
	return result;
}

// 【矩阵】转置
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

// 【矩阵】求逆
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

// 【矩阵】求高阵的广义逆（M*N的矩阵，r(A) = N, 列满秩） 实数域 
double* InvHighMat(double* A, int m, int n){
	double* result = initVector(m*n);
	if (n > m)
	{
		printf("InvHighMat(double*, int, int) 输入矩阵非高阵！！！\n");
		return result;
	}
	double *A_T = T_Mat(A, m, n);			// A 的转置
	double *ATA = MmulM(A_T, A, n, m, n);	// A.t() * A
	//printMatrix(ATA, 2,2);				
	double *ATA_inv = InvMat(ATA, n);		// (A.t()*A).inv()
	//printMatrix(ATA_inv, 2, 2);
	//result = MmulM(ATA_inv, A_T, n, n, m);	// （A.t()*A）.inv()）*A.t()
	return result;
}

// 【矩阵】求低阵的广义逆（M*N的矩阵，r(A) = M, 行满秩）实数域
double* InvLowMat(double* A, int m, int n){
	double* result = initVector(m*n);
	if (n < m)
	{
		printf("InvLowMat(double*, int, int) 输入矩阵非低阵！！！\n");
		return result;
	}
	double *A_T = T_Mat(A, m, n);			// A.t()
	double *AAT = MmulM(A, A_T, m, n, m);	// A*A.t()
	double *AAT_inv = InvMat(AAT, m);		// (A*A.t()).inv()
	//printMatrix(A_T, n, m);
	result = MmulM(A_T, AAT_inv, n, m, m);	// A.t()*[(A*A.t()).inv()]
	return result;
}

// 【矩阵】求逆
double** InvMat(double** ppDbMat, int nLen)
{
	double *pDbSrc = new double[nLen*nLen];

	int *is, *js, i, j, k;
	// 保存要求逆的输入矩阵
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

				p = fabs(pDbSrc[i*nLen + j]);		// 找到绝对值最大的系数
				if (p>d)	{
					d = p;

					// 记录绝对值最大的系数的行、列索引
					is[k] = i;
					js[k] = j;
				}
			}
		}
		if (d + 1.0 == 1.0)	{					// 系数全是0，系数矩阵为0阵，此时为奇异矩阵
			delete is;
			delete js;
			printf("【  Error!  】奇异矩阵，不可求逆！！！\n");
			return NULL;
		}
		if (is[k] != k)		{					//	当前行不包含最大元素
			for (j = 0; j < nLen; j++)	{
				// 交换两行元素
				p = pDbSrc[k*nLen + j];
				pDbSrc[k*nLen + j] = pDbSrc[(is[k] * nLen) + j];
				pDbSrc[(is[k])*nLen + j] = p;
			}
		}

		if (js[k] != k)	{						// 当前列不包含最大元素
			for (i = 0; i < nLen; i++){
				// 交换两列元素
				p = pDbSrc[i*nLen + k];
				pDbSrc[i*nLen + k] = pDbSrc[i*nLen + (js[k])];
				pDbSrc[i*nLen + (js[k])] = p;
			}
		}

		pDbSrc[k*nLen + k] = 1.0 / pDbSrc[k*nLen + k];		// 求主元的倒数

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
		// 恢复列
		if (js[k] != k){
			for (j = 0; j < nLen; j++){
				p = pDbSrc[k*nLen + j];
				pDbSrc[k*nLen + j] = pDbSrc[(js[k])*nLen + j];
				pDbSrc[(js[k])*nLen + j] = p;
			}
		}
		// 恢复行
		if (is[k] != k){
			for (i = 0; i < nLen; i++){
				p = pDbSrc[i*nLen + k];
				pDbSrc[i*nLen + k] = pDbSrc[i*nLen + (is[k])];
				pDbSrc[i*nLen + (is[k])] = p;
			}
		}

	}


	// 将结果复制回系数矩阵ppDbMat
	nCnt = 0;
	for (i = 0; i < nLen; i++){
		for (j = 0; j < nLen; j++){
			ppDbMat[i][j] = pDbSrc[nCnt++];
		}
	}

	double** result = initMatrix(pDbSrc, nLen, nLen);

	// 释放空间
	delete is;
	delete js;
	delete pDbSrc;

	return result;
}

// 【矩阵】求和
double** sumM(double** mat1, double** mat2, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = mat1[i][j] + mat2[i][j];
		}
	}
	return result;
}

// 【矩阵】求和
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

// 【矩阵】求和 保证input的通道数为1
double sumM(Mat input){
	CV_Assert(input.channels() == 1);
	return sum(input)[0];
}

// 【矩阵】比例变换 每个元素乘上一系数
double** scaleM(double** mat, double scale, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = mat[i][j] * scale;
		}
	}
	return result;
}

// 【矩阵】求差
double** subM(double** mat, double offset, int row, int col){
	double **result = initMatrix(row, col);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			result[i][j] = mat[i][j] - offset;
		}
	}
	return result;
}



// 【矩阵】求差
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

// 【矩阵】求迹
double traceM(double** mat, int size) {
	double sum = 0;
	for (int i = 0; i < size; i++){
		sum += mat[i][i];
	}
	return sum;
}

// 【矩阵】求迹
double traceM(double *m, int n)
{
	double sum = 0.0;

	for (int i = 0; i < n; i++)
		sum += m[i*n + i];

	return sum;
}

// 【矩阵的秩】
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

// 【矩阵的秩】
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

// 【行列式】值
double detM(Mat src){
	if (src.cols != src.rows){
		printf("ERROR！！ --> double detM(Mat), 求解行列式的值 输入矩阵非方阵!!");
		return 0;
	}
	return determinant(src);
}

// 【行列式】值
double detM(double* mat, int n){
	Mat src(n, n, CV_64FC1, mat);
	return detM(src);
}

// 【协方差矩阵】求矩阵的协方差矩阵
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

// 【矩阵 相关系数矩阵】
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

// 【矩阵】两矩阵相乘 mat1的列数应等于mat2的行数
double** MmulM(double** mat1, double** mat2, int row1, int col1, int col2){
	double** result = initMatrix(row1, col2);
	for (int i = 0; i < row1; i++){
		for (int j = 0; j < col2; j++){
			result[i][j] = dotV(rowM(mat1, i + 1), colM(mat2, col1, j + 1), col1);
		}
	}
	return result;
}

// 【矩阵】两矩阵相乘 mat1的列数n应等于mat2的行数n   mat1 m*n    mat2 n*k
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

// 【矩阵】两矩阵相乘 mat1的列数应等于mat2的行数
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

// 【矩阵】 两矩阵对应元素相乘，mat1的尺寸应该等于mat2的尺寸
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

// 【矩阵】 两矩阵对应元素相乘，mat1的尺寸应该等于mat2的尺寸
float* MmulMEle(float* m1, float* m2, int row, int col){
	float *result = initVectorf(row * col);
	for (int i = 0; i < row*col; i++){
		result[i] = m1[i] * m2[i];
	}
	return result;
}

// 【矩阵】打印矩阵
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

// 【矩阵】打印矩阵
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

// 【矩阵】打印矩阵
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



// 【矩阵】计算对称矩阵最大的特征值和对应的特征向量 必须是对称矩阵！！！
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

// 【矩阵】计算对称矩阵绝对值最小的特征值和对应的特征向量 必须是对称矩阵！！！
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

// 【矩阵】计算 特征值与特征向量
int getEigensAndVecs(Mat input, Mat& eigens, Mat& vectors){
	cv::eigen(input, eigens, vectors);
	return rankM(input);
}

// 【矩阵】计算 特征值与特征向量 在使用这个函数的时候 
// eigens 和 vectors 应该是已经分配好内存的
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

// 分块矩阵 A, B在对角线，其余为0
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

// 基于Householder变换，将方阵A分解为A = QR，其中Q为正交矩阵，R为上三角阵
// 参数说明
// A：需要进行QR分解的方阵
// Q：分解得到的正交矩阵
// R：分解得到的上三角阵
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


// QR分解 R的主对角线都是正值 此时得到的结果与MATLAB中不一致
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

// 【矩阵 分解】对称正定矩阵的Cholesky分解
// 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解 A = L*L.t()
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

// 【矩阵 分解】对称正定矩阵的Cholesky分解
// 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解 A = L*L.t()
Mat cholesky(Mat A){
	CV_Assert(isPositiveDefinite(A) == POSITIVE_DEFINITE);
	CV_Assert(isSymmetry(A));
	double *data = A.ptr<double>(0);
	double *result = cholesky(data, A.rows);
	Mat L(A.size(), CV_64FC1, result);
	return L;
}

// 【矩阵 分解】对称正定矩阵的Cholesky分解
// 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解 A = C.t()*L
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

// 【矩阵】将矩阵约化为海森博格矩阵
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

// 【矩阵 广义逆】
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

// 【矩阵 广义逆】 伪逆 加号逆
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
/* a:存放m*n实矩阵A,返回时亦是奇异矩阵
/* m:行数 n：列数
/* u:存放m*m左奇异向量, v:存放n*n右奇异向量
/* eps:给定精度要求,  ka: max(m,n)+1
/* output:
/* 返回值如果为负数，表示迭代了60次，还未求出奇异值；返回值为非负数，正常
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


// 奇异值分解 SVD 辅助函数
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

// 奇异值分解 SVD  辅助函数
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

// 【矩阵分解】SVD分解 A = U*W*V.t(); U是m*m的酉矩阵 V是n*n的酉矩阵 W是m*n的矩阵
void svdDecomp(double* input, int m, int n, double *P, double* delta, double *Q){
	Mat src(m, n, CV_64FC1, input);

	Mat AHA = src.t()*src;
	Mat _vec(n, n, CV_64F);
	Mat _eig(n, 1, CV_64F);

	cv::eigen(AHA, _eig, _vec);

	int r = rankM(input, m, n);

	// A = P * Δ * Q.t()
	// A  m*n
	// P  m*r
	// Δ  r*r
	// Q  n*r
	Mat _Q(n, r, CV_64FC1);
	Mat _P(m, r, CV_64FC1);
	Mat _delta = Mat::zeros(r, r, CV_64FC1);


	// 参见矩阵理论课笔记P61
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
// 【矩阵分解】SVD分解 A = P*delta*Q.t() P是m*r的矩阵 delta是r*r的矩阵，且主对角线元素为A的特征值， Q为n*r的矩阵
void svdDecomp(Mat src, Mat& P, Mat& delta, Mat& Q){
	Mat AHA = src.t()*src;
	Mat _vec(src.cols, src.cols, CV_64F);
	Mat _eig;

	cv::eigen(AHA, _eig, _vec);

	int r = rankM(src);

	// A = P * Δ * Q.t()
	// A  m*n
	// P  m*r
	// Δ  r*r
	// Q  n*r
	Q.create(src.cols, r, CV_64FC1);
	P.create(src.rows, r, CV_64FC1);
	delta = Mat::zeros(r, r, CV_64FC1);

	// 参见矩阵理论课笔记P61
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


// 【矩阵】在输入矩阵的下面插入全1行
void AddOnesRow(Mat& input){
	Mat result(input.rows + 1, input.cols, input.type());
	input.copyTo(result(Rect(0, 0, input.cols, input.rows)));
	result.row(result.rows - 1).setTo(Scalar::all(1));
	result.copyTo(input);
}

// 【矩阵】点积 计算数组的点积，源数组mat1和mat2的维数必须一致，结果存放在result中
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

// 求实对称阵的全部特征值与特征向量
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

// 求解矩阵的最大最小值
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

// 判断是否是对称矩阵
bool isSymmetry(Mat input){
	CV_Assert(input.rows == input.cols);

	Mat T = abs(input - input.t());
	double result = fabs(sum(T)[0]);
	if (result < 1e-7)
		return true;
	else
		return false;
}

// 判断是否是正定矩阵
int isPositiveDefinite(Mat input){
	Mat eigens, vectors;
	cv::eigen(input, eigens, vectors);

	//cout << eigens << endl;

	// 是否是半正定
	double* p = eigens.ptr<double>(0);

	// 是否非正定
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

// 【条件数】cond
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

// Doolittle 分解 LU 分解
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

// Crout 分解 LU 分解
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

// 【奇异阵】判断矩阵是否为奇异阵
bool isSingular(Mat A){
	CV_Assert(isSquare(A));
	if (fabs(determinant(A)) < 1e-12)
		return true;
	else
		return false;
}

// 【满秩】判断矩阵是否满秩
bool isFullRank(Mat A){
	CV_Assert(isSquare(A));

	if (rankM(A) == A.cols)
		return true;
	else
		return false;
}

// 【方阵】判断矩阵是否为方阵
bool isSquare(Mat A){
	return (A.rows == A.cols);
}

// 【无穷范数】求矩阵元素绝对值最大值
double infiniteNorm(Mat A){
	return norm(A, NORM_INF);
}

// 【1范数】求矩阵的1范数 1-范数
double norm_L1(cv::Mat A){
	return norm(A, NORM_L1);
}


// 【对角线元素最大值】
double maxValueInDiag(Mat input){
	Mat d = input.diag();
	double minV, maxV;
	minMaxLoc(d, &minV, &maxV);
	return maxV;
}

// svd分解 与MATLAB功能一致 这里的V和MATLAB中的v是转置关系
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

// 求二维傅里叶变换 double型 输入为单通道 实数; 返回值为复数
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

		// 求幅值和相位信息
		magnitude(planes[0], planes[1], mag);
		phase(planes[0], planes[1], angle);

		return complexI;
	}
	else if (input.cols > 1){						// 每列进行傅里叶变换
		Mat c(input.size(), CV_64FC2);
		for (int i = 0; i < input.cols; i++){
			Mat m, a;
			Mat tmp = fft(input.col(i), m, a);
			tmp.copyTo(c.col(i));
		}
		return c;
	}
}

//// 将矩阵展成1维
//Mat Mat2Vec(Mat input){
//	Mat result = Mat::zeros(input.cols*input.rows, 1, CV_64FC1);
//	double * presult = result.ptr<double>(0);
//	double * pinput = input.ptr<double>(0);
//	for (int i = 0; i < input.rows*input.cols; i++){
//		*presult++ = *pinput++;
//	}
//	return result;
//}


// 分块矩阵组合在一起 纵向 
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

// 分块矩阵组合在一起 横向
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

// Hermite转置
Mat hermiteT(Mat input){
	CV_Assert(input.channels() == 2);
	Mat result = input.t();
	for (int i = 0; i < result.rows*result.cols; i++){
		result.ptr<Point2d>(0)[i].y *= -1.0;
	}
	return result;
}

// 复数乘法 逐像素
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

// 复数自然指数 
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

// 复数乘法 X,Y 都是双通道的  返回值也是双通道的
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

// 复数乘法 乘以系数 Point2d X * Point2d Y
Point2d complexMul(Point2d X, Point2d Y){
	return Point2d(X.x*Y.x - X.y*Y.y, X.x*Y.y + X.y*Y.x);
}

// 复数乘法 乘以系数 Point2d X * Mat Y
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

// Mat取实部 input双通道
Mat real(Mat input){
	CV_Assert(input.channels() == 2);
	vector<Mat> r;
	split(input, r);
	return r[0];
}

// Mat取虚部 input双通道
Mat imag(Mat input){
	CV_Assert(input.channels() == 2);
	vector<Mat> r;
	split(input, r);
	return r[1];
}

// 复数矩阵幅值
Mat complexAbsMat(Mat input){
	CV_Assert(input.channels() == 2);
	vector<Mat> r;
	split(input, r);
	Mat result;
	magnitude(r[0], r[1], result);
	return result;
}

// 实部和虚部合并为2通道矩阵
Mat merge(Mat real, Mat imag){
	vector<Mat> t;
	t.push_back(real);
	t.push_back(imag);
	Mat result;
	merge(t, result);
	return result;
}

// 实数矩阵转复数矩阵 相当于补充一个全零阵
Mat real2complex(Mat input){
	CV_Assert(input.channels() == 1);
	Mat t = input.clone();
	Mat z = Mat::zeros(t.size(), input.type());
	return merge(t, z);
}

// 主成分分析
// 输入： A		---	样本矩阵，每行为一个样本
//		 k		--- 降维至k维
// 输出： pcaA	--- 降维后的k维样本特征向量组成的矩阵，每行一个样本，列数k为降维后的样本特征数
//		 V		--- 主成分分量
void PCATrans(Mat input, int k, Mat& pcaA, Mat& V){
	input.convertTo(input, CV_64FC1);

	int r = input.rows;
	int c = input.cols;

	// 样本均值 每一列的均值
	Mat meanVec = input.row(0).clone();
	for (int i = 0; i < input.cols; i++){
		meanVec.ptr<double>(0)[i] = mean(input.col(i))[0];
	}
	//cout << meanVec << endl;

	// 计算协方差矩阵的转置covMatT
	Mat Z = (input - repmat(meanVec, r, 1));
	//cout << "Z = " << Z << endl;
	Mat covMatT = Z * Z.t();
	//cout << "covMatT = " << covMatT << endl;

	//cout << "ok" << endl;
	// 计算covMatT的前k个本征值和本征向量
	// 【矩阵】计算 特征值与特征向量
	Mat eigens;
	Mat vectors;
	cv::eigen(covMatT, eigens, vectors);

	//cout << "特征值: " << eigens << endl;
	//cout << "特征向量: " << vectors << endl;
	//cout << "nihao" << endl;
	vectors(Rect(0, 0, vectors.cols, k)).copyTo(V);
	Mat D = eigens(Rect(0, 0, eigens.cols, k)).clone();

	//cout << V << endl;

	// 得到协方差矩阵(covMatT)'的本征向量
	V = Z.t()*V.t();


	// 本征向量归一化为单位本征向量
	for (int i = 0; i < V.cols; i++){
		double temp = norm(V.col(i));
		V.col(i) /= temp;
	}

	//cout << "V = " << V << endl;
	//cout << "V.size = " << V.size() << endl;

	// 线性变换（投影）降维至k维
	pcaA = Z * V;
	//cout << "pcaA = " << pcaA << endl;
}

// 展成列向量 按行展开
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

// toeplitz函数的功能是生成托普利兹（toeplitz）矩阵。
//	托普利兹矩阵的特点是：除第一行、第一列外，其他每个元素都与它左上角的元素相同。
//	调用格式：
//	A = toeplitz(第1列元素数组，第1行元素数组）
// 注意：第1行的第1个元素应与第1列的第1个元素相同，否则第1行的第一个元素将自动改为1列的第1个元素。
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

// hankel函数的功能是生成Hankel矩阵。
//	Hankel矩阵的特点是：除第一列、最后一行外，其他每个元素都与它左下角的元素相同。
//	调用格式：
//	A = hankel(第1列元素数组,最后一行元素数组）
// 注意：最后一行的第1个元素应与第1列的第1个元素相同，否则最后一行的第一个元素将自动改为1列的第1个元素。
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

// 合并两矩阵 默认为横向合并 左右拼接 左右合并 上下拼接 上下合并
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


// 将矩阵中绝对值小于val的元素置零 返回的是掩膜图像 置零元素为0 其余像素为255
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

// 判断两个矩阵是否相同 两幅图像是否相同
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


// 卷积 高斯核
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
		// 返回k为列向量
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
		*(temp + i) = -sz + i;	//	t[i] = -sze + i, t的范围从-sze到sze
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

//// 【最大值】返回矩阵的最大值
//double maxM(Mat input){
//	double maxValue;
//	minMaxLoc(input, NULL, &maxValue);
//	return maxValue;
//}

// 【最大值】返回矩阵的最大值所在偏移值
int maxIndex(Mat input){
	double maxValue;
	int max_index;
	minMaxIdx(convert2Vec(input), NULL, &maxValue, NULL, &max_index);
	return max_index;
}

// 【最小值】返回矩阵的最小值所在偏移值
int minIndex(Mat input){
	double minValue;
	int min_index;
	minMaxIdx(convert2Vec(input), &minValue, NULL, &min_index, NULL);
	return min_index;
}

// 【最小值】返回矩阵的最小值
double minM(Mat input){
	double minValue;
	minMaxLoc(input, &minValue, NULL);
	return minValue;
}

// 【比较】矩阵与单个数值 返回矩阵与输入input同尺寸，返回Mat数据类型为CV_32S1 
// 参数cmpFlag = true时，若某位置出Mat元素小于value则，结果Mat该处置为0，否则置为1
// 参数cmpFlag = false时，若某位置出Mat元素大于value则，结果Mat该处置为0，否则置为1
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

// 【比较】矩阵与矩阵比较 返回矩阵与两个输入矩阵尺寸，返回Mat数据类型为CV_32S1 
// 参数cmpFlag = CMP_EQU				时，若某位置出A元素等于B元素则，结果Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_EQU_OR_GREATER	时，若某位置出A元素大于等于B元素则，结果Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_GREATER			时，若某位置出A元素大于B元素则，结果Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_EQU_OR_LESS		时，若某位置出A元素小于等于B元素则，结果Mat该处置为1，否则置为0
// 参数cmpFlag = CMP_LESS			时，若某位置出A元素小于B元素则，结果Mat该处置为1，否则置为0
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

// 获得子矩阵 get submatrix
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

// 【矩阵】复制到 将data复制到src的(c,r)位置处 第r行，第c列
void setMat(const Mat& src, Mat data, int r, int c){
	if (r<0 || c<0 || r + data.rows > src.rows || c + data.cols>src.cols) {
		cout << "ERROR: Cannot set submatrix [" << r << ".." << r + data.rows - 1 <<
			"] x [" << c << ".." << c + data.cols - 1 << "]" <<
			" of a (" << src.rows << "x" << src.cols << ") matrix." << endl;
		return;
	}
	data.copyTo(src(Rect(c, r, data.cols, data.rows)));
}

// 【矩阵】设置对角线元素
void setDiagf(Mat& src, float s, int i1, int i2/*=-1*/){
	if (i2 == -1) i2 = min(src.rows - 1, src.cols - 1);
	for (int i = i1; i <= i2; i++)
		src.at<float>(i, i) = s;
}

// 生成N阶傅里叶矩阵Mat 双通道  第j行第k列的元素表达式为exp（2πijk/n）
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

// 求解矩阵的互相关 
// 矩阵A的互相关是A中不同列的归一化内积的绝对值中的最大值
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

// 【绝对值】Mat中各元素取绝对值
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

// 列单位化
void colNormalized(Mat& input){
	Mat W = Mat::zeros(input.cols, input.cols, CV_64FC1);

	for (int i = 0; i < input.cols; i++){
		W.at<double>(i, i) = 1 / norm(input.col(i));
	}
	input = input*W;
}

// vector<int> 最大L个元素对应的下标
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

// 顺时针转90°
Mat turnRight(Mat input){
	Mat result;
	flip(input.t(), result, 1);	// 水平翻转
	return result;
}

// 逆时针转90°
Mat turnLeft(Mat input){
	Mat result;
	flip(input.t(), result, 0);	// 竖直翻转
	return result;
}

// 矩阵的值 仅适用于那些一行一列的矩阵
double valM_double(Mat input){
	CV_Assert(input.total() == 1);
	CV_Assert(input.type() == CV_64FC1 || input.type() == CV_64F);
	return input.ptr<double>(0)[0];
}

// 矩阵的值 仅适用于那些一行一列的矩阵
double valM_int(Mat input){
	CV_Assert(input.total() == 1);
	CV_Assert(input.type() == CV_32SC1 || input.type() == CV_32S);
	return input.ptr<int>(0)[0];
}

// 符号矩阵 对矩阵进行sign运算，结果是同样大小的矩阵 值只有+1 和-1
Mat signMat(Mat input){
	Mat result = Mat::ones(input.size(), CV_64FC1);
	for (int i = 0; i < (int)input.total(); i++){
		result.ptr<double>(0)[i] = signValue(input.ptr<double>(0)[i]);
	}
	return result;
}

// 获取某元素的邻域元素
// type 分为4邻域 和 8邻域
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

// 判断两个矩阵是否相等
bool isEqual(cv::Mat A, cv::Mat B){
	CV_Assert(A.size() == B.size());
	return (norm(A - B) < 1e-12);
}

// 矩阵长度 若为向量，则为向量长度，若非向量 则 长度为列数
int length(cv::Mat m){
	if (m.cols == 1){
		return m.rows;
	}
	return m.cols;
}

// 求反对称矩阵
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