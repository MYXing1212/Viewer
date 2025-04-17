#include"stdafx.h"
//#include"stdafx.h"
#include"MyImage.h"
#include <direct.h>  

using namespace cv;

// DCT量化数组
int Quant_table[8][8] = {
	16, 11, 10, 16, 24, 40, 51, 61,
	12, 12, 14, 19, 26, 58, 60, 55,
	14, 13, 16, 24, 40, 57, 69, 56,
	14, 17, 22, 29, 51, 87, 80, 62,
	18, 22, 37, 56, 68, 109, 103, 77,
	24, 35, 55, 64, 81, 104, 113, 92,
	49, 64, 78, 87, 103, 121, 120, 101,
	72, 92, 95, 98, 112, 100, 103, 99
};

int threadImgNum = 0;
int threadEndNum = 0;

CRITICAL_SECTION cs;						// 临界区结构对象


MyImage::MyImage()
{
	my_color[0] = Scalar(0, 0, 255);
	my_color[1] = Scalar(0, 255, 0);
	my_color[2] = Scalar(255, 0, 0);
}

MyImage::~MyImage()
{}

DWORD WINAPI ThreadReadImages(LPVOID lpParameter)
{
	loadImgParas *l = (loadImgParas*)lpParameter;
	//printf("path = %s\n", (*((l->it)+threadImgNum)).c_str());

	EnterCriticalSection(&cs);					// 进入临界区
	string path = *(l->it + threadImgNum);
	vector<Mat>::iterator image = l->iMat + threadImgNum;
	vector<Mat>::iterator imageRGB = l->iMatRGB + threadImgNum;
	++threadImgNum;
	cout << path << " 载入成功!" << endl;
	LeaveCriticalSection(&cs);					// 离开临界区	

	*image = imread(path, 0);
	*imageRGB = imread(path);

	if (!image->data){
		printf("---ERROR!!!---\n载入图像错误!!!\n");
		return false;
	}

	EnterCriticalSection(&cs);					// 进入临界区
	threadEndNum++;
	//cout << "threadEndNum = " << threadEndNum << endl;
	LeaveCriticalSection(&cs);					// 离开临界区
	return 0;
}

// 图像径向梯度变换
Mat MyImage::getRadialGradient(const Mat& img, Point center, double scale)
{
	Mat result = img.clone();
	for (int i = 0; i < img.rows; i++)
	for (int j = 0; j < img.cols; j++)
	{
		double dx = (double)(j - center.x) / center.x;
		double dy = (double)(i - center.y) / center.y;
		double weight = exp((dx*dx + dy*dy)*scale);
		if (img.channels() == 3)
		{
			result.at<Vec3b>(i, j)[0] = cvRound(img.at<Vec3b>(i, j)[0] * weight);
			result.at<Vec3b>(i, j)[1] = cvRound(img.at<Vec3b>(i, j)[1] * weight);
			result.at<Vec3b>(i, j)[2] = cvRound(img.at<Vec3b>(i, j)[2] * weight);
		}
		else
			result.at<uchar>(i, j) = cvRound(img.at<uchar>(i, j)*weight);
	}
	return result;
}



// 载入图像
bool loadImages(vector<Mat>& loadImages, vector<Mat>& loadImagesRGB, vector<CString>& FPths)
{
	vector<string> filepaths;
	InitializeCriticalSection(&cs);					// 初始化临界区
	// 设置过滤器
	char szFilter[] = "图片(*.bmp)|*.bmp|所有文件(*.*)|*.*||";

	AfxSetResourceHandle(GetModuleHandle(NULL));

	// 构造打开文件对话框   
	CFileDialog fileDlg(TRUE, _T("bmp"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | OFN_ALLOWMULTISELECT, Char2LPCTSTR(szFilter));

	// 为了实现多文件同时添加
	DWORD max_file = 40000;	// 定义own filename buffer 的大小
	TCHAR* lsf = new TCHAR[max_file];
	fileDlg.m_ofn.nMaxFile = max_file;
	fileDlg.m_ofn.lpstrFile = lsf;
	fileDlg.m_ofn.lpstrFile[0] = NULL;	// 初始化对话框

	int iReturn = fileDlg.DoModal();
	//system("cls");
	int nCount = 0;
	// 显示打开文件对话框   
	if (IDOK == iReturn)
	{
		POSITION pos = fileDlg.GetStartPosition();
		while (pos != NULL){
			CString tmpPath = fileDlg.GetNextPathName(pos);
			//AfxMessageBox(tmpPath);
			//printf("%s\n", CString2string(tmpPath).c_str());
			filepaths.push_back(CString2string(tmpPath));
			FPths.push_back(tmpPath);
			nCount++;
		}
	}
	else if (iReturn == IDCANCEL){
		return false;
	}

	if (CommDlgExtendedError() == FNERR_BUFFERTOOSMALL)
		AfxMessageBox(_T("BUFFERTOOSMALL"));

	printf("Count = %d\nfilepaths.size = %d\n", nCount, filepaths.size());
	loadImages.resize(nCount);
	loadImagesRGB.resize(nCount);

	HANDLE *hThread = new HANDLE[nCount];					// 线程句柄
	vector<string>::iterator istr = filepaths.begin();
	vector<Mat>::iterator im = loadImages.begin();
	vector<Mat>::iterator imrgb = loadImagesRGB.begin();
	loadImgParas iter(istr, im, imrgb);

	for (int i = 0; i < nCount; i++){
		hThread[i] = CreateThread(NULL, 0, ThreadReadImages, &iter, 0, NULL);
	}

	// 等待所有线程结束
	WaitForMultipleObjects(nCount, hThread, TRUE, INFINITE);
	printf("【1.】载入图像成功！！！~~~\n");

	threadImgNum = 0;
	threadEndNum = 0;
	for (int i = 0; i < 25; i++)
		CloseHandle(hThread[i]);
	DeleteCriticalSection(&cs);              // 释放临界区
	return true;
}


// 添加噪声
Mat addNoises(const Mat& image, int num, int type, double mu, double sigma)
{
	if (type == NOISE_PEPPER || type == NOISE_SALT){
		Mat tmp = image.clone();

		int hx = 0, hy = 0;
		for (int k = 0; k < num; k++)
		{
			srand(hx*hy);
			int i = rand() % tmp.cols;
			srand(hx*hy*hy + 4);
			int j = rand() % tmp.rows;
			if (tmp.channels() == 1) //灰度图
			{
				if (type == NOISE_SALT) tmp.at<uchar>(j, i) = 255;
				else if (type == NOISE_PEPPER) tmp.at<uchar>(j, i) = 0;
			}
			else if (tmp.channels() == 3) // 彩色图
			{
				if (type == NOISE_SALT)
				{
					tmp.at<Vec3b>(j, i)[0] = 255;
					tmp.at<Vec3b>(j, i)[1] = 255;
					tmp.at<Vec3b>(j, i)[2] = 255;
				}
				else if (type == NOISE_PEPPER)
				{
					tmp.at<Vec3b>(j, i)[0] = 0;
					tmp.at<Vec3b>(j, i)[1] = 0;
					tmp.at<Vec3b>(j, i)[2] = 0;
				}
			}
			hx = hx + 2;
			hy = hy + 3;
		}
		return tmp;
	}
	else if (type == NOISE_GAUSSIAN){
		cv::Mat resultImage = image.clone();
		int channels = resultImage.channels();
		int nRows = resultImage.rows;
		int nCols = resultImage.cols*channels;
		// 判断图像的连续性
		if (resultImage.isContinuous()){
			nCols = nCols*nRows;
			nRows = 1;
		}
		for (int i = 0; i < nRows; i++){
			for (int j = 0; j < nCols; j++){
				// 添加高斯噪声
				int val = (int)(resultImage.ptr<uchar>(i)[j] + generateGaussianNoise(mu, sigma) * 32);
				if (val < 0)
					val = 0;
				if (val > 255)
					val = 255;
				resultImage.ptr<uchar>(i)[j] = (uchar)val;
			}
		}
		return resultImage;
	}
	return Mat();
}

// 生成高斯噪声
double generateGaussianNoise(double mu, double sigma){
	// 定义小值
	const double epsilon = 1e-7;
	//const double epsilon = std::numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	// flag 为假构造高斯随机变量X
	if (!flag)
		return z1*sigma + mu;
	double u1, u2;
	// 构造随机变量
	do{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	// flag 为真构造高斯随机变量X
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}


//
// 图像翻转
Mat getFlipImage(const Mat& image, int type)
{
	Mat result;
	if (type == FLIP_VERTICAL)
		flip(image, result, 0);
	else if (type == FLIP_HORIZONTAL)
		flip(image, result, 1);
	else if (type == FLIP_ALL)
		flip(image, result, -1);
	return result;
}

Mat GrayTrans(const Mat image)
{
	if (image.channels() == 1)
		return image;
	else
	{
		Mat result;
		cvtColor(image, result, CV_BGR2GRAY);
		return result;
	}
}

// 颜色缩减模型
void MyImage::colorReduce(Mat &image, int div)
{
	int nl = image.rows;// 行数
	int nc = image.cols;// 列数

	// 图像是连续存储的吗？
	if (image.isContinuous())
	{
		// 没有对行进行填补
		nc = nc*nl;
		nl = 1;  // 一维数组
	}
	int n = static_cast<int>(
		log(static_cast<double>(div)) / log(2.0));
	// 用来对像素值进行取整的二进制掩膜
	uchar mask = 0xFF << n;//e.g.for div = 16,mask = 0xF0
	// for all pixels
	for (int j = 0; j < nl; j++)
	{
		// 第j行的地址
		uchar *data = image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++)
		{
			// 处理每个像素-------------
			*data++ = *data&mask + div / 2;
			*data++ = *data&mask + div / 2;
			*data++ = *data&mask + div / 2;
			// 像素处理结束
		}// 行处理结束
	}
}

// 基于拉普拉斯的图像锐化
void MyImage::sharpen(const Mat &image, Mat &result)
{
	if (image.channels() == 3)
	{
		printf("请输入一幅灰度图像！！！\n");
		return;
	}
	// 如有必要则分配图像
	result.create(image.size(), image.type());
	for (int j = 1; j < image.rows - 1; j++)
	{
		// 处理除了第一行和最后一行之外的所有行
		const uchar* previous =
			image.ptr<const uchar>(j - 1);// 上一行
		const uchar* current =
			image.ptr<const uchar>(j);  // 当前行
		const uchar* next =
			image.ptr<const uchar>(j + 1);// 下一行
		uchar *output = result.ptr<uchar>(j);// 输出行
		for (int i = 1; i < image.cols - 1; i++)
		{
			*output++ = saturate_cast<uchar>(
				5 * current[i] - current[i - 1]
				- current[i + 1] - previous[i] - next[i]);
		}
	}
	// 将未处理的像素设置为0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

void MyImage::sharpen2D(const Mat &image, Mat &result)
{
	// 构造核（所有项初始化为0）
	Mat kernel(3, 3, CV_32F, Scalar(0));
	// 对核元素进行赋值
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	// 对图像进行滤波
	filter2D(image, result, image.depth(), kernel);
}

// 得到图像的负片
Mat MyImage::getInvert(const Mat& image)
{
	int dim(256);
	Mat lut(1,            // 1D
		&dim,         // 256项
		CV_8U);       // uchar
	for (int i = 0; i < 256; i++)
		lut.at<uchar>(i) = 255 - i;
	return applyLookUp(image, lut);
}

// 对图像应用查表以生成新图像
Mat MyImage::applyLookUp(const Mat& image, const Mat& lookup)
{
	// 输出图像
	Mat result;
	// 应用查找表
	LUT(image, lookup, result);
	return result;
}

// 返回横向 f(x,y) = [f(x+1,y)-f(x-1,y)]/2 , x是所在列，y是所在行
Mat gradientX(Mat src, uchar thresh){
	Mat result = Mat::zeros(src.size(), CV_8UC1);
	result.col(0).setTo(0);
	result.col(src.cols - 1).setTo(0);
	src.convertTo(src, CV_32FC1);

	int nr = 1;
	int nc = src.rows*src.cols*src.channels();
	for (int i = 0; i < nr; i++){
		float* ptr = src.ptr<float>(i);
		uchar *pr = result.ptr<uchar>(i);
		for (int j = 0; j < nc; j++){
			if (j%src.cols != 0 && j%src.cols != (src.cols - 1)){
				uchar tmp = (uchar)fabs((*(ptr + 1) - *(ptr - 1)) / 2);
				if (tmp>thresh)
					*pr = tmp;
			}
			ptr++;
			pr++;
		}
	}
	return result;
}

// 返回纵向 f(x,y) = [f(x,y+1)-f(x,y-1)]/2 , x是所在列，y是所在行
Mat gradientY(Mat src, uchar thresh){
	Mat result = Mat::zeros(src.size(), CV_8UC1);
	result.row(0).setTo(0);
	result.row(src.rows - 1).setTo(0);
	src.convertTo(src, CV_32FC1);

	int nr = 1;
	int nc = src.rows*src.cols*src.channels();
	for (int i = 0; i < nr; i++){
		float* ptr = src.ptr<float>(i);
		uchar *pr = result.ptr<uchar>(i);
		for (int j = 0; j < nc; j++){
			if (j / src.cols && j / src.cols != (src.rows - 1))
			{
				uchar tmp = (uchar)fabs((*(ptr + src.cols) - *(ptr - src.cols)) / 2);
				if (tmp>thresh)
					*pr = tmp;
			}
			ptr++;
			pr++;
		}
	}
	return result;
}

// 返回梯度幅值 注意返回的是CV_32FC1格式 返回梯度幅值大于thresh且梯度方向在beginAngl与endAngl之间的【梯度幅值】
Mat gradientAmpl(Mat src, float thresh, float beginAngl, float endAngl){
	Mat resultT = Mat::zeros(src.size(), CV_32FC1);
	resultT.row(0).setTo(0);
	resultT.row(src.rows - 1).setTo(0);
	resultT.col(0).setTo(0);
	resultT.col(resultT.cols - 1).setTo(0);

	src.convertTo(src, CV_32FC1);

	int nr = 1;
	int nc = src.rows*src.cols*src.channels();
	for (int i = 0; i < nr; i++){
		float* ptr = src.ptr<float>(i);
		float *pr = resultT.ptr<float>(i);
		for (int j = 0; j < nc; j++){
			if (j / src.cols && j / src.cols != (src.rows - 1) && j % src.cols != 0 && j % src.cols != (src.cols - 1))
			{
				float x = fabs((*(ptr + src.cols) - *(ptr - src.cols)) / 2);
				float y = fabs((*(ptr + 1) - *(ptr - 1)) / 2);
				float tmp = sqrt(x*x + y*y);
				*pr = (tmp>thresh) ? tmp : 0;
			}
			ptr++;
			pr++;
		}
	}
	Mat mask = gradientAngl(src, beginAngl, endAngl);
	Mat result;
	resultT.copyTo(result, mask);
	return result;
}


// 返回梯度方向 注意返回的是CV_8UC1格式掩模
Mat gradientAngl(Mat src, float beginAngl, float endAngl){
	Mat result = Mat::zeros(src.size(), CV_8UC1);
	result.row(0).setTo(0);
	result.row(src.rows - 1).setTo(0);
	result.col(0).setTo(0);
	result.col(result.cols - 1).setTo(0);

	if (src.elemSize() != 4)
		//printf("src.depth = %d\n", src.elemSize());
		src.convertTo(src, CV_32FC1);
	//printf("src.depth = %d\n", src.elemSize());

	int nr = 1;
	int nc = src.rows*src.cols*src.channels();

	float floorV = (float)((beginAngl - 180.0) / 180.0*CV_PI);
	float ceilV = (float)((endAngl - 180.0) / 180.0*CV_PI);
	printf("floorV = %f\nceilV = %f\n", floorV, ceilV);

	for (int i = 0; i < nr; i++){
		float* ptr = src.ptr<float>(i);
		uchar *pr = result.ptr<uchar>(i);
		for (int j = 0; j < nc; j++){
			if (j / src.cols && j / src.cols != (src.rows - 1) && j % src.cols != 0 && j % src.cols != (src.cols - 1))
			{
				float x = *(ptr + src.cols) - *(ptr - src.cols);
				float y = *(ptr + 1) - *(ptr - 1);
				float tmp = atan2(x, y);
				//printf("tmp = %f\n", tmp);
				*pr = (tmp>floorV && tmp < ceilV) ? 255 : 0;
			}
			ptr++;
			pr++;
		}
	}
	return result;
}

/** 改进后，输入图像为二值图像 0或255
* @brief 对输入图像进行细化
* @param[in] src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param[out] dst为对src细化后的输出图像,格式与src格式相同，调用前需要分配空间，元素中只有0与1,1代表有元素，0代表为空白
* @param[in] maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
*/
cv::Mat thinImage(const cv::Mat & src, const int maxIterations)
{
	assert(src.type() == CV_8UC1);

	src /= 255;

	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达  
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点  
		//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空  
		}

		//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空  
		}
	}
	return dst * 255;
}

// 载入图像
cv::Mat loadImage(std::string filepath, bool grayScale){
	Mat result;
	if (grayScale)
		result = imread(filepath, 0);
	else
		result = imread(filepath);
	return result;
}

// 载入图像 载入同一文件夹下 命名规则相同的图片
bool loadImage(std::vector<cv::Mat>& images, std::string folderpath, std::string prefix, int num, std::string suffix, int initIndex){
	for (int i = 0; i < num; i++){
		Mat tempImg = imread(folderpath + "\\" + prefix + int2string(i + initIndex) + "." + suffix, 0);
		images.push_back(tempImg);
	}
	return true;
}

// 【绘图】在一张图像上显示vector<Point2f>
void drawVecPoints(Mat &mat, vector<Point> points, Scalar color, int thickness, bool useCross){
	for (int i = 0; i < (int)points.size(); i++){
		if (useCross)
			drawCross(mat, points[i], thickness, color, 1);
		else
			circle(mat, points[i], thickness, color, -1, 8, 0);
		/*string id = int2string(i);
		putText(mat, id, Point(points[i].x + 10, points[i].y - 5), FONT_HERSHEY_COMPLEX,
		0.9, Scalar(155, 255, 255), 2);*/
	}
}

// 【绘图】在一张图像上显示vector<Point2f>
void drawVecPoints(Mat &mat, vector<Point2f> points, Scalar color, int thickness, bool useCross){
	for (int i = 0; i < (int)points.size(); i++){
		if (useCross)
			drawCross(mat, Point((int)points[i].x, (int)points[i].y), 5, color, 1);
		else
			circle(mat, Point((int)points[i].x, (int)points[i].y), 5, color, thickness, 8, 0);
		/*string id = int2string(i);
		putText(mat, id, Point(points[i].x + 10, points[i].y - 5), FONT_HERSHEY_COMPLEX,
		0.9, Scalar(155, 255, 255), 2);*/
	}
}

// 【绘图】在一张图像上显示vector<Point2f>
Mat drawVecPoints(Mat mat, vector<Point2f> points, vector<Point> rc, int col){
	Mat result = mat.clone();
	for (int i = 0; i < (int)points.size(); i++){
		drawCross(result, Point((int)points[i].x, (int)points[i].y), 5, Scalar(255, 0, 0), 2);
		string id = int2string((rc[i].x - 1)*col + rc[i].y);
		putText(result, id, Point((int)points[i].x + 10, (int)points[i].y - 5), FONT_HERSHEY_COMPLEX,
			0.9, Scalar(155, 255, 255), 2);
	}
	return result;
}

// 【绘图】在一张图像上显示vector<Point2f>
Mat drawVecPoints(vector<Point2f> points, Size size, Scalar color, int thickness){
	Mat result = Mat::zeros(size, CV_8UC3);
	for (int i = 0; i < (int)points.size(); i++){
		drawCross(result, points[i], 5, color, thickness);
		//result.at<uchar>(points[i].y, points[i].x) = 255;
	}
	return result;
}

// 【绘图】在一张图像上绘制vector<Point2f> 画布固定大小为Size(800*600) 
Mat drawVecPoints(vector<Point2f> points, Scalar color, int thickness){
	Mat canvas(800, 1000, CV_8UC3);
	Mat drawarea = canvas(Rect(50, 50, 900, 700));
	canvas.setTo(MC_BLACK);

	vector<double> tmpx;
	vector<double> tmpy;
	double xmin, xmax, ymin, ymax;
	for (int i = 0; i < (int)points.size(); i++){
		tmpx.push_back((double)points[i].x);
		tmpy.push_back((double)points[i].y);
	}
	maxmin(tmpx, xmax, xmin);
	maxmin(tmpy, ymax, ymin);

	xmax = calTopNum(xmax, 1);
	xmin = calFloorNum(xmin, 1);
	ymax = calTopNum(ymax, 1);
	ymin = calFloorNum(ymin, 1);

	double spanx = xmax - xmin;
	double spany = ymax - ymin;
	for (int i = 0; i < (int)points.size(); i++){
		int cx = (int)((tmpx[i] - xmin) / spanx * (900 - 1));
		int cy = (int)((tmpy[i] - ymin) / spany * (700 - 1));
		drawCross(drawarea, Point(cx, cy), 5, color, 2);
	}

	return canvas;
}


// 【绘图】在一张图像上显示vector<Point2f> rc是该点的行列
Mat drawVecPoints(vector<Point2f> points, vector<Point> rc, Size size){
	Mat result = Mat::zeros(size, CV_8UC3);
	for (int i = 0; i < (int)points.size(); i++){
		//string id = int2string(i) + ":(" + int2string(rc[i].x) + "," + int2string(rc[i].y) + ")";
		string id = "(" + int2string(rc[i].x) + "," + int2string(rc[i].y) + ")";
		putText(result, id, Point((int)(points[i].x + 10), (int)(points[i].y - 5)), FONT_HERSHEY_COMPLEX,
			0.9, Scalar(0, 255, 255), 2);
		drawCross(result, points[i], 5, Scalar(255, 0, 0), 2);

		//result.at<uchar>(points[i].y, points[i].x) = 255;
	}
	return result;
}

// 【绘图】在图像上显示vector<Point2f> 顺次将点用直线连接起来 二维点序列
void drawSeqPoints(Mat& canvas, vector<Point2f> points, Scalar color/* = MC_WHITE*/, int lineWidth/* = 1*/){
	for (int i = 0; i < (int)points.size()-1; i++){
		line(canvas, Point(points[i].x, points[i].y),
			Point(points[i + 1].x, points[i + 1].y),
			color, lineWidth, 8, 0);
	}
}

//【绘制】 在图像上绘制十字  len为某条线的长度
void drawCross(Mat& src, cv::Point center, int len, Scalar& color, int thickness,
	int lineType, int shift){
	len = len / 2.0;
	if (center.x < 0 || center.x >= src.cols || center.y < 0 || center.y >= src.rows){
		return;
	}
	else {
		int x0 = (center.x - len) ? (center.x - len) : 0;
		int x1 = (src.cols >(center.x + len)) ? (center.x + len) : (src.cols - 1);
		int y0 = (center.y - len) ? (center.y - len) : 0;
		int y1 = (src.rows >(center.y + len)) ? (center.y + len) : (src.rows - 1);
		line(src, Point(x0, center.y), Point(x1, center.y), color, thickness, lineType, shift);
		line(src, Point(center.x, y0), Point(center.x, y1), color, thickness, lineType, shift);
	}
}

// 绘制椭圆 x1， y1 代表左上角点坐标  x2 ,y2代表右下角点坐标
void drawEllipse(Mat& src, double x1, double y1, double x2, double y2, Scalar color, int thickness){
	RotatedRect r(Point2f((x1 + x2) / 2, (y1 + y2) / 2), Size(fabs(x2 - x1), fabs(y2 - y1)), 0);
	ellipse(src, r, color, thickness);
}

// 【绘制】 在图像两点A→B之间绘制箭头
void drawArrow(Mat& src, cv::Point2f A, cv::Point2f B, Scalar& color, int thickness,
	int lineType, int shift){
	float lenRatio = 0.32f;
	float angle = 20;
	float D = dist(A, B);
	float len = (float)(D*lenRatio*tan(angle / 180.0 * CV_PI));
	Point2f O(B.x - (B.x - A.x)*lenRatio, B.y + (A.y - B.y)*lenRatio);
	Vec2f OP((B.y - A.y) / D*len, (B.x - A.x) / D*len);		// 构造与向量AB垂直的向量
	line(src, A, Point(B.x - (B.x - A.x)*lenRatio, B.y + (A.y - B.y)*lenRatio), color, thickness, lineType, shift);
	triangle(src, Point((int)B.x, (int)B.y), Point((int)(O.x - OP[0]), (int)(O.y + OP[1])),
		Point((int)(O.x + OP[0]), (int)(O.y - OP[1])), color);
}

// 【绘制】 在图像上绘制箭头，箭头的中心点坐标cen与箭头指向angle
// 方向规定水平向右为0°，竖直向上为90°，水平向左为180° angle为角度值
void drawArrow(cv::Mat& src, cv::Point cen, double angle, double len, cv::Scalar& color, int thickness/* = 1*/, int lineType/* = 8*/, int shift/* = 0*/){
	Point2f A, B;
	A.x = cen.x - cos(angle / 180.0*CV_PI)*len / 2.0;
	A.y = cen.y - sin(angle / 180.0*CV_PI)*len / 2.0;
	B.x = cen.x + cos(angle / 180.0*CV_PI)*len / 2.0;
	B.y = cen.y + sin(angle / 180.0*CV_PI)*len / 2.0;
	drawArrow(src, A, B, color, thickness, lineType, shift);
}

// 【绘制】在图像上绘制三角形
void triangle(cv::Mat& src, cv::Point A, cv::Point B, cv::Point C, cv::Scalar& color, int thickness/* = 1*/){
	int npt[1] = { 3 };
	Point points[1][3];
	points[0][0] = A;
	points[0][1] = B;
	points[0][2] = C;
	const Point* pt[1] = { points[0] };
	if (thickness>0)
		polylines(src, pt, npt, 1, 1, color, thickness);
	else if (thickness == -1)
		fillPoly(src, pt, npt, 1, color, 8);
}

// 【绘制】在图像上绘制正三角形 r为三角形的尺寸 r为三角形外接圆半径
// thickness = -1 为实心
void triangle(cv::Mat& src, cv::Point p, double r, cv::Scalar& color, int thickness/* = 1*/){
	double tmpR = r*cos(30.0 / 180 * CV_PI);
	double x1 = p.x - tmpR;
	double x2 = p.x + tmpR;
	double y_top = p.y - r;
	double y_bottom = p.y + r*sin(30.0/180*CV_PI);
	triangle(src, Point(x1, y_bottom), Point(x2, y_bottom), Point(p.x, y_top), color, thickness);
}

// 【绘制】正多边形 在图像上绘制正多边形 ,r为正多边形的外接圆半径 
// angleOff 为偏置角度 水平向右方向为0°，逆时针转则angleOff为正值
void regularPolygon(cv::Mat& src, Point p, int n, double r,  cv::Scalar& color,double angleOff/* = 0*/, int thickness/* = 1*/){
	int npt[1] = { n };
	Point *points[1];
	points[0] = (Point*)malloc(sizeof(Point)*n);
	for (int i = 0; i<n; i++){
		double angle = 360.0 / n * i / 180.0 * CV_PI + angleOff / 180.0*CV_PI;
		(points[0])[i].x = p.x + r*cos(angle);
		(points[0])[i].y = p.y - r*sin(angle);
	}
	const Point* pt[1] = { points[0] };
	if (thickness > 0)
		polylines(src, pt, npt, 1, 1, color, thickness);
	else if (thickness == -1)
		fillPoly(src, pt, npt, 1, color, 8);
}

// 【绘制】五角星 在图像上绘制五角星，r为五角星的外接圆尺寸
void drawStar(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness/* = 1*/){
	int npt[1] = { 10 };
	Point points[1][10];
	for (int i = 0; i<5; i++){	
		double angle = 360.0 / 5.0 / 180.0 * CV_PI;
		points[0][2 * i].x = p.x + 0.37902*r*sin(angle*i - angle/2);
		points[0][2 * i].y = p.y - 0.37902*r*cos(angle*i - angle / 2);
		points[0][2 * i + 1].x = p.x + r*sin(angle*i);
		points[0][2 * i + 1].y = p.y - r*cos(angle*i);
	}
	const Point* pt[1] = { points[0] };
	if (thickness > 0)
		polylines(src, pt, npt, 1, 1, color, thickness);
	else if (thickness == -1)
		fillPoly(src, pt, npt, 1, color, 8);
}

// 【绘制】方块 在图像上绘制方块，r为方块的边长
void drawSquare(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness/* = 1*/){
	rectangle(src, Rect(p.x - r / 2.0, p.y - r / 2.0, r, r), color, thickness);
}

// 【绘制】× 绘制叉号 r为某线的长度
void drawSkewCross(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness/* = 1*/){
	double tmp = r/sqrt(2.0)/2.0;
	line(src, Point(p.x - tmp, p.y - tmp), Point(p.x + tmp, p.y + tmp), color, thickness, 8, 0);
	line(src, Point(p.x + tmp, p.y - tmp), Point(p.x - tmp, p.y + tmp), color, thickness, 8, 0);
}

void drawDashLine(Mat& src, cv::Point2f A, cv::Point2f B, Scalar& color, int thickness, int lineType, int shift){
	Point2f a = A, b = A;
	double d = dist(A, B);
	double k = slope(A, B);
	int i = 0;
	while (1){
		//b = Point2f(a.x + 100, a.y + 100 * k);
		a = Point2f((float)(A.x + i), (float)(A.y + i*k));
		b = Point2f((float)(a.x + 5), (float)(a.y + 5 * k));
		if (a.x > src.cols - 1 || b.x > src.cols - 1 || a.y > src.rows - 1 || b.y > src.rows - 1)
			break;
		line(src, a, b, color, thickness, lineType, shift);
		i = i + 10;
	}
	/*line(src, A, B, color, thickness, lineType, shift);
	line(src, B, Point((int)(O.x - OP[0]), (int)(O.y + OP[1])), color, thickness, lineType, shift);
	line(src, B, Point((int)(O.x + OP[0]), (int)(O.y - OP[1])), color, thickness, lineType, shift);*/
}

// 绘制两组像点的偏差 用十字叉丝来表示
Mat drawDisparity(vector<Point2f> pts1, vector<Point2f> pts2, Scalar color){
	Mat canvas(800, 1000, CV_8UC3);
	Mat drawarea = canvas(Rect(50, 50, 900, 700));
	canvas.setTo(MC_BLACK);

	vector<double> tmpx;
	vector<double> tmpy;
	double xmin, xmax, ymin, ymax;
	for (int i = 0; i < (int)pts1.size(); i++){
		tmpx.push_back((double)(pts1[i].x - pts2[i].x));
		tmpy.push_back((double)(pts1[i].y - pts2[i].y));
	}
	maxmin(tmpx, xmax, xmin);
	maxmin(tmpy, ymax, ymin);

	xmax = calTopNum(xmax, 0.1);
	xmin = calFloorNum(xmin, 0.1);
	ymax = calTopNum(ymax, 0.1);
	ymin = calFloorNum(ymin, 0.1);

	double spanx = xmax - xmin;
	double spany = ymax - ymin;
	for (int i = 0; i < (int)pts1.size(); i++){
		int cx = (int)((tmpx[i] - xmin) / spanx * (900 - 1));
		int cy = (int)((tmpy[i] - ymin) / spany * (700 - 1));
		drawCross(drawarea, Point(cx, cy), 5, color, 2);
	}


	return canvas;
}

Mat drawDisparity(vector<vector<Point2f>> pts1, vector<vector<Point2f>>pts2){
	Mat canvas(800, 1000, CV_8UC3);
	Mat drawarea = canvas(Rect(100, 100, 800, 600));
	canvas.setTo(MC_WHITE);

	vector<double> tmpx;
	vector<double> tmpy;
	double xmin, xmax, ymin, ymax;
	for (int i = 0; i < (int)pts1.size(); i++){
		for (int j = 0; j < (int)pts1[i].size(); j++){
			tmpx.push_back((double)(pts1[i][j].x - pts2[i][j].x));
			tmpy.push_back((double)(pts1[i][j].y - pts2[i][j].y));
		}
	}
	maxmin(tmpx, xmax, xmin);
	maxmin(tmpy, ymax, ymin);
	//CString note;
	//note.Format(_T("%lf"), ymin);
	//AfxMessageBox(note);

	xmax = calTopNum(xmax, 0.1);
	xmin = calFloorNum(xmin, 0.1);
	ymax = calTopNum(ymax, 0.1);
	ymin = calFloorNum(ymin, 0.1);

	/*CString note;
	note.Format(_T("%lf"), ymax);
	AfxMessageBox(note);*/

	double spanx = xmax - xmin;
	double spany = ymax - ymin;
	double ratiox = 799 / spanx;
	double ratioy = 599 / spany;


	int index = 0;
	for (int i = 0; i < (int)pts1.size(); i++){
		for (int j = 0; j < (int)pts1[i].size(); j++){
			int cx = (int)((tmpx[index] - xmin)*ratiox);
			int cy = (int)((tmpy[index++] - ymin)*ratioy);
			drawCross(drawarea, Point(cx, cy), 5, colorList[i], 2);
		}
	}
	// 坐标轴
	drawArrow(canvas, Point2d(100, 700), Point2d(100, 20), MC_BLACK, 1, 8, 0);
	drawArrow(canvas, Point2d(100, 700), Point2d(980, 700), MC_BLACK, 1, 8, 0);

	double stepx = 0.5;
	double stepy = 0.4;
	// 纵坐标
	for (int i = 0; ymin + i*stepy <= ymax + stepy; i++){
		int tmpx = 25;
		int tmpy = (int)(700 - i*stepy*ratioy);
		cv::putText(canvas, double2string(ymin + i*stepy), Point2d(tmpx, tmpy), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	}

	// 横坐标
	for (int i = 0; xmin + i*stepx <= xmax + stepx*0.5; i++){
		int tmpx = (int)(95 + i*stepx*ratiox);
		int tmpy = 725;
		cv::putText(canvas, double2string(xmin + i*stepx), Point2d(tmpx, tmpy), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	}


	//cv::putText(canvas, "1.0", Point2d(30, 240), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	//cv::putText(canvas, "0.0", Point2d(30, 390), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	//cv::putText(canvas, "-1.0", Point2d(25, 540), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	////cv::putText(canvas, "0.2", Point2d(30, 580), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	//cv::putText(canvas, double2string(ymax), Point2d(30, 100), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	//cv::putText(canvas, double2string(ymin), Point2d(25, 700), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);

	/*cv::putText(canvas, double2string(xmax + spanx*0.01), Point2d(900, 720), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	cv::putText(canvas, "1.0", Point2d(700, 720), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	cv::putText(canvas, "0", Point2d(500, 720), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	cv::putText(canvas, "-1.0", Point2d(300, 720), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);*/


	cv::putText(canvas, "x", Point2d(500, 760), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	cv::putText(canvas, "y", Point2d(10, 450), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	return canvas;
}

// 找亚像素角点
vector<Point2f> getGoodFeaturePoints(const Mat& src, int maxCornerNum,
	double qualityLevel, double minDist, Mat mask, int blockSize, double k){
	vector<Point2f> corners;
	if (!src.data) {
		printf("【MyImage::getGoodFeaturePoints(src, maxCornerNum, qualityLevel, minDist, blockSize, k)函数src输入图片不能为空！！！】");
		return corners;
	}

	Mat gray(src.size(), CV_8UC1);
	if (src.channels() != 1)
		cvtColor(src, gray, CV_BGR2GRAY);
	else
		src.copyTo(gray);

	goodFeaturesToTrack(gray,
		corners,
		maxCornerNum,
		qualityLevel,
		minDist,
		mask,
		blockSize,
		false,
		k);

	Size winSize = Size(5, 5);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	cornerSubPix(gray, corners, winSize, zeroZone, criteria);

	/*Mat copy = Mat::zeros(src.size(), CV_8UC3);
	for (unsigned int i = 0; i < corners.size(); i++) {
	circle(copy, Point((int)corners[i].x, (int)corners[i].y), 3, Scalar(0, 255, 255));
	}
	imshow("corners1", copy);*/

	return corners;
}

// 【判断】点邻域是否有值在low 和 high之间 需确保src为灰度图像
bool hasNbhdPointInRange(Mat& src, cv::Point center, int radius, int low, int high){
	if (center.x < 0 || center.x >= src.cols || center.y < 0 || center.y >= src.rows){
		return false;
	}
	else {
		float x0 = (float)((center.x - radius) ? (center.x - radius) : 0);
		float x1 = (float)((src.cols >(center.x + radius)) ? (center.x + radius) : (src.cols - 1));
		float y0 = (float)((center.y - radius) ? (center.y - radius) : 0);
		float y1 = (float)((src.rows >(center.y + radius)) ? (center.y + radius) : (src.rows - 1));
		for (int i = (int)x0; i <= (int)x1; i++){
			for (int j = (int)y0; j <= (int)y1; j++){
				uchar it = src.at<uchar>(j, i);
				//printf("%d ", it);
				if (it <= high && it >= low)
					return true;
			}
		}
		//printf("\n");
	}
	return false;
}

// 【判断】点是否符合掩模
bool isInMask(Mat mask, cv::Point point){
	if (point.x < 0 || point.x >= mask.cols || point.y < 0 || point.y >= mask.rows){
		return false;
	}
	else{
		if (mask.at<uchar>(point.y, point.x) == 0)
			return false;
		else
			return true;
	}
}

// 【筛选】出合规矩的二维点
void siftGoodPoints(vector<Point2f>& points, Mat src){
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	for (int i = 0; i < (int)points.size(); i++){
		if (!hasNbhdPointInRange(gray, points[i], 5, 0, 80)
			){
			points.erase(points.begin() + i);
			i--;
		}
	}

}

// 统计掩模真值像素点个数
int countTrueNums(Mat mask){
	if (mask.channels() > 1){
		printf("【统计掩模真值像素点个数】EROOR:图像非单通道灰度图像！\n");
		return 0;
	}
	int num = 0;
	uchar * pt = mask.ptr<uchar>(0);
	for (int i = 0; i < mask.rows*mask.cols; i++){
		if (*pt++)
			num++;
	}
	return num;
}

// 把图片的四条边的像素置零
void setBorderZero(Mat& mat){
	mat.row(0).setTo(0);
	mat.row(mat.rows - 1).setTo(0);
	mat.col(0).setTo(0);
	mat.col(mat.cols - 1).setTo(0);
}

// 非零像素点转换为vector<Point>
vector<Point> getTruePoints(Mat mask){
	vector<Point> result;
	uchar * pt = mask.ptr<uchar>(0);
	for (int i = 0; i < mask.rows; i++){
		for (int j = 0; j < mask.cols; j++){
			if (*pt++)
				result.push_back(Point(j, i));
		}
	}
	return result;
}

// 返回图像对角线尺寸
float digLength(Mat mat){
	return sqrt((float)(mat.rows*mat.rows + mat.cols*mat.cols));
}

// 最佳二值化阈值选取
int otsu(cv::Mat dst){
	int i, j;
	int tmp;

	double u0, u1, w0, w1, u, uk;

	double cov;
	double maxcov = 0.0;
	int maxthread = 0;

	int hst[256] = { 0 };
	double pro_hst[256] = { 0.0 };

	int height = dst.cols;
	int width = dst.rows;

	//统计每个灰度的数量
	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			tmp = dst.at<uchar>(i, j);
			hst[tmp]++;
		}
	}

	//计算每个灰度级占图像中的概率
	for (i = 0; i < 256; i++)
		pro_hst[i] = (double)hst[i] / (double)(width*height);


	//计算平均灰度值
	u = 0.0;
	for (i = 0; i < 256; i++)
		u += i*pro_hst[i];

	double det = 0.0;
	for (i = 0; i < 256; i++)
		det += (i - u)*(i - u)*pro_hst[i];

	//统计前景和背景的平均灰度值，并计算类间方差

	for (i = 0; i < 256; i++){
		w0 = 0.0; w1 = 0.0; u0 = 0.0; u1 = 0.0; uk = 0.0;
		for (j = 0; j < i; j++){
			uk += j*pro_hst[j];
			w0 += pro_hst[j];
		}
		u0 = uk / w0;

		w1 = 1 - w0;
		u1 = (u - uk) / (1 - w0);

		//计算类间方差
		cov = w0*w1*(u1 - u0)*(u1 - u0);

		if (cov > maxcov)
		{
			maxcov = cov;
			maxthread = i;
		}
	}

	std::cout << maxthread << std::endl;
	return maxthread;
}

//OTSU方法计算图像二值化的自适应阈值 
/*
OTSU 算法可以说是自适应计算单阈值（用来转换灰度图像为二值图像）的简单高效方法。
下面的代码最早由 Ryan Dibble提供，此后经过多人Joerg.Schulenburg, R.Z.Liu 等修改，补正。

转自：http://forum.assuredigit.com/display_topic_threads.asp?ForumID=8&TopicID=3480

算法对输入的灰度图像的直方图进行分析，将直方图分成两个部分，
使得两部分之间的距离最大。划分点就是求得的阈值。

parameter:   *image         --- buffer for image
rows, cols     --- size of image
x0, y0, dx, dy   --- region of vector used for computing threshold
vvv             --- debug option, is 0, no debug information outputed
*/
/*======================================================================*/
/*   OTSU global thresholding routine                                 */
/*   takes a 2D unsigned char array pointer, number of rows, and     */
/*   number of cols in the array. returns the value of the threshold     */
/*======================================================================*/
int otsu(unsigned char *image, int rows, int cols, int x0, int y0, int dx, int dy, int vvv){
	int thresholdValue = 1;	// 阈值
	int ihist[256];			// 图像直方图，256个点

	// 对直方图置零
	memset(ihist, 0, sizeof(ihist));

	int gmin = 255, gmax = 0;
	// 生成直方图
	for (int i = y0 + 1; i < y0 + dy - 1; i++){
		unsigned char* np = &image[i*cols + x0 + 1];
		for (int j = x0 + 1; j < x0 + dx - 1; j++){
			ihist[*np]++;
			if (*np>gmax)
				gmax = *np;
			if (*np < gmin)
				gmin = *np;
			np++;	// next pixel
		}
	}

	// set up everyting 

	double sum = 0.0, csum = 0.0;
	int n = 0;

	for (int k = 0; k <= 255; k++){
		sum += (double)k*(double)ihist[k];		// x*f(x) 质量矩
		n += ihist[k];							// f(x) 质量
	}

	if (!n){
		// if n has no value, there is problem ...
		fprintf(stderr, "NOT NORMAL thresholdValue = 160\n");
		return (160);
	}

	// do the ostu global thresholding method
	double fmax = -1.0;
	int n1 = 0;
	for (int k = 0; k < 255; k++){
		n1 += ihist[k];
		if (!n1){
			continue;
		}
		int n2 = n - n1;
		if (n2 == 0){
			break;
		}
		csum += (double)k*ihist[k];
		double m1 = csum / n1;
		double m2 = (sum - csum) / n2;
		double sb = (double)n1*(double)n2*(m1 - m2)*(m1 - m2);
		/* bbg: note: can be optimized*/
		if (sb > fmax){
			fmax = sb;
			thresholdValue = k;
		}
	}

	// at this point we have our thresholding value

	// debug code to display thresholding values
	if (vvv & 1){
		fprintf(stderr, "# OTSU: thresholdValue = %d gmin = %d gmax = %d\n", thresholdValue, gmin, gmax);
	}

	return (thresholdValue);
}

Mat drawHistogram(vector<double> a, Scalar color)
{
	Mat canvas(800, 1000, CV_8UC3);
	Mat drawarea = canvas(Rect(100, 100, 800, 600));
	canvas.setTo(MC_WHITE);

	double maxv, minv;
	maxmin(a, maxv, minv);

	double ratio = 599 / maxv;
	int width = (int)((float)(800 - 30) / (float)(a.size() + 1));

	double meanError = meanVector(a);

	int avg_height = 599 - (int)(meanError * ratio - 1);
	drawDashLine(drawarea, Point(0, avg_height), Point(800, avg_height), MC_NAVY_BLUE, 2, 8, 0);
	putText(canvas, double2string(meanError, -5), Point2d(860, avg_height + 90), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	drawArrow(canvas, Point2d(100, 700), Point2d(100, 20), MC_BLACK, 1, 8, 0);
	drawArrow(canvas, Point2d(100, 700), Point2d(980, 700), MC_BLACK, 1, 8, 0);

	maxv = calTopNum(maxv, 0.1);
	//int interval = (int)(drawarea.rows / ((maxv / 0.1)-1));
	for (int i = 0; i < (int)((maxv / 0.1) + 1); i++){
		cv::putText(canvas, double2string(i*0.1), Point2d(30, 699 - (int)(i*0.1*ratio - 1)), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	}


	//
	//cv::putText(canvas, "0.8", Point2d(30, 220), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	//cv::putText(canvas, "0.6", Point2d(30, 340), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	//cv::putText(canvas, "0.4", Point2d(30, 460), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	//cv::putText(canvas, "0.2", Point2d(30, 580), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	//cv::putText(canvas, "0", Point2d(50, 700), CV_FONT_ITALIC, 1, MC_BLACK, 2);

	for (int i = 0; i < (int)a.size(); i++)
	{
		int height = (int)(a[i] * ratio - 1);
		rectangle(drawarea, Rect(width*(i + 1), 599 - height, width - 10, height), color, -1);
		cv::putText(canvas, int2string(i + 1), Point2d(100 + (width * 1 / 3) + width*(i + 1), 750), CV_FONT_ITALIC, 1, MC_BLACK, 2);
	}
	return canvas;
}

/* 修改BMP图像尺寸*/
void Resize(CBitmap* src, CBitmap *dst, cv::Size size){
	CDC dcScreen;
	dcScreen.Attach(::GetDC(NULL));

	// 取得原始圖檔的 dc
	CDC dcMemory;
	dcMemory.CreateCompatibleDC(&dcScreen);
	CBitmap *pOldOrgBitmap = dcMemory.SelectObject(src);

	// 建立新的结果图形 (指定大小)
	dst->CreateCompatibleBitmap(&dcScreen, size.width, size.height);

	CDC dcFixMemory;
	dcFixMemory.CreateCompatibleDC(&dcScreen);
	CBitmap *pOldReslutBitmap = dcFixMemory.SelectObject(dst);

	// 把原始图形缩放画到 Memory DC上面
	BITMAP bmpInfo;
	src->GetBitmap(&bmpInfo); // 取得 原始图形的宽度与高度
	int mode = SetStretchBltMode(dcFixMemory, COLORONCOLOR); //设置不失真缩放
	StretchBlt(dcFixMemory, 0, 0, size.width, size.height, dcMemory, 0, 0, bmpInfo.bmWidth, bmpInfo.bmHeight, SRCCOPY);
	//DC2.StretchBlt(0, 0, 200, 200, &DC1, 0, 0, info.bmWidth, info.bmHeight, SRCCOPY);
	SetStretchBltMode(dcFixMemory, mode);

	// Set Back
	dcMemory.SelectObject(pOldOrgBitmap);
	dcFixMemory.SelectObject(pOldReslutBitmap);
}
// 载入图像
bool loadImage(Mat& image, CString& filepath, bool grayScale)
{
	// 设置过滤器
	char szFilter[] = "图片(*.bmp)|*.bmp|所有文件(*.*)|*.*||";
	AfxSetResourceHandle(GetModuleHandle(NULL));

	// 构造打开文件对话框   
	CFileDialog fileDlg(TRUE, _T("bmp"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, Char2LPCTSTR(szFilter));

	int iReturn = fileDlg.DoModal();

	// 显示打开文件对话框   
	if (IDOK == iReturn)
	{
		filepath = fileDlg.GetPathName();
		if (grayScale)
			image = imread(CString2string(filepath), 0);
		else
			image = imread(CString2string(filepath));
	}
	else if (iReturn == IDCANCEL){
		return false;
	}

	if (CommDlgExtendedError() == FNERR_BUFFERTOOSMALL)
		AfxMessageBox(_T("BUFFERTOOSMALL"));
	return true;
}

// 返回横向梯度图
Mat gradX(Mat src){
	Mat grad_x, abs_grad_x;
	Scharr(src, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	return abs_grad_x;
}

// 返回纵向梯度图
Mat gradY(Mat src){
	Mat grad_y, abs_grad_y;
	Scharr(src, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	return abs_grad_y;
}

// 返回纵向梯度图
Mat gradXY(Mat src){
	Mat gx = gradX(src);
	Mat gy = gradY(src);
	Mat result;
	addWeighted(gx, 0.5, gy, 0.5, 0, result);
	return result;
}

// 获得平面旋转矩阵 带缩放 angle为角度制，逆时针转动angle为正值
Mat getRotatedImg(const Mat& img, double angle, double scale)
{
	Mat tmp = img.clone();          // 声明临时处理用的图片
	Point2f center = Point2f((float)tmp.cols / 2, (float)tmp.rows / 2);// 旋转中心  
	Mat matrix2D = getRotationMatrix2D(center, angle, scale);
	Mat rotateImg;
	warpAffine(tmp, rotateImg, matrix2D, tmp.size());
	return rotateImg;
}

// 旋转变换,原始图像的面积不变 其他部分为黑色 类似最小包围四边形
// 顺时针为正
cv::Mat angleRotate(cv::Mat& src, int angle){
	float theta = angle * CV_PI / 180.0f;

	int oldWidth = src.cols;
	int oldHeight = src.rows;

	// 源图像四个角的坐标（以图像中心为坐标系原点）
	float fSrcX1 = (float)(-(oldWidth - 1) / 2);
	float fSrcY1 = (float)((oldHeight - 1) / 2);

	float fSrcX2 = (float)((oldWidth - 1) / 2);
	float fSrcY2 = (float)((oldHeight - 1) / 2);

	float fSrcX3 = (float)(-(oldWidth - 1) / 2);
	float fSrcY3 = (float)(-(oldHeight - 1) / 2);

	float fSrcX4 = (float)((oldWidth - 1) / 2);
	float fSrcY4 = (float)(-(oldHeight - 1) / 2);

	// 旋转后四个角的坐标（以图像中心为坐标系原点）
	float fDstX1 = cos(theta)*fSrcX1 + sin(theta) * fSrcY1;
	float fDstY1 = -sin(theta)*fSrcX1 + cos(theta) * fSrcY1;

	float fDstX2 = cos(theta)*fSrcX2 + sin(theta) * fSrcY2;
	float fDstY2 = -sin(theta)*fSrcX2 + cos(theta) * fSrcY2;

	float fDstX3 = cos(theta)*fSrcX3 + sin(theta) * fSrcY3;
	float fDstY3 = -sin(theta)*fSrcX3 + cos(theta) * fSrcY3;

	float fDstX4 = cos(theta)*fSrcX4 + sin(theta) * fSrcY4;
	float fDstY4 = -sin(theta)*fSrcX4 + cos(theta) * fSrcY4;

	int newWidth = (max(fabs(fDstX4 - fDstX1), fabs(fDstX3 - fDstX2)) + 0.5);
	int newHeight = (max(fabs(fDstY4 - fDstY1), fabs(fDstY3 - fDstY2)) + 0.5);

	Mat dst = Mat::zeros(newHeight, newWidth, src.type());




	float dx = -0.5*newWidth*cos(theta) - 0.5*newHeight*sin(theta) + 0.5*oldWidth;
	float dy = 0.5*newWidth*sin(theta) - 0.5*newHeight*cos(theta) + 0.5*oldHeight;

	float x, y;
	for (int i = 0; i < newHeight; i++){
		for (int j = 0; j < newWidth; j++){
			x = float(j)*cos(theta) + float(i)*sin(theta) + dx;
			y = float(-j)*sin(theta) + float(i)*cos(theta) + dy;
			if ((x < 0) || (x >= oldWidth) || (y < 0) || (y >= oldHeight)){
				if (src.channels() == 3){
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
				else if (src.channels() == 1){
					dst.at<uchar>(i, j) = 0;
				}
			}
			else {
				if (src.channels() == 3){
					Vec3d tmp = bilinearlInterpolation(src, Point2f(x, y));
					dst.at<Vec3b>(i, j)[0] = static_cast<uchar>(tmp[0]);
					dst.at<Vec3b>(i, j)[1] = static_cast<uchar>(tmp[1]);
					dst.at<Vec3b>(i, j)[2] = static_cast<uchar>(tmp[2]);
					//dst.at<Vec3b>(i,j) = src.at<Vec3b>(y, x);
				}
				else if (src.channels() == 1){
					Vec3d tmp = bilinearlInterpolation(src, Point2f(x, y));
					dst.at<uchar>(i, j) = (uchar)(tmp[0]);
					//dst.at<uchar>(i, j) = src.at<uchar>(y, x);
				}
			}
		}
	}

	return dst;
}

Mat match2Img(Mat A, Mat B){
	vector<Point2f> selPoints1, selPoints2;

	vector<KeyPoint> keypoints1, keypoints2;
	vector<int> pointIndexes1, pointIndexes2;

//	SurfFeatureDetector surf(2500.);	// 阈值

//	surf.detect(A, keypoints1);
//	surf.detect(B, keypoints2);

	KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
	KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

	Mat fundemental = findFundamentalMat(Mat(selPoints1), Mat(selPoints2), CV_FM_7POINT);

	// 在右图中绘制对应的极线
	vector<Vec3f> lines1;
	computeCorrespondEpilines(
		Mat(selPoints1),
		1,
		fundemental,
		lines1);

	for (vector<cv::Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it){
		line(B,
			Point(0, (int)(-(*it)[2] / (*it)[1])),
			Point(B.cols, (int)(-((*it)[2] + (*it)[0] * B.cols) / (*it)[1])),
			MC_WHITE);
	}

	return B;
}

// 灰度图像变为三通道图像，另外两通道复制第一通道
Mat convert2BGR(Mat gray){
	if (gray.channels() == 3)
		return gray;
	else {
		Mat result;
		vector<Mat> v;
		v.push_back(gray);
		v.push_back(gray);
		v.push_back(gray);

		merge(v, result);
		return result;
	}
}

// 绘制H-S直方图 输入图像为彩色图像
Mat getH_SHistgram(Mat src, MatND& hist, int hueBinNum){
	if (src.channels() == 1){
		return Mat();
	}
	Mat hsvImage;
	cvtColor(src, hsvImage, CV_BGR2HSV);

	//int hueBinNum = 30;	// 色调的直方图直条数量
	int saturationBinNum = 32;		// 饱和度的直方图直条数量
	int histSize[] = { hueBinNum, saturationBinNum };

	// 定义色调的变化范围为0到179
	float hueRanges[] = { 0, 180 };
	// 定义饱和度的变化范围为0（黑、白、灰）到255（纯光谱颜色）
	float saturationRanges[] = { 0, 256 };
	const float* ranges[] = { hueRanges, saturationRanges };

	// 参数准备,calcHist函数中将计算第0通道和第1通道的直方图
	int channels[] = { 0, 1 };

	calcHist(&hsvImage,		// 输入的图像
		1,					// 数组个数为1
		channels,			// 通道索引
		Mat(),				// 不使用掩膜
		hist,			// 输出的目标直方图
		2,					// 需要计算的直方图的维度为2
		histSize,			// 存放每个维度的直方图尺寸的数组
		ranges,				// 每一维数值的取值范围数组
		true,				// 指示直方图是否均匀的标识符，true表示均匀的直方图
		false);				// 累计标识符，false表示直方图在配置阶段会被清零

	// 为绘制直方图准备参数
	double maxValue = 0;	// 最大值
	minMaxLoc(hist, 0, &maxValue, 0, 0);		// 查找数组和子数组的全局最小值
	int scale = 10;

	Mat histImg = Mat::zeros(saturationBinNum*scale, hueBinNum * 10, CV_8UC3);

	for (int hue = 0; hue < hueBinNum; hue++){
		for (int saturation = 0; saturation < saturationBinNum; saturation++){
			float binValue = hist.at<float>(hue, saturation);	// 直方图直条的值
			int intensity = cvRound(binValue * 255 / maxValue);		// 强度

			rectangle(histImg, Point(hue*scale, saturation*scale),
				Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1),
				Scalar::all(intensity), CV_FILLED);
		}
	}

	return histImg;

}

// 绘制RGB三色直方图 输入图像为彩色图像
Mat getRGBHistgram(Mat src, MatND& hist, int bins){
	int hist_size[] = { bins };
	float range[] = { 0, (float)bins };
	const float* ranges[] = { range };
	MatND redHist, greenHist, blueHist;
	int channels_r[] = { 0 };

	// 进行直方图的计算（红色分量部分）
	calcHist(&src, 1, channels_r, Mat(),		// 不使用掩膜
		redHist, 1, hist_size, ranges, true, false);

	// 进行直方图的计算（绿色分量部分）
	int channels_g[] = { 1 };
	calcHist(&src, 1, channels_g, Mat(),		// 不使用掩膜
		greenHist, 1, hist_size, ranges, true, false);

	// 进行直方图的计算（蓝色分量部分）
	int channels_b[] = { 2 };
	calcHist(&src, 1, channels_b, Mat(),		// 不使用掩膜
		blueHist, 1, hist_size, ranges, true, false);

	// 绘制出三色直方图
	double maxValue_red, maxValue_green, maxValue_blue;
	minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
	minMaxLoc(greenHist, 0, &maxValue_green, 0, 0);
	minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
	int scale = 1;
	int histHeight = 256;
	Mat histImage = Mat::zeros(histHeight, bins * 3, CV_8UC3);

	for (int i = 0; i < bins; i++){
		float binValue_red = redHist.at<float>(i);
		float binValue_green = greenHist.at<float>(i);
		float binValue_blue = blueHist.at<float>(i);
		int intensity_red = cvRound(binValue_red*histHeight / maxValue_red);
		int intensity_green = cvRound(binValue_green*histHeight / maxValue_green);
		int intensity_blue = cvRound(binValue_blue*histHeight / maxValue_blue);

		rectangle(histImage, Point(i*scale, histHeight - 1),
			Point((i + 1)*scale - 1, histHeight - intensity_red),
			Scalar(255, 0, 0));

		rectangle(histImage, Point((i + bins)*scale, histHeight - 1),
			Point((i + bins + 1)*scale - 1, histHeight - intensity_green),
			Scalar(0, 255, 0));

		rectangle(histImage, Point((i + bins * 2)*scale, histHeight - 1),
			Point((i + bins * 2 + 1)*scale - 1, histHeight - intensity_blue),
			Scalar(0, 0, 255));
	}

	return histImage;
}


// 绘制直方图 输入图像为彩色图像或灰度图像 若为彩色图像会转换为灰度图先
Mat getHistgram(Mat src, MatND &hist, int bins){

	if (src.channels() == 3){
		cvtColor(src, src, CV_BGR2GRAY);
	}

	int dims = 1;
	float hranges[] = { 0, (float)bins };
	const float *ranges[] = { hranges };
	int size = bins;
	int channels = 0;

	calcHist(&src, 1, &channels, Mat(), hist, dims, &size, ranges);
	/*CString note;
	note.Format(_T("%d"), src.channels());
	AfxMessageBox(note);*/

	int scale = 1;

	Mat dstImage(size*scale, size, CV_8U, Scalar(0));
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(hist, &minValue, &maxValue, 0, 0);

	int hpt = saturate_cast<int>(0.9*size);
	for (int i = 0; i < bins; i++){
		float binValue = hist.at<float>(i);
		int realValue = saturate_cast<int>(binValue*hpt / maxValue);
		rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
	}
	return dstImage;
}

// 计算两幅图像hist直方图比较的结果
// 四种比较方法为 
//【CV_COMP_CORREL】
//【CV_COMP_CHISQR】
//【CV_COMP_INTERSECT】
//【CV_COMP_BHATTACHARYYA】 
double calCompareH_SHist(Mat A, Mat B, int method){
	MatND histA, histB;
	getH_SHistgram(A, histA);
	getH_SHistgram(B, histB);
	return compareHist(histA, histB, method);
}

// 【反向投影】获取图像相对参考图像的反向投影图
MatND getBackProjImage(Mat src, Mat ref, Mat& hueRefHist, int bins, bool equalHist){
	if (src.channels() == 1){
		return MatND();
	}
	Mat g_hsvImage, g_hueImage, hsv_src, hue_src;
	cvtColor(ref, g_hsvImage, CV_BGR2HSV);
	cvtColor(src, hsv_src, CV_BGR2HSV);


	g_hueImage.create(g_hsvImage.size(), g_hsvImage.depth());
	hue_src.create(hsv_src.size(), hsv_src.depth());
	int ch[] = { 0, 0 };
	mixChannels(&g_hsvImage, 1, &g_hueImage, 1, ch, 1);
	mixChannels(&hsv_src, 1, &hue_src, 1, ch, 1);

	MatND hist;
	int histSize = max(bins, 2);
	float hue_range[] = { 0, 180 };
	const float* ranges = { hue_range };

	calcHist(&g_hueImage, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	cv::normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	// 计算反向投影
	MatND backproj;
	cv::calcBackProject(&hue_src, 1, 0, hist, backproj, &ranges, 1, true);

	int w = 400;
	int h = 400;
	int bin_w = cvRound((double)w / histSize);
	hueRefHist = Mat::zeros(w, h, CV_8UC3);
	for (int i = 0; i < bins; i++){
		rectangle(hueRefHist, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)),
			getSeqColor((float)i / bins), -1);
	}

	if (equalHist){
		Mat equalResult;
		equalizeHist(backproj, equalResult);
		return equalResult;
	}
	else {
		return backproj;
	}
}

// 模版匹配操作 method是匹配方法 共有六种
// 【CV_TM_SQDIFF】
// 【CV_TM_SQDIFF_NORMED】
// 【CV_TM_CCORR】
// 【CV_TM_CCORR_NORMED】
// 【CV_TM_CCOEFF】
// 【CV_TM_CCOEFF_NORMED】
Mat getTemplateMatchImage(Mat src, Mat ref, Point &p, int method){
	int resultImage_cols = src.cols - ref.cols + 1;
	int resultImage_rows = src.rows - ref.rows + 1;
	Mat g_resultImage(resultImage_cols, resultImage_rows, CV_32FC1);

	matchTemplate(src, ref, g_resultImage, method);
	normalize(g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat());

	double minValue;
	double maxValue;
	Point minLocation, maxLocation;
	minMaxLoc(g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());

	if (method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED){
		p = minLocation;
	}
	else {
		p = maxLocation;
	}

	rectangle(src, p, Point(p.x + ref.cols, p.y + ref.rows), Scalar(0, 0, 255), 2, 8, 0);
	return src;
}


// 【保存图像】保存BMP图像
bool saveBmp(CString bmpName, unsigned char* imgBuf, int width, int height, int byte){
	if (!imgBuf)
		return false;

	int palettesize = 0;
	if (byte == 1)
		palettesize = 1024;

	int lineByte = (width * byte + 3) / 4 * 4;

	FILE *fp = fopen(CString2pChar(bmpName), "wb");
	if (fp == 0)
		return false;

	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42;	// bmp
	fileHead.bfSize =
		sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER)+palettesize + lineByte * height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	fileHead.bfOffBits = 54 + palettesize;

	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

	// 填写文件头
	BITMAPINFOHEADER head;
	head.biBitCount = byte * 8;
	head.biHeight = height;
	head.biWidth = width;
	head.biCompression = 0;	// 0 表示不压缩
	head.biSizeImage = 0;	// 非压缩情况下可以为0
	head.biClrImportant = 0;
	head.biClrUsed = 0;
	head.biPlanes = 1;
	head.biSize = 40;
	head.biXPelsPerMeter = 0;
	head.biYPelsPerMeter = 0;
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

	// 颜色表拷贝
	if (palettesize == 1024)
	{
		unsigned char palette[1024];
		for (int i = 0; i < 256; i++){
			*(palette + i * 4 + 0) = i;
			*(palette + i * 4 + 1) = i;
			*(palette + i * 4 + 2) = i;
			*(palette + i * 4 + 3) = 0;
		}
		fwrite(palette, 1024, 1, fp);
	}

	// 扩展每行字节数，准备图像数据并保存
	unsigned char*buf = new unsigned char[height * lineByte];
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width*byte; j++){
			*(buf + i*lineByte + j) = *(imgBuf + (height - 1 - i)*width * byte + j);
		}
	}
	fwrite(buf, height*lineByte, 1, fp);

	fclose(fp);

	delete[]buf;		// 释放资源
	return 1;
}

// 【保存 | 读取 图像】BMP图像序列  输入为 unsigned char*  coordinateleft1.bin
// 保存为二进制文件
int BmpSerialize(CString bmpName, unsigned short *imgBuf, bool bRead, int width, int height, int PicCount)		// int iType
{
	if (bRead)
	{
		CFile loadF;
		if (FALSE == loadF.Open(bmpName, CFile::modeRead)){
			AfxMessageBox(_T("Serialize pic open error"));
			return 0;
		}

		CArchive ar(&loadF, CArchive::load);
		ar >> width >> height >> PicCount;
		for (int i = 0; i < width; i++){
			for (int j = 0; j < height; j++){
				ar >> *(imgBuf + j*width + i);
			}
		}
		ar.Close();
		loadF.Close();
	}
	else {
		CFile saveF;
		if (FALSE == saveF.Open(bmpName, CFile::modeCreate | CFile::modeWrite))
		{
			AfxMessageBox(_T("Serialize pic save error"));
			return 0;
		}
		CArchive ar(&saveF, CArchive::store);
		ar << width << height << PicCount;
		for (int i = 0; i < width; i++){
			for (int j = 0; j < height; j++){
				ar << *(imgBuf + j*width + i);
			}
		}
		ar.Close();
		saveF.Close();
	}
	return PicCount;
}

// 【保存图像】BMP图像序列 输入为 float*         coordinateleft1.bin
// Mat保存为二进制文件
bool BmpSerialize(string fileName, Mat data, bool bRead)
{
	if (bRead)
	{
		CFile loadF;
		if (FALSE == loadF.Open(string2CString(fileName), CFile::modeRead)){
			AfxMessageBox(_T("Serialize pic open error"));
			return false;
		}

		// 首先读取宽高信息 按列读取
		CArchive ar(&loadF, CArchive::load);
		ar >> data.cols >> data.rows;
		for (int i = 0; i < data.cols; i++){
			for (int j = 0; j < data.rows; j++){
				ar >> data.ptr<float>(0)[j*data.cols + i];//     *(imgBuf + j*data.cols + i);
			}
		}
		ar.Close();
		loadF.Close();
	}
	else {
		CFile saveF;
		if (FALSE == saveF.Open(string2CString(fileName), CFile::modeCreate | CFile::modeWrite))
		{
			AfxMessageBox(_T("Serialize pic save error"));
			return false;
		}
		CArchive ar(&saveF, CArchive::store);
		ar << data.cols << data.rows;
		// 首先存储宽高信息
		for (int i = 0; i < data.cols; i++){
			for (int j = 0; j < data.rows; j++){
				ar << data.ptr<float>(0)[j*data.cols + i]; //*(imgBuf + j*data.cols + i);
			}
		}
		ar.Close();
		saveF.Close();
	}
	return true;
}

// 【保存图像】BMP图像序列 输入为 float*         coordinateleft1.bin
// 保存为二进制文件
int BmpSerialize(CString bmpName, float *imgBuf, bool bRead, int width, int height, int PicCount)		// int iType
{
	if (bRead)
	{
		CFile loadF;
		if (FALSE == loadF.Open(bmpName, CFile::modeRead)){
			AfxMessageBox(_T("Serialize pic open error"));
			return 0;
		}

		CArchive ar(&loadF, CArchive::load);
		ar >> width >> height >> PicCount;
		for (int i = 0; i < width; i++){
			for (int j = 0; j < height; j++){
				ar >> *(imgBuf + j*width + i);
			}
		}
		ar.Close();
		loadF.Close();
	}
	else {
		CFile saveF;
		if (FALSE == saveF.Open(bmpName, CFile::modeCreate | CFile::modeWrite))
		{
			AfxMessageBox(_T("Serialize pic save error"));
			return 0;
		}
		CArchive ar(&saveF, CArchive::store);
		ar << width << height << PicCount;
		for (int i = 0; i < width; i++){
			for (int j = 0; j < height; j++){
				ar << *(imgBuf + j*width + i);
			}
		}
		ar.Close();
		saveF.Close();
	}
	return PicCount;
}

// 输出Mat的基本信息
void printMatInfo(Mat input)
{
	// 获取矩阵行列数
	std::cout << "Input row: " << input.rows << std::endl;
	std::cout << "Input col: " << input.cols << std::endl;

	cout << input.step.buf[0] << endl;
}

// 平移操作，图像大小不变
cv::Mat imageTranslation1(cv::Mat& srcImage, int xOffset, int yOffset){
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	cv::Mat resultImage(srcImage.size(), srcImage.type());

	if (srcImage.channels() == 3){
		// 遍历图像
		for (int i = 0; i < nRows; ++i){
			for (int j = 0; j < nCols; ++j){
				// 映射变换
				int x = j - xOffset;
				int y = i - yOffset;
				// 边界判断
				if (x >= 0 && y >= 0 && x < nCols && y < nRows)
					resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
			}
		}
	}
	else if (srcImage.channels() == 1){
		// 遍历图像
		for (int i = 0; i < nRows; ++i){
			for (int j = 0; j < nCols; ++j){
				// 映射变换
				int x = j - xOffset;
				int y = i - yOffset;
				// 边界判断
				if (x >= 0 && y >= 0 && x < nCols && y < nRows)
					resultImage.at<uchar>(i, j) = srcImage.ptr<uchar>(y)[x];
			}
		}
	}

	return resultImage;
}


// 平移操作，图像大小改变
cv::Mat imageTranslation2(cv::Mat &srcImage, int xOffset, int yOffset){
	// 设置平移尺寸
	int nRows = srcImage.rows + abs(yOffset);
	int nCols = srcImage.cols + abs(xOffset);

	cv::Mat resultImage(nRows, nCols, srcImage.type());

	// 图像遍历
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols; j++){
			// 映射变换
			int x = j - xOffset;
			int y = i - yOffset;
			// 边界判断
			if (x >= 0 && y >= 0 && x < nCols&&y < nRows){
				if (srcImage.channels() == 3){
					resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
				}
				else if (srcImage.channels() == 1){
					resultImage.at<uchar>(i, j) = srcImage.ptr<uchar>(y)[x];
				}
			}
		}
	}
	return resultImage;
}

// 基于等间隔提取图像缩放
cv::Mat imageReduction1(cv::Mat &srcImage, float kx, float ky){
	if (kx > 1.0f || ky > 1.0f){
		printf("【Error】imageReduction1函数出错！此函数无法执行放大操作\n");
		return Mat();
	}
	// 获取输出图像分辨率
	int nRows = cvRound(srcImage.rows*kx);
	int nCols = cvRound(srcImage.cols*ky);

	cv::Mat resultImage(nRows, nCols, srcImage.type());
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			// 根据水平因子计算坐标
			int x = static_cast<int>((i + 1) / kx + 0.5) - 1;
			// 根据垂直因子计算坐标
			int y = static_cast<int>((j + 1) / ky + 0.5) - 1;
			if (srcImage.channels() == 3){
				resultImage.at<cv::Vec3b>(i, j) = srcImage.at<cv::Vec3b>(x, y);
			}
			else if (srcImage.channels() == 1){
				resultImage.at<uchar>(i, j) = srcImage.at<uchar>(x, y);
			}
		}
	}
	return resultImage;
}

// 基于区域子块 像素灰度平均值
cv::Vec3b areaAverage(const cv::Mat& srcImage, Point_<int> leftPoint, Point_<int> rightPoint){
	int temp1 = 0, temp2 = 0, temp3 = 0;
	// 计算区域子块像素点个数
	int nPix = (rightPoint.x - leftPoint.x + 1)*(rightPoint.y - leftPoint.y + 1);
	// 对区域子块各个通道对像素值求和
	for (int i = leftPoint.x; i <= rightPoint.x; i++){
		for (int j = leftPoint.y; j <= rightPoint.y; j++){
			if (srcImage.channels() == 3){
				temp1 += srcImage.at<cv::Vec3b>(i, j)[0];
				temp2 += srcImage.at<cv::Vec3b>(i, j)[1];
				temp3 += srcImage.at<cv::Vec3b>(i, j)[2];
			}
			else if (srcImage.channels() == 1){
				temp1 += srcImage.at<uchar>(i, j);
			}
		}
	}
	// 对每个通道求均值
	Vec3b vecTemp;
	vecTemp[0] = temp1 / nPix;
	vecTemp[1] = temp2 / nPix;
	vecTemp[2] = temp3 / nPix;
	return vecTemp;
}


// 基于区域子块提取图像缩放
// 区域子块提取图像缩放是通过对源图像进行区域子块划分，然后提取子块中像素值作为采样像素以构成新图像来实现的。
cv::Mat imageReduction2(const Mat& srcImage, double kx, double ky){
	// 获取输出图像分辨率
	int nRows = cvRound(srcImage.rows*kx);
	int nCols = cvRound(srcImage.cols*ky);

	cv::Mat resultImage(nRows, nCols, srcImage.type());

	// 区域子块的左上角行列坐标
	int leftRowCoordinate = 0;
	int leftColCoordinate = 0;

	for (int i = 0; i < nRows; ++i){
		// 根据水平因子计算坐标
		int x = static_cast<int>((i + 1) / kx + 0.5) - 1;
		for (int j = 0; j < nCols; ++j){
			// 根据垂直因子计算坐标
			int y = static_cast<int>((j + 1) / ky + 0.5) - 1;

			Vec3b tempV = areaAverage(srcImage,
				Point_<int>(leftRowCoordinate, leftColCoordinate),
				Point_<int>(x, y));

			// 求解区域子块的均值
			if (srcImage.channels() == 3){
				resultImage.at<Vec3b>(i, j) = tempV;
			}
			else if (srcImage.channels() == 1){
				resultImage.at<uchar>(i, j) = tempV[0];
			}
			// 更新下子块左上角的列坐标，行坐标不变
			leftColCoordinate = y + 1;
		}
		leftColCoordinate = 0;
		// 更新下子块左上角的行坐标
		leftRowCoordinate = x + 1;
	}
	return resultImage;
}

// 获得仿射变换图像
cv::Mat getAffineTransformImage(cv::Mat srcImage, const Point2f srcPts[], const Point2f dstPts[]){
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	// 定义仿射变换矩阵2*3
	cv::Mat warpMat(cv::Size(2, 3), CV_32F);
	cv::Mat resultImage = cv::Mat::zeros(nRows, nCols, srcImage.type());
	// 计算仿射变换矩阵，即仿射变换的2*3数组
	warpMat = cv::getAffineTransform(srcPts, dstPts);
	// 根据仿射矩阵计算图像仿射变换
	cv::warpAffine(srcImage, resultImage, warpMat, resultImage.size());
	return resultImage;
}

// 获得斜切图像
// 左斜为正  右斜为负 角度制
cv::Mat getSkewImage(cv::Mat srcImage, float angle){
	// 角度转换
	float alpha = (float)fabs(angle * CV_PI / 180);

	int nRows = srcImage.rows;
	int nCols = (int)(srcImage.rows * tan(alpha) + srcImage.cols);

	// 定义仿射变换矩阵 斜切
	Mat warpMat = (Mat_<float>(2, 3) << 1, tan(alpha), 0, 0, 1, 0);
	cv::Mat resultImage = cv::Mat::zeros(nRows, nCols, srcImage.type());
	// 根据仿射矩阵计算图像仿射变换
	cv::warpAffine(srcImage, resultImage, warpMat, resultImage.size());

	if (angle < 0)
		return getFlipImage(resultImage, FLIP_HORIZONTAL);
	else
		return resultImage;
}

// 视频质量评价
// 计算PSNR峰值信噪比，返回数值为30~50dB,值越大越好
double PSNR(const Mat& I1, const Mat& I2){
	cv::Mat s1;
	// 计算图像差|I1 - I2|
	absdiff(I1, I2, s1);
	// 转成32浮点数进行平方运算
	s1.convertTo(s1, CV_32F);
	// s1*s1, 即|I1 - I2|^2
	s1 = s1.mul(s1);
	// 分别叠加每个通道的元素，存与s中
	cv::Scalar s = sum(s1);
	// 计算所有通道元素和
	double sse = s.val[0] + s.val[1] + s.val[2];
	cout << "sse = " << sse << endl;
	// 当元素很小时返回0值
	if (sse <= 1e-10)
		return 0;
	else {
		// 根据公式计算当前I1与I2的均方误差
		double mse = sse / (double)(I1.channels() * I1.total());
		// 计算峰值信噪比
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

// 计算MSSIM结构相似性，返回值从0到1，值越大越好
cv::Scalar MSSIM(const Mat& i1, const Mat& i2){
	const double C1 = 6.5025, C2 = 58.5225;
	cv::Mat I1, I2;
	// 转换成32浮点数进行平方运算
	i1.convertTo(I1, CV_32F);
	i2.convertTo(I2, CV_32F);
	// I2^2
	cv::Mat I2_2 = I2.mul(I2);
	cv::Mat I1_2 = I1.mul(I1);
	cv::Mat I1_I2 = I1.mul(I2);

	cv::Mat mu1, mu2;
	// 高斯加权计算每一窗口的均值、方差以及协方差
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	// 高斯平滑
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	// 根据公式计算相应参数
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	// t3 = ((2*mu1_mu2 +C1).*(2*sigma12+C2))
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	// t1 = ((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
	t1 = t1.mul(t2);
	Mat ssim_map;
	// ssim_map  = t3./t1;
	divide(t3, t1, ssim_map);
	// 将平均值作为两图像的结构相似性度量
	cv::Scalar mssim = mean(ssim_map);
	return mssim;
}

// MatIterator_ 迭代器反色处理
cv::Mat inverseColor4(cv::Mat srcImage){
	cv::Mat tempImage = srcImage.clone();
	if (srcImage.channels() == 3){
		// 初始化源图像迭代器
		cv::MatConstIterator_<cv::Vec3b> srcIterStart = srcImage.begin<cv::Vec3b>();
		cv::MatConstIterator_<cv::Vec3b> srcIterEnd = srcImage.end<cv::Vec3b>();
		// 初始化输出图像迭代器
		cv::MatIterator_<cv::Vec3b> resIterStart = tempImage.begin<cv::Vec3b>();
		cv::MatIterator_<cv::Vec3b> resIerEnd = tempImage.end<cv::Vec3b>();

		// 遍历图像反色处理
		while (srcIterStart != srcIterEnd){
			(*resIterStart)[0] = 255 - (*srcIterStart)[0];
			(*resIterStart)[1] = 255 - (*srcIterStart)[1];
			(*resIterStart)[2] = 255 - (*srcIterStart)[2];
			// 迭代器增加
			srcIterStart++;
			resIterStart++;
		}
	}
	else if (srcImage.channels() == 1){
		// 初始化源图像迭代器
		cv::MatConstIterator_<uchar> srcIterStart = srcImage.begin<uchar>();
		cv::MatConstIterator_<uchar> srcIterEnd = srcImage.end<uchar>();
		// 初始化输出图像迭代器
		cv::MatIterator_<uchar> resIterStart = tempImage.begin<uchar>();
		cv::MatIterator_<uchar> resIerEnd = tempImage.end<uchar>();

		// 遍历图像反色处理
		while (srcIterStart != srcIterEnd){
			(*resIterStart) = 255 - (*srcIterStart);
			// 迭代器增加
			srcIterStart++;
			resIterStart++;
		}
	}
	return tempImage;
}

// isContinuous 反色处理
cv::Mat inverseColor5(cv::Mat srcImage){
	int row = srcImage.rows;
	int col = srcImage.cols;
	Mat tempImage = srcImage.clone();
	// 判断图像是否是连续图像，即是否有像素填充
	if (srcImage.isContinuous() && tempImage.isContinuous()){
		row = 1;
		// 按照行展开
		col = col * srcImage.rows * srcImage.channels();
	}
	// 遍历图像的每个像素
	for (int i = 0; i < row; i++){
		// 设定图像数据源指针及输出图像数据指针
		const uchar* pSrcData = srcImage.ptr<uchar>(i);
		uchar* pResultData = tempImage.ptr<uchar>(i);
		for (int j = 0; j < col; j++){
			*pResultData++ = 255 - *pSrcData++;
		}
	}
	return tempImage;
}

// 代码2-28 LUT 查表反色处理
cv::Mat inverseColor6(cv::Mat srcImage){
	int row = srcImage.rows;
	int col = srcImage.cols;
	cv::Mat tempImage = srcImage.clone();
	// 建立LUT反色table
	uchar LutTable[256];
	for (int i = 0; i < 256; i++){
		LutTable[i] = 255 - i;
	}
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar *pData = lookUpTable.data;
	// 建立映射表
	for (int i = 0; i < 256; i++){
		pData[i] = LutTable[i];
	}
	// 应用索引表进行查找
	cv::LUT(srcImage, lookUpTable, tempImage);
	return tempImage;
}

// 单窗口显示多幅图像
void showManyImages(const std::vector<cv::Mat> &srcImages, cv::Size imgSize){
	int nNumImages = srcImages.size();
	//cout << "nNumImage  = " << nNumImages << std::endl;
	cv::Size nSizeWindows;
	if (nNumImages > 12){
		std::cout << "Not more than 12 images!" << std::endl;
		return;
	}
	// 根据图片序列数量来确定分割小窗口的形态
	switch (nNumImages){
	case 1: nSizeWindows = cv::Size(1, 1); break;
	case 2: nSizeWindows = cv::Size(2, 1); break;
	case 3:
	case 4: nSizeWindows = cv::Size(2, 2); break;
	case 5:
	case 6: nSizeWindows = cv::Size(3, 2); break;
	case 7:
	case 8: nSizeWindows = cv::Size(4, 2); break;
	case 9: nSizeWindows = cv::Size(3, 3); break;
	default:nSizeWindows = cv::Size(4, 3); break;
	}
	// 设置小图像尺寸、间隙、边界
	Size nShowImageSize = imgSize;
	int nSplitLineSize = 15;
	int nAroundLineSize = 50;
	// 创建输出图像，图像大小根据输入源来确定
	const int imagesHeight = nShowImageSize.height *nSizeWindows.height + nAroundLineSize + (nSizeWindows.height - 1)*nSplitLineSize;
	const int imagesWidth = nShowImageSize.width*nSizeWindows.width + nAroundLineSize + (nSizeWindows.width - 1)*nSplitLineSize;
	//std::cout << imagesWidth << " " << imagesHeight << std::endl;

	cv::Mat showWindowImages(imagesHeight, imagesWidth, CV_8UC3, cv::Scalar::all(0));
	// 提取对应小图像的左上角坐标X、Y
	int posX = (showWindowImages.cols - (nShowImageSize.width*nSizeWindows.width + (nSizeWindows.width - 1)*nSplitLineSize)) / 2;
	int posY = (showWindowImages.rows - (nShowImageSize.height*nSizeWindows.height + (nSizeWindows.height - 1)*nSplitLineSize)) / 2;

	//std::cout << posX << " " << posY << std::endl;
	int tempPosX = posX;
	int tempPosY = posY;
	// 将每一小幅图像整合成大图像
	for (int i = 0; i < nNumImages; i++){
		cout << srcImages[i].size() << endl;
		// 小图像坐标转换
		if ((i%nSizeWindows.width == 0) && (tempPosX != posX)){
			tempPosX = posX;
			tempPosY += (nSplitLineSize + nShowImageSize.height);
		}
		//printf("tempPosX = %d tempPosY = %d\n", tempPosX, tempPosY);
		putText(showWindowImages, "PIC " + int2string(i + 1), Point2d(tempPosX + 120, tempPosY + nShowImageSize.height + 12), CV_FONT_ITALIC, 0.5, MC_YELLOW, 1);
		// 利用Rect区域将小图像置于大图像的相应区域
		cv::Mat tempImage = showWindowImages(cv::Rect(tempPosX, tempPosY, nShowImageSize.width, nShowImageSize.height));
		// 利用resize函数实现图像缩放
		Mat tmp = convert2BGR(srcImages[i]);
		resize(tmp, tempImage, nShowImageSize);
		tempPosX += (nSplitLineSize + nShowImageSize.width);
	}
	cv::imshow("showWindowImages", showWindowImages);
}

// 获取HSV图像
cv::Mat getHSVImage(const Mat& image, Mat& image_H, Mat& image_S, Mat& image_V){
	cv::Mat image_hsv;
	cvtColor(image, image_hsv, CV_BGR2HSV);

	// 分离HSV各个通道
	std::vector<cv::Mat> hsvChannels;
	cv::split(image_hsv, hsvChannels);
	// 0 通道为H分量, 1通道为S分量， 2通道为V分量
	image_H = hsvChannels[0];
	image_S = hsvChannels[1];
	image_V = hsvChannels[2];
	return image_hsv;
}

// 自适应阈值化
cv::Mat getAdaptiveThresholdImage(const Mat& image, double maxValue, int blockSize, double C,
	int adaptiveMethod, int thresholdType){
	Mat input = GrayTrans(image);
	Mat dstImage;
	cv::adaptiveThreshold(input, dstImage, maxValue, adaptiveMethod, thresholdType, blockSize, C);
	return dstImage;
}

// 双阈值化
cv::Mat getDoubleThreshImage(const Mat& image, double lowthresh, double highthresh, double maxValue){
	if (!image.data)
	{
		printf("MyImage.cpp getDoubleThreshImage: 输入图像数据为空!\n");
		return Mat();
	}
	Mat srcGray = GrayTrans(image);
	Mat dstTempImage1, dstTempImage2, dstImage;
	// 小阈值对源灰度图像进行阈值化操作
	cv::threshold(srcGray, dstTempImage1, lowthresh, maxValue, cv::THRESH_BINARY);
	// 大阈值对源灰度图像进行阈值化操作
	cv::threshold(srcGray, dstTempImage2, highthresh, maxValue, cv::THRESH_BINARY_INV);
	// 矩阵与运算得到二值化结果
	cv::bitwise_and(dstTempImage1, dstTempImage2, dstImage);
	return dstImage;
}

// 半阈值化
cv::Mat getHalfThreshImage(const Mat& image, double thresh){
	if (!image.data)
	{
		printf("MyImage.cpp getHalfThreshImage: 输入图像数据为空!\n");
		return Mat();
	}
	Mat srcGray = GrayTrans(image);
	Mat dstTempImage, dstImage;
	// 阈值对源灰度图进行阈值化操作
	cv::threshold(srcGray, dstTempImage, thresh, 255, cv::THRESH_BINARY);
	// 矩阵与运算得到二值化结果
	cv::bitwise_and(srcGray, dstTempImage, dstImage);
	return dstImage;
}

// 直方图均衡化
cv::Mat getEqualHistImage(const Mat& image, bool useRGB){
	if (!image.data)
	{
		printf("MyImage.cpp getEqualHistImage: 输入图像数据为空!\n");
		return Mat();
	}
	if (useRGB && image.channels() == 3){
		Mat colorHeqImage;
		std::vector<cv::Mat> BGR_plane;
		// 对BGR通道进行分离
		cv::split(image, BGR_plane);
		// 分别对BGR进行直方图均衡化
		for (int i = 0; i < (int)BGR_plane.size(); i++){
			cv::equalizeHist(BGR_plane[i], BGR_plane[i]);
		}
		// 合并对应的各个通道
		cv::merge(BGR_plane, colorHeqImage);
		return colorHeqImage;
	}
	else{
		Mat srcGray = GrayTrans(image);
		Mat heqResult;
		equalizeHist(srcGray, heqResult);
		return heqResult;
	}
}

// 直方图变换——查找
cv::Mat getHistogramTransLUT(const Mat& srcImage, int segThreshold){
	// 第一步： 计算图像的直方图
	Mat srcGray = GrayTrans(srcImage);
	MatND hist;
	getHistgram(srcImage, hist);

	// 第二步，根据预设参数统计灰度级变换
	// 由低到高进行查找
	int iLow = 0;
	for (; iLow < 256; iLow++){
		if (hist.at<float>(iLow) > segThreshold){
			break;
		}
	}
	// 由高到低进行查找
	int iHigh = 255;
	for (; iHigh >= 0; iHigh--){
		if (hist.at<float>(iHigh) > segThreshold){
			break;
		}
	}
	// 第三步：建立查找表
	cv::Mat lookUpTable(cv::Size(1, 256), CV_8U);
	for (int i = 0; i < 256; i++){
		if (i < iLow){
			lookUpTable.at<uchar>(i) = 0;
		}
		else if (i > iHigh){
			lookUpTable.at<uchar>(i) = 255;
		}
		else {
			lookUpTable.at<uchar>(i) = static_cast<uchar>(255.0 * (i - iLow) / (iHigh - iLow + 0.5));
		}
	}
	// 第四步: 通过查找表进行映射变换
	cv::Mat histTransResult;
	cv::LUT(srcGray, lookUpTable, histTransResult);

	return histTransResult;
}

// 直方图变换——累计
// 直方图变换累计方法实现的思路；
// （1） 将源图像转换为灰度图，计算图像的灰度直方图
// （2） 建立映射表，对直方图进行像素累积
// （3） 根据映射表进行元素映射得到最终的直方图变换
cv::Mat getHistogramTransAggregate(const Mat& srcImage){
	Mat srcGray = GrayTrans(srcImage);
	MatND hist;
	getHistgram(srcImage, hist);
	float table[256];
	int nPix = srcGray.cols * srcGray.rows;
	// 建立映射表
	for (int i = 0; i < 256; i++){
		float temp[256];
		// 像素变换
		temp[i] = hist.at<float>(i) / nPix * 255;
		if (i != 0){
			// 像素累计
			table[i] = table[i - 1] + temp[i];
		}
		else {
			table[i] = temp[i];
		}
	}

	// 通过映射进行表查找
	cv::Mat lookUpTable(cv::Size(1, 256), CV_8U);
	for (int i = 0; i < 256; i++){
		lookUpTable.at<uchar>(i) = static_cast<uchar>(table[i]);
	}

	cv::Mat histTransResult;
	cv::LUT(srcGray, lookUpTable, histTransResult);
	return histTransResult;
}

// 直方图匹配
// (1) 分别计算源图像与目标图像的累计概率分布
// (2) 分别对源图像与目标图像进行直方图均衡化操作
// (3) 利用组映射关系使源图像直方图按照规定进行变换
cv::Mat getHistgramMatchImage(const Mat& srcImage, Mat target){
	if (!srcImage.data || !target.data){
		printf("MyImage.cpp getHistgramMatchImage 输入图像为空!\n");
		return Mat();
	}
	resize(target, target, srcImage.size(), 0, 0, CV_INTER_LINEAR);

	// 初始化累计分布参数
	float srcCdfArr[256];
	float dstCdfArr[256];
	int srcAddTemp[256];
	int dstAddTemp[256];
	int histMatchMap[256];
	for (int i = 0; i < 256; i++){
		srcAddTemp[i] = 0;
		dstAddTemp[i] = 0;
		srcCdfArr[i] = 0;
		dstCdfArr[i] = 0;
		histMatchMap[i] = 0;
	}
	float sumSrcTemp = 0;
	float sumDstTemp = 0;
	int nSrcPix = srcImage.cols * srcImage.rows;
	int nDstPix = target.cols * target.rows;
	int matchFlag = 0;
	// 求解源图像与目标图像的累计直方图
	for (int nrow = 0; nrow < srcImage.rows; nrow++){
		for (int ncol = 0; ncol < srcImage.cols; ncol++){
			srcAddTemp[(int)srcImage.at<uchar>(nrow, ncol)]++;
			dstAddTemp[(int)target.at<uchar>(nrow, ncol)]++;
		}
	}
	// 求解源图像与目标图像的累计概率分布
	for (int i = 0; i < 256; i++){
		sumSrcTemp += srcAddTemp[i];
		srcCdfArr[i] = sumSrcTemp / nSrcPix;
		sumDstTemp += dstAddTemp[i];
		dstCdfArr[i] = sumDstTemp / nDstPix;
	}

	// 直方图匹配实现
	for (int i = 0; i < 256; i++){
		float minMatchPara = 20;
		for (int j = 0; j < 256; j++){
			// 判断当前直方图累计差异
			if (minMatchPara > abs(srcCdfArr[i] - dstCdfArr[j])){
				minMatchPara = abs(srcCdfArr[i] - dstCdfArr[j]);
				matchFlag = j;
			}
		}
		histMatchMap[i] = matchFlag;
	}

	// 初始化匹配图像
	cv::Mat HistMatchImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
	cv::cvtColor(srcImage, HistMatchImage, CV_BGR2GRAY);
	// 通过map映射成匹配图像
	for (int i = 0; i < HistMatchImage.rows; i++){
		for (int j = 0; j < HistMatchImage.cols; j++){
			HistMatchImage.at<uchar>(i, j) = histMatchMap[(int)HistMatchImage.at<uchar>(i, j)];
		}
	}
	return HistMatchImage;
}

// 距离变换
/*
	根据Rosenfeld and Pfaltz提出的距离变换理论，对于二值图像（前景目标为1，背景为0），距离变换实现图像的每个像素
	到最近前景目标或到图像边界的距离。距离变换的步骤如下：
	（1） 将图像输入转为二值图像，前景目标为1，背景为0.
	（2） 第一遍水平扫描从左上角开始，依次从左往右扫描，扫描完一行自动转到下一行的最左端继续扫描，按行遍历图像。
	掩膜模板mask为maskL，应用下面的公式进行计算
	f(p)  = min[f(p), D(p,q) + f(q)]   q属于maskL
	其中D为距离，包含欧式距离、棋盘距离或街区距离，f(p)为像素点p的像素值
	（3） 第二遍水平扫描从右下角开始，依次从右往左逐行扫描，扫描完一行自行转到上一行的最右端继续扫描，按行遍历图像，
	掩膜模板mask为maskR，方法同步骤（2）。
	（4） 根据模板maskL和maskR的扫描结果得到最终的距离变换图像
	*/
cv::Mat getDistTransImage(Mat& srcImage, int thresh){
	CV_Assert(srcImage.data != NULL);
	cv::Mat srcGray = GrayTrans(srcImage);
	cv::Mat srcBinary;
	// 转换成二值图像
	threshold(srcGray, srcBinary, thresh, 255, cv::THRESH_BINARY);

	imshow("binary", srcBinary);

	int rows = srcBinary.rows;
	int cols = srcBinary.cols;
	uchar* pDataOne;
	uchar* pDataTwo;
	float disPara = 0;
	float fDisMin = 0;
	// 第一遍遍历图像，用左模板更新像素值
	for (int i = 1; i < rows - 1; i++){
		// 图像指针获取
		pDataOne = srcBinary.ptr<uchar>(i);
		for (int j = 1; j < cols; j++){

			//	printf("(%d , %d)\n", i, j);

			// 分别计算其左模板掩码的相关距离
			// pL pL
			// pL p
			// pL
			pDataTwo = srcBinary.ptr<uchar>(i - 1);
			disPara = calcEuclideanDistance(i, j, i - 1, j - 1);
			fDisMin = min((float)pDataOne[j], disPara + pDataTwo[j - 1]);
			//printf("fDisMin = %f\n", fDisMin);

			disPara = calcEuclideanDistance(i, j, i - 1, j);
			fDisMin = min(fDisMin, disPara + pDataTwo[j]);
			//printf("fDisMin = %f\n", fDisMin);

			pDataTwo = srcBinary.ptr<uchar>(i);
			disPara = calcEuclideanDistance(i, j, i, j - 1);
			fDisMin = min(fDisMin, disPara + pDataTwo[j - 1]);
			//printf("fDisMin = %f\n", fDisMin);

			pDataTwo = srcBinary.ptr<uchar>(i + 1);
			disPara = calcEuclideanDistance(i, j, i + 1, j - 1);
			fDisMin = min(fDisMin, disPara + pDataTwo[j - 1]);
			//printf("fDisMin = %f\n", fDisMin);

			pDataOne[j] = (uchar)cvRound(fDisMin);
			//cout << endl << endl;
		}
	}

	// 第二遍遍历图像，用右模板更新像素值
	for (int i = rows - 2; i > 0; i--){
		// 图像指针获取
		pDataOne = srcBinary.ptr<uchar>(i);
		for (int j = cols - 1; j >= 0; j--){
			// 分别计算其左模板掩码的相关距离
			// pR pR
			// pR p
			// pR
			pDataTwo = srcBinary.ptr<uchar>(i + 1);
			disPara = calcEuclideanDistance(i, j, i + 1, j);
			fDisMin = min((float)pDataOne[j], disPara + pDataTwo[j]);
			disPara = calcEuclideanDistance(i, j, i + 1, j + 1);
			fDisMin = min(fDisMin, disPara + pDataTwo[j + 1]);

			pDataTwo = srcBinary.ptr<uchar>(i);
			disPara = calcEuclideanDistance(i, j, i, j + 1);
			fDisMin = min(fDisMin, disPara + pDataTwo[j + 1]);

			pDataTwo = srcBinary.ptr<uchar>(i - 1);
			disPara = calcEuclideanDistance(i, j, i - 1, j + 1);
			fDisMin = min(fDisMin, disPara + pDataTwo[j + 1]);
			pDataOne[j] = (uchar)cvRound(fDisMin);
		}
	}
	//double x, y;
	//minMaxLoc(srcBinary, &x, &y);
	//cout << "min = " << x << endl;
	//cout << "max = " << y << endl;
	return srcBinary;
}

// 采用opencv自带的距离变换函数
cv::Mat getDistTransImage2(Mat &srcImage, int thresh){
	if (!srcImage.data){
		return Mat();
	}
	// 转换为灰度图像
	cv::Mat srcGray = GrayTrans(srcImage);
	// 转换为二值图像
	cv::Mat srcBinary;
	threshold(srcGray, srcBinary, thresh, 255, cv::THRESH_BINARY);
	imshow("binary", srcBinary);
	// 距离变换
	cv::Mat dstImage;
	cv::distanceTransform(srcBinary, dstImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	// 归一化矩阵
	cv::normalize(dstImage, dstImage, 0, 1., cv::NORM_MINMAX);
	return dstImage;
}

// Gamma校正 伽马校正
// 一般情况下，当Gamma矫正的值大于1时，图像的高光部分被压缩而暗调部分被扩展
// 当Gamma矫正值小于1时，图像的高光部分被扩展而暗调部分被压缩。
cv::Mat getGammaTransformImage(cv::Mat& srcImage, float kFactor){
	// 建立查表LUT
	unsigned char LUT[256];
	for (int i = 0; i < 256; i++){
		// Gamma变换表达式
		LUT[i] = saturate_cast<uchar>(pow((float)(i / 255.0), kFactor)*255.0f);
	}

	cv::Mat resultImage = srcImage.clone();
	// 输出通道为单通道时，直接进行变换
	if (srcImage.channels() == 1){
		cv::MatIterator_<uchar> iterator = resultImage.begin<uchar>();
		cv::MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
		for (; iterator != iteratorEnd; iterator++)
			*iterator = LUT[(*iterator)];
	}
	else {
		// 输入通道为3通道时，需对每个通道分别进行变换
		cv::MatIterator_<cv::Vec3b> iterator = resultImage.begin<Vec3b>();
		cv::MatIterator_<cv::Vec3b> iteratorEnd = resultImage.end<Vec3b>();
		// 通过查找表进行变换
		for (; iterator != iteratorEnd; iterator++){
			(*iterator)[0] = LUT[((*iterator)[0])];
			(*iterator)[1] = LUT[((*iterator)[1])];
			(*iterator)[2] = LUT[((*iterator)[2])];
		}
	}
	return resultImage;
}

// 图像线性变换操作
cv::Mat getLinearTransformImage(cv::Mat& srcImage, float a, int b){
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	// 图像元素遍历
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols; j++){
			if (srcImage.channels() == 3){
				for (int c = 0; c < 3; c++){
					// 矩阵at操作，检查下标防止越界
					resultImage.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(a*(srcImage.at<cv::Vec3b>(i, j)[c]) + b);
				}
			}
			else {
				resultImage.at<uchar>(i, j) = saturate_cast<uchar>(a*srcImage.at<uchar>(i, j) + b);
			}
		}
	}
	return resultImage;
}

// 图像对数变换方法1
// 图像对数变换是将图像输入中范围较窄的低灰度值映射成输出中较宽范围的灰度值，
// 常用于扩展图像中被压缩的（灰度值较高区域的）低像素值。
cv::Mat getLogTransform1(cv::Mat srcImage, float c){
	// 输入图像判断
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	// 计算1+r
	cv::add(srcImage, cv::Scalar(1.0), srcImage);
	// 转换为32位浮点数
	srcImage.convertTo(srcImage, CV_32F);
	// 计算log(1+r)
	log(srcImage, resultImage);
	resultImage = c * resultImage;
	// 归一化处理
	cv::normalize(resultImage, resultImage, 0, 255, NORM_MINMAX);
	cv::convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

// 图像对数变换方法2 不太好使 这个方法没有先转CV_32F
// 图像对数变换是将图像输入中范围较窄的低灰度值映射成输出中较宽范围的灰度值，
// 常用于扩展图像中被压缩的（灰度值较高区域的）低像素值。
cv::Mat getLogTransform2(cv::Mat srcImage, float c){
	// 输入图像判断
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	double gray = 0;
	// 图像遍历,分别计算每个像素点的对数变换
	for (int i = 0; i < srcImage.rows; i++){
		for (int j = 0; j < srcImage.cols; j++){
			if (srcImage.channels() == 1){
				gray = (double)srcImage.at<uchar>(i, j);
				gray = c*log((double)(1 + gray));
				resultImage.at<uchar>(i, j) = saturate_cast<uchar>(gray);
			}
			else {
				double temp[3];
				temp[0] = (double)srcImage.at<cv::Vec3b>(i, j)[0];
				temp[1] = (double)srcImage.at<cv::Vec3b>(i, j)[1];
				temp[2] = (double)srcImage.at<cv::Vec3b>(i, j)[2];
				temp[0] = c*log((double)(1 + temp[0]));
				temp[1] = c*log((double)(1 + temp[1]));
				temp[2] = c*log((double)(1 + temp[2]));
				resultImage.at<cv::Vec3b>(i, j)[0] = saturate_cast<uchar>(temp[0]);
				resultImage.at<cv::Vec3b>(i, j)[1] = saturate_cast<uchar>(temp[1]);
				resultImage.at<cv::Vec3b>(i, j)[2] = saturate_cast<uchar>(temp[2]);
			}
		}
	}
	// 归一化处理
	cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

// 图像对数变换方法3
// 图像对数变换是将图像输入中范围较窄的低灰度值映射成输出中较宽范围的灰度值，
// 常用于扩展图像中被压缩的（灰度值较高区域的）低像素值。
cv::Mat getLogTransform3(cv::Mat srcImage, float c){
	// 输入图像判断
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	// 图像类型转换
	srcImage.convertTo(resultImage, CV_32F);
	// 图像矩阵元素加1操作
	resultImage = resultImage + 1;
	// 图像对数操作
	cv::log(resultImage, resultImage);
	resultImage = c*resultImage;
	// 归一化处理
	cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

// 对比度拉伸操作
cv::Mat getContrastStretchImage(cv::Mat srcImage){
	cv::Mat resultImage = srcImage.clone();
	if (srcImage.channels() == 1){
		int nRows = resultImage.rows;
		int nCols = resultImage.cols;
		// 图像连续性判断
		if (resultImage.isContinuous()){
			nCols = nCols * nRows;
			nRows = 1;
		}
		// 图像指针操作
		uchar* pDataMat;
		double pixMin = 0, pixMax = 255;
		minMaxLoc(resultImage, &pixMax, &pixMax);
		// 对比度拉伸映射
		for (int j = 0; j < nRows; j++){
			pDataMat = resultImage.ptr<uchar>(j);
			for (int i = 0; i < nCols; i++){
				pDataMat[i] = (uchar)((pDataMat[i] - pixMin) * 255 / (pixMax - pixMin));
			}
		}
		return resultImage;
	}
	else {
		vector<Mat> layer;
		vector<Mat> r;
		r.resize(3);
		split(srcImage, layer);
		for (int i = 0; i < 3; i++){
			r[i] = getContrastStretchImage(layer[i]);
		}
		Mat resultImage;
		merge(r, resultImage);
		return resultImage;
	}
}

// 灰度级分层
// 将待提取的感兴趣区域的灰度值映射变大或变小，其他不感兴趣的灰度值保持原有值不变，最终输出图像仍为灰度图像
cv::Mat getGrayLayeredImage(cv::Mat srcImage, int controlMin, int controlMax){
	cv::Mat resultImage = GrayTrans(srcImage);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	// 图像连续性判断
	if (resultImage.isContinuous()){
		nCols = nCols * nRows;
		nRows = 1;
	}
	// 图像指针操作
	uchar *pDataMat;
	// 计算图像的灰度级分层
	for (int j = 0; j < nRows; j++){
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++){
			// 区域映射
			if (pDataMat[i] > controlMin && pDataMat[i] < controlMax)
				pDataMat[i] = controlMax;
		}
	}
	return resultImage;
}

// 获得灰度比特平面序列
std::vector<cv::Mat> getMBitPlans(cv::Mat srcImage){
	// 若非灰度图则转换为灰度图
	Mat srcGray = GrayTrans(srcImage);
	int nRows = srcGray.rows;
	int nCols = srcGray.cols;
	// 图像连续性判断
	if (srcGray.isContinuous()){
		nCols = nCols * nRows;
		nRows = 1;
	}
	// 图像指针操作
	uchar *pSrcMat;
	uchar *pResultMat;
	cv::Mat resultImage = srcGray.clone();
	std::vector<cv::Mat> bitPlanes;
	bitPlanes.resize(8);
	int pixMax = 0, pixMin = 0;
	for (int n = 1; n <= 8; n++){
		// 比特平面分层像素构成
		pixMin = (int)pow(2.0, n - 1);
		pixMax = (int)pow(2.0, n);
		for (int j = 0; j < nRows; j++){
			// 获取图像数据指针
			pSrcMat = srcGray.ptr<uchar>(j);
			pResultMat = resultImage.ptr<uchar>(j);
			for (int i = 0; i < nCols; i++){
				//printf("pSrcMat(%d) = %d\n", i, pSrcMat[i]);
				// 相应比特平面层二值化
				if (pSrcMat[i] >= pixMin && pSrcMat[i] < pixMax)
					pResultMat[i] = 255;
				else
					pResultMat[i] = 0;
			}
		}
		// 比特平面层输出
		//char windowsName[20];
		//sprintf(windowsName, "BitPlane %d", n);
		bitPlanes[n - 1] = resultImage.clone();
		//imshow(windowsName, resultImage);
	}
	return bitPlanes;
}

// 最大熵阈值分割
float calculateCurrentEntropy(cv::Mat hist, int threshold){
	float BackgroundSum = 0, targetSum = 0;
	const float* pDataHist = (float*)hist.ptr<float>(0);
	for (int i = 0; i < 256; i++){
		// 累计背景值
		if (i < threshold){
			BackgroundSum += pDataHist[i];
		}
		else {	// 累计目标值
			targetSum += pDataHist[i];
		}
	}
	// std::cout<< BackgroundSum <<" "<<targetSum<<std::endl;
	float BackgroundEntropy = 0, targetEntropy = 0;
	for (int i = 0; i < 256; i++){
		// 计算背景熵
		if (i < threshold){
			if (pDataHist[i] == 0)
				continue;
			float ratio1 = pDataHist[i] / BackgroundSum;
			// 计算当前能量熵
			BackgroundEntropy += -ratio1 * logf(ratio1);
		}
		else {	// 计算目标熵
			if (pDataHist[i] == 0)
				continue;
			float ratio2 = pDataHist[i] / targetSum;
			targetEntropy += -ratio2 * logf(ratio2);
		}
	}
	return (targetEntropy + BackgroundEntropy);
}

// 寻找最大熵阈值并分割
cv::Mat maxEntropySegMentation(cv::Mat inputImage){
	Mat inputGray = GrayTrans(inputImage);
	cv::MatND hist;
	getHistgram(inputGray, hist);
	float maxentropy = 0;
	int max_index = 0;
	cv::Mat result;
	// 遍历得到最大熵阈值分割的最佳阈值
	for (int i = 0; i < 256; i++){

		float cur_entropy = calculateCurrentEntropy(hist, i);
		// 计算当前最大值的位置
		if (cur_entropy > maxentropy){
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	printf("max_index=%d\n", max_index);
	// 二值化分割
	threshold(inputGray, result, max_index, 255, CV_THRESH_BINARY);
	return result;
}

// 计算图像波峰点
// 投影曲线的波峰/波谷是通过判定其一阶导数为零点，二阶导数为正或负值来确定的，即对于一阶差分D，
// 我们关注的是图像差分的值的大小，因此这里需要将其进行符号化，然后再通过计算二阶差分的变化，
// 找到曲线斜率点U的满足条件（由正到负或由负到正）,点集U正是投影曲线的波峰波谷值.
// 返回的图像为找到的图像波峰图像 resultVec记录了所有的波峰所在列数
cv::Mat findPeak(cv::Mat srcImage, vector<int>& resultVec, int thresh){
	cv::Mat verMat;
	cv::Mat resMat = srcImage.clone();
	// 阈值化操作
	//int threshType = 0;
	// 预设最大值
	const int maxVal = 255;
	// 固定阈值化操作
	cv::threshold(srcImage, srcImage, thresh, maxVal, CV_THRESH_BINARY);
	imshow("threshold", srcImage);

	srcImage.convertTo(srcImage, CV_32FC1);
	// 计算垂直投影
	cv::reduce(srcImage, verMat, 0, CV_REDUCE_SUM);
	//imshow("reduce", verMat);
	// std::cout<<verMat<<std::endl;
	// 遍历求差分符号函数
	float *iptr = ((float*)verMat.data) + 1;
	vector<int> tempVec(verMat.cols - 1, 0);
	for (int i = 1; i < verMat.cols - 1; ++i, ++iptr){
		if (*(iptr + 1) - *iptr >0)
			tempVec[i] = 1;
		else if (*(iptr + 1) - *iptr < 0)
			tempVec[i] = -1;
		else
			tempVec[i] = 0;
	}
	// 对符号函数进行遍历
	for (int i = tempVec.size() - 1; i >= 0; i--){
		if (tempVec[i] == 0 && i == tempVec.size() - 1){
			tempVec[i] = 1;
		}
		else if (tempVec[i] == 0){
			if (tempVec[i + 1] >= 0)
				tempVec[i] = 1;
			else
				tempVec[i] = -1;
		}
	}
	// 波峰判断输出
	for (vector<int>::size_type i = 0; i != tempVec.size() - 1; i++){
		if (tempVec[i + 1] - tempVec[i] == -2)
			resultVec.push_back(i + 1);
	}
	// 输出波峰位置
	for (int i = 0; i < (int)resultVec.size(); i++){
		//std::cout << resultVec[i] << " ";
		// 波峰位置为255
		resMat.col(resultVec[i]).setTo(Scalar::all(255));
	}
	return resMat;
}

// 获得垂直投影图像 计算各列白点个数 reduceMat 存储的是计算结果 是CV_32F形式的
cv::Mat getVerticalProjImage(cv::Mat srcImage, Mat & reduceMat){
	// 输入图像判断
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	srcImage.convertTo(srcImage, CV_32FC1);
	cv::reduce(srcImage, reduceMat, 0, CV_REDUCE_SUM);

	int nRows = 150;

	double maxValue = 0, minValue = 0;
	minMaxLoc(reduceMat, &minValue, &maxValue);
	//printf("maxValue = %lf\n", maxValue);

	Mat result = Mat::zeros(nRows, srcImage.cols, CV_8UC3);
	//imshow("result", result);
	int hpt = saturate_cast<int>(0.9*nRows);
	for (int i = 0; i < srcImage.cols; i++){
		float binValue = *(reduceMat.ptr<float>(0) + i);
		if (binValue < 1.0f)
			continue;
		int realValue = saturate_cast<int>(binValue*hpt / maxValue);
		//printf("realValue = %d\n", realValue);
		rectangle(result, Point(i, 299 - 1), Point(i, nRows - realValue), MC_WHITE);
	}
	return result;
}

// 获得水平投影图像 计算各行白点个数 reduceMat 存储的是计算结果 是CV_32F形式的
cv::Mat getHorizontalProjImage(cv::Mat srcImage, Mat & reduceMat){
	// 输入图像判断
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	srcImage.convertTo(srcImage, CV_32FC1);
	cv::reduce(srcImage, reduceMat, 1, CV_REDUCE_SUM);

	double maxValue = 0, minValue = 0;
	minMaxLoc(reduceMat, &minValue, &maxValue);
	//printf("maxValue = %lf\n", maxValue);

	int nCols = 150;

	Mat result = Mat::zeros(srcImage.rows, nCols, CV_8UC3);
	//imshow("result", result);
	int hpt = saturate_cast<int>(0.9 * nCols);
	for (int i = 0; i < srcImage.rows; i++){
		float binValue = *(reduceMat.ptr<float>(0) + i);
		if (binValue < 1.0f)
			continue;
		int realValue = saturate_cast<int>(binValue*hpt / maxValue);
		//printf("realValue = %d\n", realValue);
		rectangle(result, Point(0, i), Point(realValue, i), MC_WHITE);
	}
	return result;
}

// 图像金字塔操作
// 上层图像是下层低通滤波后通过下采样得到的，扩大后与原级的差值反应的是高斯金字塔两级间的信息差
void Pyramid(cv::Mat srcImage){
	// 根据图像源尺寸判断是否需要缩放
	if (srcImage.rows > 400 && srcImage.cols > 400)
		cv::resize(srcImage, srcImage, cv::Size(), 0.5, 0.5);
	else // 不需要进行缩放
		cv::resize(srcImage, srcImage, cv::Size(), 1, 1);
	cv::imshow("srcImage", srcImage);
	cv::Mat pyrDownImage, pyrUpImage;
	// 下采样过程
	pyrDown(srcImage, pyrDownImage, cv::Size(srcImage.cols / 2, srcImage.rows / 2));
	cv::imshow("pyrDown", pyrDownImage);

	// 上采样过程
	pyrUp(srcImage, pyrUpImage, cv::Size(srcImage.cols * 2, srcImage.rows * 2));
	cv::imshow("pyrUp", pyrUpImage);

	// 对下采样过程重构
	cv::Mat pyrBuildImage;
	pyrUp(pyrDownImage, pyrBuildImage, cv::Size(pyrDownImage.cols * 2, pyrDownImage.rows * 2));
	cv::imshow("pyrBuildImage", pyrBuildImage);




	// 比较重构后的性能
	cv::Mat diffImage;
	resize(pyrBuildImage, pyrBuildImage, srcImage.size());

	/*cout << srcImage.size() << endl;
	cout << pyrBuildImage.size() << endl;
	cout << srcImage.channels() << endl;
	cout << pyrBuildImage.channels() << endl;*/

	cv::absdiff(srcImage, pyrBuildImage, diffImage);
	cv::imshow("diffImage", diffImage);
	cv::waitKey(0);
}

// 图像掩码操作的两种实现
// 基于像素邻域的掩码操作 低通滤波
cv::Mat Myfilter2D(cv::Mat srcImage){
	const int nChannels = srcImage.channels();
	cv::Mat resultImage(srcImage.size(), srcImage.type());
	for (int j = 1; j < srcImage.rows - 1; j++){
		// 获取邻域指针
		const uchar* previous = srcImage.ptr<uchar>(j - 1);
		const uchar* current = srcImage.ptr<uchar>(j);
		const uchar* next = srcImage.ptr<uchar>(j + 1);
		uchar * output = resultImage.ptr<uchar>(j);
		for (int i = nChannels; i < nChannels*(srcImage.cols - 1); ++i){
			// 4-邻域均值掩码操作
			*output++ = saturate_cast<uchar>((current[i - nChannels] + current[i + nChannels] + previous[i] + next[i]) / 4);
		}
	}
	// 边界处理
	resultImage.row(0).setTo(Scalar(0));
	resultImage.row(resultImage.rows - 1).setTo(Scalar(0));
	resultImage.col(0).setTo(Scalar(0));
	resultImage.col(resultImage.cols - 1).setTo(Scalar(0));
	return resultImage;
}

// opencv自带库掩码操作 低通滤波
cv::Mat filter2D_(cv::Mat srcImage){
	cv::Mat resultImage(srcImage.size(), srcImage.type());
	// 构造核函数因子
	Mat kern = (Mat_<float>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0) / (float)(4);
	filter2D(srcImage, resultImage, srcImage.depth(), kern);
	return resultImage;
}

// 图像傅里叶变换
cv::Mat DFT(cv::Mat srcImage){
	cv::Mat srcGray = GrayTrans(srcImage);

	// 将输入图像延扩到最佳的尺寸
	int nRows = getOptimalDFTSize((srcGray.rows + 1)/2*2);
	int nCols = getOptimalDFTSize((srcGray.cols+1)/2*2);
	cv::Mat resultImage;
	// 把灰度图像放在左上角，向右边和下边扩展图像
	// 将添加的像素初始化为0
	copyMakeBorder(srcGray, resultImage, 0, nRows - srcGray.rows, 0, nCols - srcGray.cols,
		BORDER_CONSTANT, Scalar::all(0));
	// 为傅里叶变换的结果（实部和虚部）分配存储空间
	cv::Mat planes[] = { cv::Mat_<float>(resultImage), cv::Mat::zeros(resultImage.size(), CV_32F) };
	Mat completeI;
	// 为延扩后的图像增添一个初始化为0的通道
	merge(planes, 2, completeI);
	// 进行离散傅里叶变换
	dft(completeI, completeI);
		// 将负数转换为幅度
	split(completeI, planes);
	magnitude(planes[0], planes[1], planes[0]);

	//saveMat("planes.xml", planes[0]);
	cv::Mat dftResultImage = planes[0];
	// 对数尺度(logarithmic scale 缩放
	dftResultImage += 1;
	log(dftResultImage, dftResultImage);
	dftResultImage = fftshift(dftResultImage);
	//// 剪切和重分布幅度图像限
	//dftResultImage = dftResultImage(Rect(0, 0, srcGray.cols, srcGray.rows));
	//// 归一化图像
	//normalize(dftResultImage, dftResultImage, 0, 1, CV_MINMAX);

	//int cx = dftResultImage.cols / 2;
	//int cy = dftResultImage.rows / 2;
	//Mat tmp;
	//// Top-left——为每一个象限创建ROI
	//Mat q0(dftResultImage, Rect(0, 0, cx, cy));
	//// Top-Right
	//Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
	//// Bottom-Left
	//Mat q2(dftResultImage, Rect(0, cy, cx, cy));
	//// Bottom——Right
	//Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
	//// 交换象限 (Top-Left with Bottom-Right)
	//q0.copyTo(tmp);
	//q3.copyTo(q0);
	//tmp.copyTo(q3);
	//// 交换象限(Top-Right with Bottom-Left)
	//q1.copyTo(tmp);
	//q2.copyTo(q1);
	//tmp.copyTo(q2);
	return dftResultImage;
}

// 图像傅里叶变换
cv::Mat DCT(cv::Mat srcImage){
	Mat src = GrayTrans(srcImage);

	src.convertTo(src, CV_64FC1);

	//DCT系数的三个通道    
	Mat dctImage(src.size(), CV_64FC1);

	//DCT变换    
	dct(src, dctImage);
	return dctImage;
}


//// 图像傅里叶逆变换
//cv::Mat INV_DFT(cv::Mat srcImage){
//	cv::Mat srcGray = GrayTrans(srcImage);
//
//	Mat image_Re = Mat::zeros(srcGray.size(), CV_64FC1);
//	Mat image_Im = Mat::zeros(srcGray.size(), CV_64FC1);
//
//	// 将输入图像延扩到最佳的尺寸
//	int nRows = getOptimalDFTSize(srcGray.rows);
//	int nCols = getOptimalDFTSize(srcGray.cols);
//
//	cv::Mat resultImage;
//	// 把灰度图像放在左上角，向右边和下边扩展图像
//	// 将添加的像素初始化为0
//	copyMakeBorder(srcGray, resultImage, 0, nRows - srcGray.rows, 0, nCols - srcGray.cols,
//		BORDER_CONSTANT, Scalar::all(0));
//	// 为傅里叶变换的结果（实部和虚部）分配存储空间
//	cv::Mat planes[] = { cv::Mat_<float>(resultImage), cv::Mat::zeros(resultImage.size(), CV_32F) };
//	Mat completeI;
//	// 为延扩后的图像增添一个初始化为0的通道
//	merge(planes, 2, completeI);
//	// 进行离散傅里叶变换
//	dft(completeI, completeI);
//	// 将负数转换为幅度
//	split(completeI, planes);
//	magnitude(planes[0], planes[1], planes[0]);
//	cv::Mat dftResultImage = planes[0];
//	// 对数尺度(logarithmic scale 缩放
//	dftResultImage += 1;
//	log(dftResultImage, dftResultImage);
//	// 剪切和重分布幅度图像限
//	dftResultImage = dftResultImage(Rect(0, 0, srcGray.cols, srcGray.rows));
//	// 归一化图像
//	normalize(dftResultImage, dftResultImage, 0, 1, CV_MINMAX);
//
//	int cx = dftResultImage.cols / 2;
//	int cy = dftResultImage.rows / 2;
//	Mat tmp;
//	// Top-left——为每一个象限创建ROI
//	Mat q0(dftResultImage, Rect(0, 0, cx, cy));
//	// Top-Right
//	Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
//	// Bottom-Left
//	Mat q2(dftResultImage, Rect(0, cy, cx, cy));
//	// Bottom——Right
//	Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
//	// 交换象限 (Top-Left with Bottom-Right)
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//	// 交换象限(Top-Right with Bottom-Left)
//	q1.copyTo(tmp);
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//	return dftResultImage;
//}

// 图像卷积操作
cv::Mat convolution(cv::Mat srcImage, cv::Mat kernel){
	Mat srcGray = GrayTrans(srcImage);
	srcGray.convertTo(srcGray, CV_32F);
	// 输出图像定义
	Mat dst = Mat::zeros(abs(srcGray.rows - kernel.rows) + 1, abs(srcGray.cols - kernel.cols) + 1, srcGray.type());
	cv::Size dftSize;
	// 计算傅里叶变换尺寸
	dftSize.width = getOptimalDFTSize(srcGray.cols + kernel.cols - 1);
	dftSize.height = getOptimalDFTSize(srcGray.rows + kernel.rows - 1);

	// 创建临时图像，初始化为0
	cv::Mat tempA(dftSize, srcGray.type(), Scalar::all(0));
	cv::Mat tempB(dftSize, kernel.type(), Scalar::all(0));
	// 对区域进行复制
	cv::Mat roiA(tempA, Rect(0, 0, srcGray.cols, srcGray.rows));
	srcGray.copyTo(roiA);
	cv::Mat roiB(tempB, Rect(0, 0, kernel.cols, kernel.rows));
	kernel.copyTo(roiB);

	// 傅里叶变换
	dft(tempA, tempA, 0, srcGray.rows);
	dft(tempB, tempB, 0, kernel.rows);
	// 对频谱中的每个元素进行惩罚操作
	mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT);
	// 变换结果,所有行非零
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, dst.rows);
	// 复制结果到输出图像
	tempA(Rect(0, 0, dst.cols, dst.rows)).copyTo(dst);
	normalize(dst, dst, 0, 1, CV_MINMAX);
	return dst;
}

// 均值滤波
cv::Mat getBlurImage(const Mat& src, Size ksize){
	if (!src.data) {
		printf("【MyBlur::getBlurImage(src, dst, ksize)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize.height % 2 != 1 || ksize.width % 2 != 1) {
		printf("【MyBlur::getBlurImage(src, dst, ksize)函数ksize的长和宽必须为奇数! ! !】");
		return Mat();
	}
	Mat dst;
	blur(src, dst, ksize);
	return dst;
}

// 中值滤波
cv::Mat getMedianBlurImage(const Mat& src, Size ksize){
	if (!src.data) {
		printf("【MyBlur::getMedianBlurImage(src, dst, ksize)函数src输入图片不能为空！！！】");
		return Mat();
	}

	if (ksize.height % 2 != 1 || ksize.width % 2 != 1) {
		printf("【MyBlur::getMedianBlurImage(src, dst, ksize)函数ksize的长和宽必须为奇数! ! !】");
		return Mat();
	}
	Mat dst;
	medianBlur(src, dst, ksize.width);
	return dst;
}

// 高斯滤波
cv::Mat getGaussianBlurImage(const Mat& src, Size ksize, double sigmaX, double sigmaY){
	if (!src.data) {
		printf("【MyBlur::getGaussianBlurImage(src, dst, ksize, sigmaX, sigmaY)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize.height % 2 != 1 || ksize.width % 2 != 1) {
		printf("【MyBlur::getGaussianBlurImage(src, dst, ksize)函数ksize的长和宽必须为奇数! ! !】");
		return Mat();
	}
	Mat dst;
	GaussianBlur(src, dst, ksize, sigmaX, sigmaY);
	return dst;
}

// 双边滤波
cv::Mat getBilateralFilterImage(const Mat& src, int d, double sigmaColor, double sigmaSpace){
	if (!src.data) {
		printf("【MyBlur::getBilateralFilterImage(src, dst, ksize, sigmaColor, sigmaSpace)函数src输入图片不能为空！！！】");
		return Mat();
	}
	Mat dst;
	bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);
	return dst;
}

// 图像导向滤波
cv::Mat guidefilter(Mat &srcImage, int r, double eps){
	if (srcImage.empty()){
		printf("MyImage.cpp guidefilter 输入图像为空!\n");
		return Mat();
	}
	if (srcImage.channels() == 3){
		// 通道分离
		vector<Mat> vSrcImage, vResultImage;
		split(srcImage, vSrcImage);
		Mat resultMat;
		for (int i = 0; i < 3; i++){
			printf("i = %d\n", i);
			// 分通道转换成浮点型数据
			Mat tempImage;
			vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
			Mat p = tempImage.clone();
			// 分别进行导向滤波
			printf("convertTo success!\n");

			// 转换源图像信息
			tempImage.convertTo(tempImage, CV_64FC1);
			p.convertTo(p, CV_64FC1);
			printf("转换原图像信息成功!\n");

			int nRows = tempImage.rows;
			int nCols = tempImage.cols;
			cv::Mat boxResult;
			// 步骤一： 计算均值
			cv::boxFilter(cv::Mat::ones(nRows, nCols, tempImage.type()), boxResult, CV_64FC1, cv::Size(r, r));
			// 生成导向均值mean_I
			cv::Mat mean_I;
			cv::boxFilter(tempImage, mean_I, CV_64FC1, cv::Size(r, r));
			// 生成原始均值mean_p
			cv::Mat mean_p;
			cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
			// 生成互相关均值mean_Ip
			cv::Mat mean_Ip;
			cv::boxFilter(tempImage.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
			cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
			// 生成自相关均值mean_II
			cv::Mat mean_II;
			// 应用盒滤波器计算相关均值
			cv::boxFilter(tempImage.mul(tempImage), mean_II, CV_64FC1, cv::Size(r, r));
			printf("Step 1 done!\n");

			// 步骤二: 计算相关系数
			cv::Mat var_I = mean_II - mean_I.mul(mean_I);
			cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
			printf("Step 2 done!\n");

			// 步骤三: 计算参数系数a、b
			cv::Mat a = cov_Ip / (var_I + eps);
			cv::Mat b = mean_p - a.mul(mean_I);
			printf("Step 3 done!\n");
			// 步骤四: 计算系数a、b的均值
			cv::Mat mean_a;
			cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
			mean_a = mean_a / boxResult;
			cv::Mat mean_b;
			cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
			mean_b = mean_b / boxResult;
			printf("Step 4 done!\n");
			// 步骤五: 生成输出矩阵
			cv::Mat resultImage = mean_a.mul(tempImage) + mean_b;


			vResultImage.push_back(resultImage);
		}
		// 通道结果合并
		merge(vResultImage, resultMat);
		return resultMat;
	}
	else {
		Mat tempImage;
		srcImage.convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
		Mat p = tempImage.clone();
		// 分别进行导向滤波
		printf("convertTo success!\n");

		// 转换源图像信息
		tempImage.convertTo(tempImage, CV_64FC1);
		p.convertTo(p, CV_64FC1);
		printf("转换原图像信息成功!\n");

		int nRows = tempImage.rows;
		int nCols = tempImage.cols;
		cv::Mat boxResult;
		// 步骤一： 计算均值
		cv::boxFilter(cv::Mat::ones(nRows, nCols, tempImage.type()), boxResult, CV_64FC1, cv::Size(r, r));
		// 生成导向均值mean_I
		cv::Mat mean_I;
		cv::boxFilter(tempImage, mean_I, CV_64FC1, cv::Size(r, r));
		// 生成原始均值mean_p
		cv::Mat mean_p;
		cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
		// 生成互相关均值mean_Ip
		cv::Mat mean_Ip;
		cv::boxFilter(tempImage.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
		cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
		// 生成自相关均值mean_II
		cv::Mat mean_II;
		// 应用盒滤波器计算相关均值
		cv::boxFilter(tempImage.mul(tempImage), mean_II, CV_64FC1, cv::Size(r, r));
		printf("Step 1 done!\n");

		// 步骤二: 计算相关系数
		cv::Mat var_I = mean_II - mean_I.mul(mean_I);
		cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
		printf("Step 2 done!\n");

		// 步骤三: 计算参数系数a、b
		cv::Mat a = cov_Ip / (var_I + eps);
		cv::Mat b = mean_p - a.mul(mean_I);
		printf("Step 3 done!\n");
		// 步骤四: 计算系数a、b的均值
		cv::Mat mean_a;
		cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
		mean_a = mean_a / boxResult;
		cv::Mat mean_b;
		cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
		mean_b = mean_b / boxResult;
		printf("Step 4 done!\n");
		// 步骤五: 生成输出矩阵
		cv::Mat resultImage = mean_a.mul(tempImage) + mean_b;
		return resultImage;
	}
}

// 差分边缘检测实现
void diffOperation(const cv::Mat srcImage, cv::Mat& edgeXImage, cv::Mat& edgeYImage){
	if (srcImage.empty()){
		printf("MyImage.cpp diffOperation 输入图像无数据!!!\n");
	}
	cv::Mat tempImage = GrayTrans(srcImage);
	edgeXImage.create(tempImage.size(), tempImage.type());
	edgeYImage.create(tempImage.size(), tempImage.type());
	int nRows = tempImage.rows;
	int nCols = tempImage.cols;
	for (int i = 0; i < nRows - 1; i++){
		for (int j = 0; j < nCols - 1; j++){
			// 计算垂直边边缘
			edgeXImage.at<uchar>(i, j) = abs(tempImage.at<uchar>(i + 1, j) - tempImage.at<uchar>(i, j));
			// 计算水平边缘
			edgeYImage.at<uchar>(i, j) = abs(tempImage.at<uchar>(i, j + 1) - tempImage.at<uchar>(i, j));
		}
	}
}

// 图像非极大值抑制Sobel边缘实现
cv::Mat getSobelVerEdge(cv::Mat srcImage){
	CV_Assert(srcImage.channels() == 1);
	srcImage.convertTo(srcImage, CV_32FC1);
	// 水平方向的Sobel算子
	cv::Mat sobelx = (cv::Mat_<float>(3, 3) << -0.125, 0, 0.125,
		-0.25, 0, 0.25,
		-0.125, 0, 0.125);
	cv::Mat ConResMat;
	// 卷积运算
	cv::filter2D(srcImage, ConResMat, srcImage.type(), sobelx);
	// 计算梯度的幅度
	cv::Mat graMagMat;
	cv::multiply(ConResMat, ConResMat, graMagMat);
	// 根据梯度幅度及参数设置阈值
	int scaleVal = 4;
	double thresh = scaleVal * cv::mean(graMagMat).val[0];
	cv::Mat resultTempMat = cv::Mat::zeros(graMagMat.size(), graMagMat.type());
	float *pDataMag = (float*)graMagMat.data;
	float* pDataRes = (float*)resultTempMat.data;
	const int nRows = ConResMat.rows;
	const int nCols = ConResMat.cols;
	for (int i = 1; i != nRows - 1; i++){
		for (int j = 1; j != nCols - 1; j++){
			// 计算该点梯度与水平或垂直梯度值得大小并比较结果
			bool b1 = (pDataMag[i*nCols + j] > pDataMag[i*nCols + j - 1]);
			bool b2 = (pDataMag[i*nCols + j] > pDataMag[i*nCols + j + 1]);
			bool b3 = (pDataMag[i*nCols + j] > pDataMag[(i - 1)*nCols + j]);
			bool b4 = (pDataMag[i*nCols + j] > pDataMag[(i + 1)*nCols + j]);

			// 判断邻域梯度是否满足大于水平或垂直梯度的条件
			// 并根据自适应阈值参数进行二值化
			pDataRes[i*nCols + j] = (float)(255 * ((pDataMag[i*nCols + j] > thresh) && ((b1&&b2) || (b3&&b4))));
		}
	}
	resultTempMat.convertTo(resultTempMat, CV_8UC1);
	Mat resultImage = resultTempMat.clone();
	return resultImage;
}

// 图像直接卷积Sobel边缘实现 模长阈值为梯度幅值的阈值
cv::Mat getsobelEdge(const cv::Mat& srcImage, uchar threshold){
	CV_Assert(srcImage.channels() == 1);
	// 初始化水平核因子
	Mat sobelx = (Mat_<double>(3, 3) << 1, 0,
		-1, 2, 0, -2, 1, 0, -1);
	// 初始化垂直核因子
	Mat sobely = (Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0, -1, -2, -1);
	Mat resultImage = Mat::zeros(srcImage.rows - 2, srcImage.cols - 2, srcImage.type());
	double edgeX = 0;
	double edgeY = 0;
	double graMag = 0;
	for (int k = 1; k < srcImage.rows - 1; ++k){
		for (int n = 1; n < srcImage.cols - 1; ++n){
			edgeX = 0;
			edgeY = 0;
			// 遍历计算水平与垂直梯度
			for (int i = -1; i <= 1; ++i){
				for (int j = -1; j <= 1; j++){
					edgeX += srcImage.at<uchar>(k + i, n + j)*sobelx.at<double>(1 + i, 1 + j);
					edgeY += srcImage.at<uchar>(k + i, n + j)*sobely.at<double>(1 + i, 1 + j);
				}
			}
			// 计算梯度模长
			graMag = sqrt(pow(edgeY, 2) + pow(edgeX, 2));
			// 二值化
			resultImage.at<uchar>(k - 1, n - 1) = ((graMag > threshold) ? 255 : 0);
		}
	}
	return resultImage;
}

// 图像卷积下非极大值抑制Sobel实现
// flag = 0 横向梯度 
// flag = 1 纵向梯度
// flag = 2 全面的梯度
cv::Mat getsobelOptaEdge(const cv::Mat& srcImage, int flag){
	CV_Assert(srcImage.channels() == 1);
	// 初始化Sobel水平核因子
	cv::Mat sobelX = (cv::Mat_<double>(3, 3) << 1, 0, -1,
		2, 0, -2,
		1, 0, -1);
	// 初始化Sobel垂直核因子
	cv::Mat sobelY = (cv::Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0,
		-1, -2, -1);
	// 计算水平与垂直卷积
	cv::Mat edgeX, edgeY;
	filter2D(srcImage, edgeX, CV_32F, sobelX);
	filter2D(srcImage, edgeY, CV_32F, sobelY);
	// 根据传入参数确定计算水平或垂直边缘
	int paraX = 0;
	int paraY = 0;
	switch (flag){
	case 0: paraX = 1;
		paraY = 0;
		break;
	case 1: paraX = 0;
		paraY = 1;
		break;
	case 2: paraX = 1;
		paraY = 1;
		break;
	default: break;
	}
	edgeX = abs(edgeX);
	edgeY = abs(edgeY);
	cv::Mat graMagMat = paraX*edgeX.mul(edgeX) + paraY*edgeY.mul(edgeY);
	// 计算阈值
	int scaleVal = 4;
	double thresh = scaleVal * cv::mean(graMagMat).val[0];
	Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	for (int i = 1; i < srcImage.rows - 1; i++){
		float *pDataEdgeX = edgeX.ptr<float>(i);
		float *pDataEdgeY = edgeY.ptr<float>(i);
		float *pDataGraMag = graMagMat.ptr<float>(i);
		// 阈值化和极大值抑制
		for (int j = 1; j < srcImage.cols - 1; j++){
			// 判断当前邻域梯度是否大于阈值与大于水平或垂直梯度
			if (pDataGraMag[j] > thresh && ((pDataEdgeX[j] > paraX * pDataEdgeY[j] &&
				pDataGraMag[j] > pDataGraMag[j - 1] && pDataGraMag[j] >
				pDataGraMag[j + 1]) ||
				(pDataEdgeY[j] > paraY * pDataEdgeX[j] &&
				pDataGraMag[j] >
				pDataGraMag[j - 1] && pDataGraMag[j] >
				pDataGraMag[j + 1])))
				resultImage.at<uchar>(i, j) = 255;
		}
	}
	return resultImage;
}

// OpenCV自带库图像边缘计算 
// flag可取下面的值
//#define EDGE_SOBEL_VER 0
//#define EDGE_SOBEL_HOR 1
//#define EDGE_SOBEL_ALL 2
//#define EDGE_SCHARR_VER 3
//#define EDGE_SCHARR_HOR 4
//#define EDGE_SCHARR_ALL 5
cv::Mat getSobelEdgeImage(const cv::Mat srcImage, int flag){
	CV_Assert(srcImage.data != NULL);
	Mat srcGray = GrayTrans(srcImage);
	Mat resultImage, edgeXMat, edgeYMat;
	switch (flag)
	{
	case EDGE_SOBEL_VER:
		Sobel(srcGray, resultImage, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		// 线性变换，转换输入数组元素为8位无符号类型
		convertScaleAbs(resultImage, resultImage);
		break;
	case EDGE_SOBEL_HOR:
		Sobel(srcGray, resultImage, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(resultImage, resultImage);
		break;
	case EDGE_SOBEL_ALL:
		edgeXMat = getSobelEdgeImage(srcGray, EDGE_SOBEL_VER);
		edgeYMat = getSobelEdgeImage(srcGray, EDGE_SOBEL_HOR);
		addWeighted(edgeXMat, 0.5, edgeYMat, 0.5, 0, resultImage);
		break;
	case EDGE_SCHARR_VER:
		Scharr(srcGray, resultImage, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(resultImage, resultImage);
		break;
	case EDGE_SCHARR_HOR:
		Scharr(srcGray, resultImage, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(resultImage, resultImage);
		break;
	case EDGE_SCHARR_ALL:
		edgeXMat = getSobelEdgeImage(srcGray, EDGE_SCHARR_VER);
		edgeYMat = getSobelEdgeImage(srcGray, EDGE_SCHARR_HOR);
		addWeighted(edgeXMat, 0.5, edgeYMat, 0.5, 0, resultImage);
		break;
	default:
		break;
	}
	return resultImage;
}

// 获取Laplace边缘
cv::Mat getLaplaceEdge(cv::Mat srcImage){
	CV_Assert(!srcImage.empty());
	// 高斯平滑
	GaussianBlur(srcImage, srcImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cv::Mat dstImage;
	// 拉普拉斯变换
	Laplacian(srcImage, dstImage, CV_16S, 3);
	convertScaleAbs(dstImage, dstImage);
	return dstImage;
}

// Robert边缘检测
// Robert算子是利用局部差分寻找边缘的一种算子，是最简单的边缘检测算子。Roberts算子利用对角线
// 方向相邻两像素之差近似梯度幅值来检测边缘，检测垂直边缘的效果要优于其他方向边缘，定位精度高，
// 但对噪声的抑制能力较弱。边缘检测算子检查每个像素的邻域并对灰度变化量进行量化，同时也包含
// 方向的确定。
cv::Mat getRobertsEdge(cv::Mat srcImage){
	cv::Mat dstImage = srcImage.clone();
	int nRows = dstImage.rows;
	int nCols = dstImage.cols;
	for (int i = 0; i < nRows - 1; i++){
		for (int j = 0; j < nCols - 1; j++){
			// 根据公式计算
			int t1 = (srcImage.at<uchar>(i, j) - srcImage.at<uchar>(i + 1, j + 1)) *
				(srcImage.at<uchar>(i, j) - srcImage.at<uchar>(i + 1, j + 1));
			int t2 = (srcImage.at<uchar>(i + 1, j) - srcImage.at<uchar>(i, j + 1)) *
				(srcImage.at<uchar>(i + 1, j) - srcImage.at<uchar>(i, j + 1));
			// 计算对角线像素差
			dstImage.at<uchar>(i, j) = (uchar)sqrt((double)(t1 + t2));
		}
	}
	return dstImage;
}

// Prewitt边缘检测
// Prewitt算子是一阶边缘检测算子，该算子对噪声有抑制作用。Prewitt算子对边缘的定位精度不如Roberts算子，
// Sobel算子对边缘检测的准确性更优于Prewitt算子。
cv::Mat getPrewittEdge(cv::Mat srcImage, bool verFlag){
	srcImage.convertTo(srcImage, CV_32FC1);
	cv::Mat prewitt_kernel = (cv::Mat_<float>(3, 3) << 0.1667, 0.1667, 0.1667,
		0, 0, 0,
		-0.1667, -0.1667, -0.1667);
	// 垂直边缘
	if (verFlag){
		prewitt_kernel = prewitt_kernel.t();
		cv::Mat z1 = cv::Mat::zeros(srcImage.rows, 1, CV_32FC1);
		cv::Mat z2 = cv::Mat::zeros(1, srcImage.cols, CV_32FC1);
		// 将图像的四边设为0
		z1.copyTo(srcImage.col(0));
		z1.copyTo(srcImage.col(srcImage.cols - 1));
		z2.copyTo(srcImage.row(0));
		z2.copyTo(srcImage.row(srcImage.rows - 1));
	}
	cv::Mat edges;
	cv::filter2D(srcImage, edges, srcImage.type(), prewitt_kernel);
	cv::Mat mag;
	cv::multiply(edges, edges, mag);
	// 去除垂直边缘
	if (verFlag){
		cv::Mat black_region = srcImage < 0.03;
		cv::Mat se = cv::Mat::ones(5, 5, CV_8UC1);
		cv::dilate(black_region, black_region, se);
		mag.setTo(0, black_region);
	}
	// 根据模长计算出梯度的阈值
	double thresh = 4.0f * cv::mean(mag).val[0];
	// 仅在某点梯度大于方向或垂直方向的邻点梯度时
	// 才设该位置的输出值为255
	// 并应用阈值thresh
	cv::Mat dstImage = cv::Mat::zeros(mag.size(), mag.type());
	float *dptr = (float*)mag.data;
	float *tptr = (float*)dstImage.data;
	int r = edges.rows, c = edges.cols;
	for (int i = 1; i != r - 1; ++i){
		for (int j = 1; j != c - 1; ++j){
			// 非极大值抑制
			bool b1 = (dptr[i*c + j] > dptr[i*c + j - 1]);
			bool b2 = (dptr[i*c + j] > dptr[i*c + j + 1]);
			bool b3 = (dptr[i*c + j] > dptr[(i - 1)*c + j]);
			bool b4 = (dptr[i*c + j] > dptr[(i + 1)*c + j]);
			tptr[i*c + j] = (float)(255 * ((dptr[i*c + j] > thresh) && ((b1&&b2) || (b3&&b4))));
		}
	}
	dstImage.convertTo(dstImage, CV_8UC1);
	return dstImage;
}

// Canny库函数实现 推荐的高与低阈值比值在2:1到3:1之间
cv::Mat getCannyEdge(cv::Mat srcImage, int lowThresh, int highThresh){
	CV_Assert(!srcImage.empty());
	Mat resultImage;
	// Canny检测
	Canny(srcImage, resultImage, lowThresh, highThresh, 3);
	return resultImage;
}

// 改进边缘检测算子Marr-Hildreth LoG算子
// 它把高斯平滑滤波器和拉普拉斯锐化滤波器结合起来，先平Q滑掉噪声，再进行边缘检测
cv::Mat getMarrEdge(const Mat src, int kerValue, double delta){
	// 计算LOG算子
	Mat kernel;
	// 半径
	int kerLen = kerValue / 2;
	kernel = Mat_<double>(kerValue, kerValue);
	// 滑窗
	for (int i = -kerLen; i <= kerLen; i++){
		for (int j = -kerLen; j <= kerLen; j++){
			// 生成核因子
			kernel.at<double>(i + kerLen, j + kerLen) =
				exp(-((pow(j, 2.0) + pow(i, 2.0)) /
				(pow(delta, 2.0) * 2)))
				*(((pow(j, 2.0) + pow(i, 2.0) - 2 *
				pow(delta, 2.0)) / (2 * pow(delta, 4.0))));
		}
	}
	// 设置输出参数
	int kerOffset = kerValue / 2;
	Mat laplacian = (Mat_<double>(src.rows - kerOffset * 2, src.cols - kerOffset * 2));
	Mat result = Mat::zeros(src.rows - kerOffset * 2, src.cols - kerOffset * 2, src.type());
	double sumLaplacian;
	// 遍历计算卷积图像的拉普拉斯算子
	for (int i = kerOffset; i < src.rows - kerOffset; ++i){
		for (int j = kerOffset; j < src.cols - kerOffset; ++j){
			sumLaplacian = 0;
			for (int k = -kerOffset; k <= kerOffset; ++k){
				for (int m = -kerOffset; m <= kerOffset; ++m){
					// 计算图像卷积
					sumLaplacian += src.at<uchar>(i + k, j + m)*kernel.at<double>(kerOffset + k, kerOffset + m);
				}
			}
			// 生成拉普拉斯结果
			laplacian.at<double>(i - kerOffset, j - kerOffset) = sumLaplacian;
		}
	}

	// 过零点交叉，寻找边缘像素
	for (int y = 1; y < result.rows - 1; y++){
		for (int x = 1; x < result.cols - 1; x++){
			result.at<uchar>(y, x) = 0;
			// 邻域判定
			if (laplacian.at<double>(y - 1, x) * laplacian.at<double>(y + 1, x) < 0){
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y, x - 1) * laplacian.at<double>(y, x + 1) < 0){
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y + 1, x - 1)*laplacian.at<double>(y - 1, x + 1) < 0){
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y - 1, x - 1)*laplacian.at<double>(y + 1, x + 1) < 0){
				result.at<uchar>(y, x) = 255;
			}
		}
	}
	return result;
}

// MoravecCorners角点检测
cv::Mat MoravecCorners(cv::Mat srcImage, vector<Point> & points, int kSize, int threshold){
	cv::Mat resMorMat = convert2BGR(srcImage);
	// 获取初始化参数信息
	int r = kSize / 2;
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	int nCount = 0;

	// 图像遍历
	for (int i = r; i < srcImage.rows - r; i++){
		for (int j = r; j < srcImage.cols - r; j++){
			int wV1, wV2, wV3, wV4;
			wV1 = wV2 = wV3 = wV4 = 0;
			// 计算水平方向窗内的兴趣值
			for (int k = -r; k < r; k++){
				wV1 += (srcImage.at<uchar>(i, j + k) - srcImage.at<uchar>(i, j + k + 1)) * (srcImage.at<uchar>(i, j + k) - srcImage.at<uchar>(i, j + k + 1));
			}
			// 计算垂直方向窗内的兴趣值
			for (int k = -r; k < r; k++){
				wV2 += (srcImage.at<uchar>(i + k, j) - srcImage.at<uchar>(i + k + 1, j)) * (srcImage.at<uchar>(i + k, j) - srcImage.at<uchar>(i + k + 1, j));
			}
			// 计算45°方向窗内的兴趣值
			for (int k = -r; k < r; k++){
				wV3 += (srcImage.at<uchar>(i + k, j + k) - srcImage.at<uchar>(i + k + 1, j + k + 1)) * (srcImage.at<uchar>(i + k, j + k) - srcImage.at<uchar>(i + k + 1, j + k + 1));
			}
			// 计算135°方向窗内的兴趣值
			for (int k = -r; k < r; k++){
				wV4 += (srcImage.at<uchar>(i + k, j - k) - srcImage.at<uchar>(i + k + 1, j - k - 1)) * (srcImage.at<uchar>(i + k, j - k) - srcImage.at<uchar>(i + k + 1, j - k - 1));
			}

			// 取其中的最小值作为该像素点的最终兴趣值
			int value = min(min(wV1, wV2), min(wV3, wV4));
			// 若兴趣值大于阈值，则将点的坐标存入数组中
			if (value > threshold){
				points.push_back(Point(j, i));
				nCount++;
			}
		}
	}
	drawVecPoints(resMorMat, points);
	return resMorMat;
}


// 绘制Harris角点
Mat getHarrisCornersImage(const Mat& srcImage, float thresh, int blockSize, int kSize, double k){
	CV_Assert(!srcImage.empty());
	Mat src = GrayTrans(srcImage);
	Mat result(src.size(), CV_32F);
	int depth = src.depth();
	// 检测掩模尺寸
	double scale = (double)(1 << ((kSize > 0 ? kSize : 3) - 1))*blockSize;
	if (depth == CV_8U)
		scale *= 255.;
	scale = 1. / scale;
	// Sobel滤波
	Mat dx, dy;
	Sobel(src, dx, CV_32F, 1, 0, kSize, scale, 0);
	Sobel(src, dy, CV_32F, 0, 1, kSize, scale, 0);
	Size size = src.size();
	cv::Mat cov(size, CV_32FC3);

	// 求解水平与竖直梯度
	for (int i = 0; i < size.height; i++){
		float *covData = (float*)(cov.data + i*cov.step);
		const float *dxData = (const float*)(dx.data + i*dx.step);
		const float *dyData = (const float*)(dy.data + i*dy.step);
		for (int j = 0; j < size.width; j++){
			float dx_ = dxData[j];
			float dy_ = dyData[j];
			covData[3 * j] = dx_*dx_;
			covData[3 * j + 1] = dx_*dy_;
			covData[3 * j + 2] = dy_*dy_;
		}
	}

	//printf("boxFilter go!\n");
	// 对图像进行盒滤波操作
	boxFilter(cov, cov, cov.depth(), Size(blockSize, blockSize), Point(-1, -1), false);
	// 判断图像连续性
	if (cov.isContinuous() && result.isContinuous()){
		size.width *= size.height;
		size.height = 1;
	}
	else
		size = result.size();
	// 计算响应函数
	for (int i = 0; i < size.height; i++){
		// 获取图像矩阵指针
		float *resultData = (float*)(result.data + i*result.step);
		const float* covData = (const float*)(cov.data + i*cov.step);
		for (int j = 0; j < size.width; j++){
			// 焦点响应生成
			float a = covData[3 * j];
			float b = covData[3 * j + 1];
			float c = covData[3 * j + 2];
			resultData[j] = (float)(a*c - b*b - k*(a + c) * (a + c));
		}
	}

	Mat drawing = convert2BGR(srcImage);
	// 矩阵归一化
	normalize(result, result, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(result, result);
	//printf("drawing go!\n");

	// 绘制角点检测结果
	for (int j = 0; j < result.rows; j++){
		for (int i = 0; i < result.cols; i++){
			//printf("i = %d j = %d\n", i, j);
			if (result.at<float>(j, i) > thresh){
				circle(drawing, Point(i, j), 5, MC_YELLOW, 2, 8, 0);
			}
		}
	}
	return drawing;
}

// 得到自定义核
// 参数一 int   kshape	: 表示内核的形状，有三种选择<1> 矩形： MORPH_RECT; <2> 交叉形: MORPH_CROSS; <3> 圆形: MORPH_ELLIPSE
// 参数二 int   ksize	: 表示内核的尺寸
// 参数三 Point kpos		: 表示锚点的位置
Mat getCustomKernel(int ksize, int kshape, Point kpos) {
	// 默认的锚点在内核的中心点
	if (kpos.x == -1){
		kpos.x = ksize / 2;
		kpos.y = ksize / 2;
	}

	Mat element = getStructuringElement(kshape,
		Size(ksize, ksize),
		Point(kpos.x, kpos.y));
	return element;
}

// 顶帽运算 Top Hat 又称“礼帽”运算, 得到的效果图突出了比原图轮廓周围的区域更明亮的区域，顶帽操作往往用来分离比邻近点亮一些的斑块。
// 在一幅图像具有大幅的背景，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取.
// dst = tophat(src, dst) = src = open(src, element)
Mat getMorphTopHatImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("【MyImage::getMorphTopHatImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getMorphTopHatImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_TOPHAT, element);
	return dst;
}

// 黑帽运算 Black Hat ,得到的效果图突出了比原图轮廓周围的区域更暗的区域，黑帽运算用来分离比邻近点暗一些的斑块，效果图有着非常完美的轮廓。
Mat getMorphBlackHatImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("【MyImage::getMorphBlackHatImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getMorphBlackHatImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_BLACKHAT, element);
	return dst;
}

// 形态学梯度
// dst = morph-grad(src,element) = dilate(src,element) - erode(src,element)
Mat getMorphGradientImage(const Mat& src, int ksize, int kshape, Point kpos) {
	if (!src.data) {
		printf("【MyImage::getMorphGradientImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getMorphGradientImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_GRADIENT, element);
	return dst;
}

// 开运算
// dst = open(src, element) = dilate(erode(src, element))
// 开运算可以用来消除小物体，在纤细点处分离物体，并且在平滑较大物体的边界的同时不明显改变其面积。
Mat getOpeningOperationImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("【MyImage::getOpeningOperationImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getOpeningOperationImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_OPEN, element);
	return dst;
}

// 闭运算
// dst = close(src, element) = erode(dilate(src, element))
// 闭运算能够排除小型黑洞（黑色区域）
Mat getClosingOperationImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("【MyImage::getClosingOperationImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getClosingOperationImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_CLOSE, element);
	return dst;
}

// 得到腐蚀的图像
Mat getErodeImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("【MyImage::getErodeImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getErodeImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}

	Mat ele = getStructuringElement(kshape, Size(ksize, ksize), kpos);
	Mat dst;
	erode(src, dst, ele);
	return dst;
}

// 得到膨胀的图像
Mat getDilateImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("【MyImage::getDilateImage(src, dst, ksize, shape, kpos)函数src输入图片不能为空！！！】");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("【MyImage::getDilateImage(src, dst, ksize, shape, kpos)函数ksize必须为奇数! ! !】");
		return Mat();
	}

	Mat ele = getStructuringElement(kshape, Size(ksize, ksize), kpos);
	Mat dst;
	dilate(src, dst, ele);
	return dst;
}

// 分水岭图像分割
Mat getWatershedSegmentImage(Mat &srcImage, int& noOfSegments, Mat& markers){
	Mat grayMat = GrayTrans(srcImage);
	Mat otsuMat;
	//imshow("graymat", grayMat);
	// 阈值操作
	threshold(grayMat, otsuMat, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	//imshow("otsuMat", otsuMat);

	// 形态学开操作
	morphologyEx(otsuMat, otsuMat, MORPH_OPEN, Mat::ones(7, 7, CV_8SC1), Point(4, 4), 2);
	//imshow("Mor-openMat", otsuMat);
	// 距离变换
	Mat disTranMat(otsuMat.size(), CV_32FC1);
	distanceTransform(otsuMat, disTranMat, CV_DIST_L2, 3);
	// 归一化
	normalize(disTranMat, disTranMat, 0.0, 1, NORM_MINMAX);
	//imshow("DistranMat", disTranMat);
	// 阈值化分割图像
	threshold(disTranMat, disTranMat, 0.1, 1, CV_THRESH_BINARY);
	// 归一化统计图像到0~255
	normalize(disTranMat, disTranMat, 0.0, 255.0, NORM_MINMAX);
	disTranMat.convertTo(disTranMat, CV_8UC1);
	//imshow("TDisTranMat", disTranMat);
	// 计算标记的分割块
	int compCount = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(disTranMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	if (contours.empty())
		return Mat();
	//Mat markers(disTranMat.size(), CV_32S);
	markers.create(disTranMat.size(), CV_32S);
	markers = Scalar::all(0);
	int idx = 0;
	// 绘制区域块
	for (; idx >= 0; idx = hierarchy[idx][0], compCount++){
		drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
	}
	if (compCount == 0)
		return Mat();
	// 计算算法的时间复杂度
	double t = (double)getTickCount();
	watershed(srcImage, markers);
	t = (double)getTickCount() - t;
	printf("execution time = %gms\n", t*1000. / getTickFrequency());

	noOfSegments = compCount;
	//Mat wshed = markers * 255;
	/*Mat wshed = (markers, compCount);
	imshow("watershed transform", wshed);
	*/
	Mat wshed = showWaterSegResult(markers);
	return wshed;
}

// 显示分水岭分割算法结果图像
Mat showWaterSegResult(Mat markers){
	Mat wshed;
	markers.convertTo(wshed, CV_8U);
	wshed = getContrastStretchImage(wshed);
	imshow("wshed", wshed);
	return wshed;
}


// 分割合并
void segMerge(Mat& image, Mat& segments, int & numSeg){
	// 对一个分割部分进行像素统计
	vector<Mat> samples;
	// 统计数据更新
	int newNumSeg = numSeg;
	// 初始化分割部分
	for (int i = 0; i <= numSeg; i++){
		Mat sampleImage;
		//cout << "ok" << endl;
		samples.push_back(sampleImage);
	}
	// 统计每一个部分
	for (int i = 0; i < segments.rows; i++){
		for (int j = 0; j < segments.cols; j++){
			// 检查每个像素的归属
			//cout << "hh" << endl;
			int index = segments.at<int>(i, j);
			//cout << "hh" << endl;
			if (index >= 0 && index < numSeg)
				samples[index].push_back(image(Rect(j, i, 1, 1)));
		}
	}

	// 创建直方图
	vector<MatND> hist_bases;
	Mat hsv_base;
	// 设置直方图参数
	int h_bins = 35;
	int s_bins = 30;
	int histSize[] = { h_bins, s_bins };
	// hue变换范围0~256, saturation变换范围0~180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };
	const float* ranges[] = { h_ranges, s_ranges };
	// 使用第0与第1通道
	int channels[] = { 0, 1 };
	// 生成直方图
	MatND hist_base;
	for (int c = 1; c < numSeg; c++){
		if (samples[c].dims>0){
			// 将区域部分转换成HSV
			cvtColor(samples[c], hsv_base, CV_BGR2HSV);
			// 直方图统计
			calcHist(&hsv_base, 1, channels, Mat(),
				hist_base, 2, histSize, ranges, true, false);
			// 直方图归一化
			normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
			// 添加到统计集
			hist_bases.push_back(hist_base);
		}
		else {
			hist_bases.push_back(MatND());
		}
		hist_base.release();
	}
	cout << "直方图ok" << endl;

	double similarity = 0;
	vector<bool> mearged;
	for (int k = 0; k < (int)hist_bases.size(); k++){
		mearged.push_back(false);
	}
	// 统计每一个部分的直方图相似
	for (int c = 0; c < (int)hist_bases.size(); c++){
		for (int q = c + 1; q < (int)hist_bases.size(); q++){
			if (!mearged[q]){
				// 判断直方图的维度
				if (hist_bases[c].dims > 0 && hist_bases[q].dims > 0){
					// 直方图对比
					similarity = compareHist(hist_bases[c], hist_bases[q], CV_COMP_BHATTACHARYYA);

					if (similarity > 0.99){
						mearged[q] = true;
						if (q != c){
							// 减少区域部分
							newNumSeg--;
							for (int i = 0; i < segments.rows; i++){
								for (int j = 0; j < segments.cols; j++){
									int index = segments.at<int>(i, j);
									// 合并
									if (index == q){
										segments.at<int>(i, j) = c;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	numSeg = newNumSeg;
}

// 颜色通道分离
static void MergeSeg(Mat& img, const Scalar& colorDiff){
	CV_Assert(!img.empty());
	Mat img_copy = img.clone();
	RNG rng = theRNG();
	// 定义掩码图像
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	Mat mask1(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			if (mask.at<uchar>(y + 1, x + 1) == 0){
				// 定义颜色
				Scalar newVal(rng(256), rng(256), rng(256));
				// 泛洪合并
				//if (floodFill(img_copy, mask1, Point(x, y), newVal, 0, colorDiff, colorDiff) > 120){
				//cout << "hello" << endl;
				floodFill(img, mask, Point(x, y), newVal, 0, colorDiff, colorDiff);
				//}
				//cout << "area = " << area << endl;
			}
		}
	}
}


// 泛洪填充FloodFill图像分割
Mat getFloodFillImage(const Mat&srcImage, Mat mask, Point pt, int& area, int ffillMode, int loDiff, int upDiff,
	int connectivity, bool useMask, Scalar color, int newMaskVal){
	// floodfill参数设置
	Point seed = pt;
	int lo = ffillMode == 0 ? 0 : loDiff;
	int up = ffillMode == 0 ? 0 : upDiff;
	int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	Mat dst = convert2BGR(srcImage);
	Rect ccomp;

	// 根据标志位选择泛红填充
	if (useMask){
		// 阈值化操作
		cv::threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
		area = floodFill(dst, mask, seed, color, &ccomp, Scalar(lo, lo, lo), Scalar(up, up, up), flags);
		//imshow("mask", mask);
		return mask;
	}
	else {
		// 泛洪填充
		area = floodFill(dst, seed, color, &ccomp, Scalar(lo, lo, lo), Scalar(up, up, up), flags);
		//imshow("image", dst);
		return dst;
	}
}

// MeanShift图像分割
Mat getMeanShiftImage(const Mat &srcImage, int spatialRad, int colorRad, int maxPyrLevel){
	CV_Assert(!srcImage.empty());
	Mat resImg;
	// 均值漂移分割
	pyrMeanShiftFiltering(srcImage, resImg, spatialRad, colorRad, maxPyrLevel);

	imshow("resImg", resImg);

	// 颜色通道分离合并
	MergeSeg(resImg, Scalar::all(2));
	return resImg;
}

// Grabcut图像分割 返回前景图像
Mat getGrabcutImage(const Mat& srcImage, Rect roi){
	CV_Assert(!srcImage.empty());
	// 定义前景与输出图像;
	Mat srcImage2 = srcImage.clone();
	cv::Mat foreground(srcImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat result(srcImage.size(), CV_8UC1);
	// Grabcut分割前景与背景
	cv::Mat fgMat, bgMat;

	// 迭代次数
	int i = 20;
	std::cout << "20 iters" << std::endl;
	// 实现图割操作
	grabCut(srcImage, result, roi, bgMat, fgMat, i, GC_INIT_WITH_RECT);

	// 图像匹配
	compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);

	// 生成前景图像
	srcImage.copyTo(foreground, result);

	return foreground;
}

// 尺度变换实现
bool CreateScaleSpace(cv::Mat srcImage, std::vector<std::vector<Mat>> &ScaleSpace, std::vector<std::vector<Mat>> &DoG){
	if (!srcImage.data){
		return false;
	}
	cv::Size ksize(5, 5);
	double sigma;
	Mat srcBlurMat, up, down;
	// 高斯平滑
	GaussianBlur(srcImage, srcBlurMat, ksize, 0.5);
	// 金字塔
	pyrUp(srcBlurMat, up);
	up.copyTo(ScaleSpace[0][0]);
	// 高斯平滑
	GaussianBlur(ScaleSpace[0][0], ScaleSpace[0][0], ksize, 1.0);
	// 图像遍历
	for (int i = 0; i < 4; i++){
		// 平滑因子
		double sigma = 1.1412135;
		for (int j = 0; j < 5 + 2; j++){
			sigma = sigma*pow(2.0, j / 2.0);
			// 对下一尺度进行高斯操作
			GaussianBlur(ScaleSpace[i][j], ScaleSpace[i][j + 1], ksize, sigma);
			// 生成多尺度空间
			DoG[i][j] = ScaleSpace[i][j] - ScaleSpace[i][j + 1];
			// 输出对应特征空间尺度
			cout << "iave:" << i << " Scale:" << j << "size:" <<
				ScaleSpace[i][j].rows << "x" <<
				ScaleSpace[i][j].cols << endl;
		}

		// 如果不能完成，继续进行金字塔操作
		if (i < 3){
			// 金字塔下采样
			pyrDown(ScaleSpace[i][0], down);
			down.copyTo(ScaleSpace[i + 1][0]);
		}
	}
	return true;
}


// 积分图实现 HOG特征描述实现
// 计算积分图
std::vector<cv::Mat> CalculateIntegralHOG(Mat &srcMat){
	// Sobel边缘检测
	Mat sobelMatX, sobelMatY;
	Sobel(srcMat, sobelMatX, CV_32F, 1, 0);
	Sobel(srcMat, sobelMatY, CV_32F, 0, 1);
	std::vector<Mat> bins(NBINS);

	for (int i = 0; i < NBINS; i++){
		bins[i] = Mat::zeros(srcMat.size(), CV_32F);
	}

	Mat magnMat, angleMat;
	// 坐标转换
	cartToPolar(sobelMatX, sobelMatY, magnMat, angleMat, true);
	// 角度变换
	add(angleMat, Scalar(180), angleMat, angleMat < 0);
	add(angleMat, Scalar(-180), angleMat, angleMat >= 180);
	angleMat /= THETA;
	for (int y = 0; y < srcMat.rows; y++){
		for (int x = 0; x < srcMat.cols; x++){
			// 计算bins下幅值
			int ind = angleMat.at<float>(y, x);
			bins[ind].at<float>(y, x) += magnMat.at<float>(y, x);
		}
	}
	// 生成积分图图像
	std::vector<Mat> integrals(NBINS);
	for (int i = 0; i < NBINS; i++){
		integral(bins[i], integrals[i]);
	}
	return integrals;
}

// 快速区域积分直方图实现
// 计算单个cell HOG特征
void calHOGinCell(Mat& HOGCellMat, Rect roi, std::vector<Mat>& integrals){
	// 实现快速积分HOG
	int x0 = roi.x, y0 = roi.y;
	int x1 = x0 + roi.width;
	int y1 = y0 + roi.height;
	for (int i = 0; i < NBINS; i++){
		// 根据矩阵的上下左右坐标计算
		Mat integral = integrals[i];
		float a = integral.at<double>(y0, x0);
		float b = integral.at<double>(y1, x1);
		float c = integral.at<double>(y0, x1);
		float d = integral.at<double>(y1, x0);
		HOGCellMat.at<float>(0, i) = (a + b) - (c + d);
	}
}

// 获取HOG直方图
cv::Mat getHog(Point pt, std::vector<Mat> &integrals){
	// 判断当前点的位置是否符合条件
	if (pt.x - R_HOG < 0 || pt.y - R_HOG < 0 || pt.x + R_HOG >= integrals[0].cols || pt.y + R_HOG >= integrals[0].rows){
		return Mat();
	}

	// 直方图
	Mat hist(Size(NBINS*BLOCKSIZE*BLOCKSIZE, 1), CV_32F);
	Point t1(0, pt.y - R_HOG);
	int c = 0;
	// 遍历块
	for (int i = 0; i < BLOCKSIZE; i++){
		t1.x = pt.x - R_HOG;
		for (int j = 0; j < BLOCKSIZE; j++){
			// 获取当前窗口，计算局部直方图
			Rect roi(t1, t1 + Point(CELLSIZE, CELLSIZE));
			// 计算当前bins下直方图
			Mat hist_temp = hist.colRange(c, c + NBINS);
			calHOGinCell(hist_temp, roi, integrals);
			// cell 步长尺寸
			t1.x += CELLSIZE;
			c += NBINS;
		}
		t1.y = CELLSIZE;
	}

	// 归一化L2范数
	normalize(hist, hist, 1, 0, NORM_L2);
	return hist;
}

// 计算HOG特征 此函数存在问题 需待验证
std::vector<Mat> calHOGFeature(cv::Mat srcImage){
	Mat grayImage = GrayTrans(srcImage);
	std::vector<Mat> HOGMatVector;
	grayImage.convertTo(grayImage, CV_8UC1);

	// 生成积分图
	std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
	Mat image = grayImage.clone();
	// 灰度值缩小
	image *= 0.5;
	// HOG特征矩阵
	cv::Mat HOGBlockMat(Size(NBINS, 1), CV_32F);
	// cell遍历
	for (int y = CELLSIZE / 2; y < grayImage.rows; y += CELLSIZE){
		for (int x = CELLSIZE / 2; x < grayImage.cols; x += CELLSIZE){
			// 获取当前窗口HOG
			cv::Mat hist = getHog(Point(x, y), integrals);
			if (hist.empty()){
				continue;
			}
			HOGBlockMat = Scalar(0);
			for (int i = 0; i < NBINS; i++){
				for (int j = 0; j < BLOCKSIZE; j++){
					HOGBlockMat.at<float>(0, i) += hist.at<float>(0, i + j*NBINS);
				}
			}
			// L2范数归一化
			normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
			HOGMatVector.push_back(HOGBlockMat);

			Point center(x, y);

			//if (y % 7 != 0 || x % 7 != 0)
			//	continue;

			// 绘制HOG特征图
			for (int i = 0; i < NBINS; i++){
				// 角度获取
				double theta = (i*THETA + 90.0) * CV_PI / 180.0;
				Point rd(CELLSIZE * 0.5*cos(theta), CELLSIZE*0.5*sin(theta));

				// 获取绘制中心
				Point rp = center - rd;
				Point lp = center - (-rd);
				// 绘制HOG特征块
				line(image, rp, lp, Scalar(255 * HOGBlockMat.at<float>(0, i), 255, 255));
			}
		}
	}
	imshow("out", image);
	return HOGMatVector;
}

// 计算LBP特征
cv::Mat getLBPImage(cv::Mat & srcImage){
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	srcImage = GrayTrans(srcImage);
	cv::Mat resultMat(srcImage.size(), srcImage.type());
	// 遍历图像，生成LBP特征
	for (int y = 1; y < nRows - 1; y++){
		for (int x = 1; x < nCols - 1; x++){
			// 定义邻域
			uchar neighbor[8] = { 0 };
			neighbor[0] = srcImage.at<uchar>(y - 1, x - 1);
			neighbor[1] = srcImage.at<uchar>(y - 1, x);
			neighbor[2] = srcImage.at<uchar>(y - 1, x + 1);
			neighbor[3] = srcImage.at<uchar>(y, x + 1);
			neighbor[4] = srcImage.at<uchar>(y + 1, x + 1);
			neighbor[5] = srcImage.at<uchar>(y + 1, x);
			neighbor[6] = srcImage.at<uchar>(y + 1, x - 1);
			neighbor[7] = srcImage.at<uchar>(y, x - 1);

			// 当前图像的处理中心
			uchar center = srcImage.at<uchar>(y, x);
			uchar temp = 0;
			// 计算LBP的值
			for (int k = 0; k < 8; k++){
				// 遍历中心点邻域
				temp += (neighbor[k] >= center)*(1 << k);
			}
			resultMat.at<uchar>(y, x) = temp;
		}
	}
	return resultMat;
}

// Haar特征提取 计算Haar特征
double HaarExtract(double const **image, int type_, cv::Rect roi){
	double value;
	double wh1, wh2;
	double bk1, bk2;
	int x = roi.x;
	int y = roi.y;
	int width = roi.width;
	int height = roi.height;
	switch (type_){
		// Haar水平边缘
	case 0:	// HaarHEdege
		wh1 = calcIntegral(image, x, y, width, height);
		bk1 = calcIntegral(image, x + width, y, width, height);
		value = (wh1 - bk1) / static_cast<double>(width * height);
		break;
		// Haar竖直边缘
	case 1:
		wh1 = calcIntegral(image, x, y, width, height);
		bk1 = calcIntegral(image, x, y + height, width, height);
		value = (wh1 - bk1) / static_cast<double>(width * height);
		break;
		// Haar水平线型
	case 2:
		wh1 = calcIntegral(image, x, y, width * 3, height);
		bk1 = calcIntegral(image, x + width, y, width, height);
		value = (wh1 - 3.0*bk1) / static_cast<double>(2 * width * height);
		break;
		// Haar垂直线型
	case 3:
		wh1 = calcIntegral(image, x, y, width, height * 3);
		bk1 = calcIntegral(image, x, y + height, width, height);
		value = (wh1 - 3.0*bk1) / static_cast<double>(2 * width*height);
		break;
		// Haar棋盘型
	case 4:
		wh1 = calcIntegral(image, x, y, width * 2, height * 2);
		bk1 = calcIntegral(image, x + width, y, width, height);
		bk2 = calcIntegral(image, x, y + height, width, height);
		value = (wh1 - 2.0*(bk1 + bk2)) / static_cast<double>(2 * width*height);
		break;
		// Haar中心包围型
	case 5:
		wh1 = calcIntegral(image, x, y, width * 3, height * 3);
		bk1 = calcIntegral(image, x + width, y + height, width, height);
		value = (wh1 - 9.0*bk1) / static_cast<double>(8 * width*height);
		break;
	}
	return value;
}

// 计算单窗口的积分图
double calcIntegral(double const** image, int x, int y, int width, int height){
	double term_1 = image[y - 1 + height][x - 1 + width];
	double term_2 = image[y - 1][x - 1];
	double term_3 = image[y - 1 + height][x - 1];
	double term_4 = image[y - 1][x - 1 + width];
	return (term_1 + term_2) - (term_3 + term_4);
}

// 在图片上添加文体辅助函数
void GetStringSize(HDC hDC, const char* str, int* w, int* h)
{
	SIZE size;
	GetTextExtentPoint32A(hDC, str, strlen(str), &size);
	if (w != 0) *w = size.cx;
	if (h != 0) *h = size.cy;
}

// 在图片上添加文字 带旋转
void drawString(Mat& dst, string text, Point org, double angle, Scalar color, int fontSize, bool italic, bool underline,
	bool black /* = false*/, const char* fontType){
	const char* str = string2pChar(text);
	int width = static_cast<int>(strlen(str) * fontSize * 0.5);
	int height = static_cast<int>(fontSize);

	Mat r = Mat::zeros(height, width, CV_8UC1);
	Mat s = Mat::zeros(height, width, dst.type());

	drawString(r, str, Point(0, 0), MC_WHITE, fontSize, italic, underline, black, fontType);
	drawString(s, str, Point(0, 0), color, fontSize, italic, underline, black, fontType);

	Mat mask = angleRotate(r, (int)angle);
	threshold(mask, mask, 77, 255, THRESH_BINARY);
	Mat colorString = angleRotate(s, (int)angle);
	GaussianBlur(colorString, colorString, Size(11, 11), 0.5, 0.5);

	Mat maskEdge;
	Canny(mask, maskEdge, 50, 150);
	maskEdge = getDilateImage(maskEdge, 5);

	Mat roi = dst(Rect(org.x - mask.cols / 2, org.y - mask.rows / 2, mask.cols, mask.rows));

	Mat gaussRoi;
	colorString.copyTo(roi, mask);

	GaussianBlur(roi, gaussRoi, Size(13, 13), 0.5, 0.5);
	gaussRoi.copyTo(roi, maskEdge);
	roi.copyTo(dst(Rect(org.x - mask.cols / 2, org.y - mask.rows / 2, mask.cols, mask.rows)));
}


// 在图片上添加文字 支持汉字 可换行 可调字体 可斜体
void drawString(Mat& dst, string text, Point org, Scalar color, int fontSize, bool italic, bool underline, bool black /* = false*/, const char* fontType)
{
	CV_Assert(dst.data != 0 && (dst.channels() == 1 || dst.channels() == 3));
	const char* str = string2pChar(text);

	int x, y, r, b;
	if (org.x > dst.cols || org.y > dst.rows) return;
	x = org.x < 0 ? -org.x : 0;
	y = org.y < 0 ? -org.y : 0;

	LOGFONTA lf;
	lf.lfHeight = -fontSize;
	lf.lfWidth = 0;
	lf.lfEscapement = 0;
	lf.lfOrientation = 0;
	lf.lfWeight = ((black) ? FW_BOLD : 5);

	//FW_BOLD

	lf.lfItalic = italic;  //斜体  
	lf.lfUnderline = underline;   //下划线  
	lf.lfStrikeOut = 0;
	lf.lfCharSet = DEFAULT_CHARSET;
	lf.lfOutPrecision = 0;
	lf.lfClipPrecision = 0;
	lf.lfQuality = PROOF_QUALITY;
	lf.lfPitchAndFamily = 0;
	strcpy(lf.lfFaceName, fontType);

	HFONT hf = CreateFontIndirectA(&lf);
	HDC hDC = CreateCompatibleDC(0);
	HFONT hOldFont = (HFONT)SelectObject(hDC, hf);

	int strBaseW = 0, strBaseH = 0;
	int singleRow = 0;
	char buf[1 << 12];
	strcpy(buf, str);

	//处理多行  
	{
		int nnh = 0;
		int cw, ch;
		const char* ln = strtok(buf, "\n");
		while (ln != 0)
		{
			GetStringSize(hDC, ln, &cw, &ch);
			strBaseW = max(strBaseW, cw);
			strBaseH = max(strBaseH, ch);

			ln = strtok(0, "\n");
			nnh++;
		}
		singleRow = strBaseH;
		strBaseH *= nnh;
	}

	if (org.x + strBaseW < 0 || org.y + strBaseH < 0)
	{
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hDC);
		return;
	}

	r = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
	b = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
	org.x = org.x < 0 ? 0 : org.x;
	org.y = org.y < 0 ? 0 : org.y;

	BITMAPINFO bmp = { 0 };
	BITMAPINFOHEADER& bih = bmp.bmiHeader;
	int strDrawLineStep = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

	bih.biSize = sizeof(BITMAPINFOHEADER);
	bih.biWidth = strBaseW;
	bih.biHeight = strBaseH;
	bih.biPlanes = 1;
	bih.biBitCount = 24;
	bih.biCompression = BI_RGB;
	bih.biSizeImage = strBaseH * strDrawLineStep;
	bih.biClrUsed = 0;
	bih.biClrImportant = 0;

	void* pDibData = 0;
	HBITMAP hBmp = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);

	CV_Assert(pDibData != 0);
	HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);

	//color.val[2], color.val[1], color.val[0]  
	SetTextColor(hDC, RGB(255, 255, 255));
	SetBkColor(hDC, 0);
	//SetStretchBltMode(hDC, COLORONCOLOR);  

	strcpy(buf, str);
	const char* ln = strtok(buf, "\n");
	int outTextY = 0;
	while (ln != 0)
	{
		TextOutA(hDC, 0, outTextY, ln, strlen(ln));
		outTextY += singleRow;
		ln = strtok(0, "\n");
	}
	uchar* dstData = (uchar*)dst.data;
	int dstStep = dst.step / sizeof(dstData[0]);
	unsigned char* pImg = (unsigned char*)dst.data + org.x * dst.channels() + org.y * dstStep;
	unsigned char* pStr = (unsigned char*)pDibData + x * 3;
	for (int tty = y; tty <= b; ++tty)
	{
		unsigned char* subImg = pImg + (tty - y) * dstStep;
		unsigned char* subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
		for (int ttx = x; ttx <= r; ++ttx)
		{
			for (int n = 0; n < dst.channels(); ++n){
				double vtxt = subStr[n] / 255.0;
				int cvv = (int)(vtxt * color.val[n] + (1 - vtxt) * subImg[n]);
				subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
			}

			subStr += 3;
			subImg += dst.channels();
		}
	}

	SelectObject(hDC, hOldBmp);
	SelectObject(hDC, hOldFont);
	DeleteObject(hf);
	DeleteObject(hBmp);
	DeleteDC(hDC);
}

// 两点间绘制虚线
void dottedLine(Mat& src, Point A, Point B, Scalar color, int thickness){
	double distTotal = dist(A, B);
	double distDelta = distTotal / 10;
	distDelta = (distDelta > 7.0) ? 7.0 : distDelta;
	int N = (int)floor(distTotal / distDelta);
	double stepX = (B.x - A.x) / N;
	double stepY = (B.y - A.y) / N;
	for (int i = 0; i <= N; i++){
		if (i % 2 == 0){
			if (A.x + stepX * (i + 1) < B.x || A.y + stepY * (i + 1) < B.y){
				line(src, Point(A.x + stepX*i, A.y + stepY*i), Point(A.x + stepX*(i + 1), A.y + stepY*(i + 1)), color, thickness, 8, 0);
			}
			else {
				line(src, Point(A.x + stepX*i, A.y + stepY*i), B, color, thickness, 8, 0);
				break;
			}
		}
	}
}

// 在图像的某一行绘制虚线
void drawDottedLineRow(Mat& src, int row, Scalar color, int thickness){
	CV_Assert(row < src.rows);
	CV_Assert(row >= 0);
	dottedLine(src, Point(0, row), Point(src.cols - 1, row), color, thickness);
}

// 在图像的某一列绘制虚线
void drawDottedLineCol(Mat& src, int col, Scalar color, int thickness){
	CV_Assert(col < src.cols);
	CV_Assert(col >= 0);
	dottedLine(src, Point(col, 0), Point(col, src.rows - 1), color, thickness);
}

// 双线性插值
Vec3d bilinearlInterpolation(Mat src, Point2f pt){

	//printf("width = %d, height = %d pt.x = %f pt.y = %f\n", src.cols, src.rows, pt.x, pt.y);
	CV_Assert(pt.x >= 0 && pt.y >= 0 && pt.x <= src.cols && pt.y <= src.rows);
	int x0 = (int)(floor(pt.x));
	int y0 = (int)(floor(pt.y));
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	Vec3d result = Vec3d(0, 0, 0);
	if (src.channels() == 3){
		if (x1 >= src.cols - 1 || y1 >= src.rows - 1)
		{
			result[0] = static_cast<double>(src.at<Vec3b>((int)y0, (int)x0)[0]);
			result[1] = static_cast<double>(src.at<Vec3b>((int)y0, (int)x0)[1]);
			result[2] = static_cast<double>(src.at<Vec3b>((int)y0, (int)x0)[2]);
			return result;
		}
		for (int i = 0; i < 3; i++){
			double y_up = (double)(src.at<Vec3b>(y0, x0)[i] * (x1 - pt.x) + src.at<Vec3b>(y0, x1)[i] * (pt.x - x0));
			double y_down = (double)(src.at<Vec3b>(y1, x0)[i] * (x1 - pt.x) + src.at<Vec3b>(y1, x1)[i] * (pt.x - x0));
			result[i] = y_up*(y1 - pt.y) + y_down*(pt.y - y0);
		}
		return result;
	}
	else if (src.channels() == 1){
		if (x1 >= src.cols - 1 || y1 >= src.rows - 1)
		{
			result[0] = src.at<uchar>((int)y0, (int)x0);
			return result;
		}

		double y_up = (double)(src.at<uchar>(y0, x0)*(x1 - pt.x) + src.at<uchar>(y0, x1)*(pt.x - x0));
		//printf("y_up = %lf\n", y_up);
		double y_down = (double)(src.at<uchar>(y1, x0)*(x1 - pt.x) + src.at<uchar>(y1, x1)*(pt.x - x0));
		//printf("y_down = %lf\n", y_down);
		result[0] = y_up*(y1 - pt.y) + y_down*(pt.y - y0);
		return result;
	}
	else {
		return result;
	}
}

// 计算对比度 返回Mat记录各点的对比度 double
Mat calculateContrast(Mat src, bool useD){
	Mat img_contrast;

	Mat tmpMat, img_gray;
	src.convertTo(tmpMat, CV_8UC3, 255);
	cv::cvtColor(tmpMat, img_gray, CV_BGR2GRAY);

	img_gray.convertTo(img_gray, CV_64FC1, 1 / 255.0);

	Mat h = (Mat_<double>(3, 3) << 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0);
	filter2D(img_gray, img_contrast, CV_64FC1, h, Point(-1, -1), 0.0, BORDER_REPLICATE);
	img_contrast = abs(img_contrast);

	return img_contrast;
}

// 计算饱和度 返回Mat记录各点的饱和度 double
Mat calculateSaturate(Mat src, bool useD){
	CV_Assert(src.channels() == 3);

	Mat result(src.size(), CV_64FC1);

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			//饱和度计算为颜色通道的标准差
			double R = src.at<Vec3b>(i, j)[2];
			double G = src.at<Vec3b>(i, j)[1];
			double B = src.at<Vec3b>(i, j)[0];
			double mu = (R + G + B) / 3.0;

			result.at<double>(i, j) = sqrt(((R - mu)*(R - mu) + (G - mu)*(G - mu) + (B - mu)*(B - mu)) / 3.0);
		}
	}
	if (!useD)
		convertScaleAbs(result, result);  //取abs绝对值
	return result;
}

// 计算曝光时间性能
Mat calculateWellExpose(Mat src)
{
	double sig = 0.2;
	int imgs_row = src.rows; //行
	int imgs_col = src.cols; //列

	Mat img_wellExposedness(imgs_row, imgs_col, CV_64F);
	Mat srcD;
	src.convertTo(srcD, CV_64FC3);
	double * p = srcD.ptr<double>(0);
	double * pd = img_wellExposedness.ptr<double>(0);

	for (int r = 0; r < imgs_row * imgs_col; r++){
		//高斯函数计算wellExposedness
		double IB = (*p++);
		double IG = (*p++);
		double IR = (*p++);

		double R = exp(-0.5 * ((IR - 0.5)*(IR - 0.5)) / (sig * sig));
		double G = exp(-0.5 * ((IG - 0.5)*(IG - 0.5)) / (sig * sig));
		double B = exp(-0.5 * ((IB - 0.5)*(IB - 0.5)) / (sig * sig));

		*pd++ = R * G * B;
	}
	return img_wellExposedness;
}

// 高斯金字塔
vector<Mat> gaussian_pyramid(Mat src, int nlev){
	int r = src.rows;
	int c = src.cols;

	Mat gray = GrayTrans(src);

	// compute the highest possible pyramid
	if (nlev == -1)
		nlev = (int)floor(log((double)(min(r, c))) / log(2.0));


	cout << nlev << endl;

	// recursively build pyramid
	vector<Mat> pyr(nlev);
	pyr[0] = gray.clone();
	Mat J = gray.clone();

	//J.convertTo(J, CV_64F);
	//cout << J.elemSize() << endl;

	for (int l = 1; l <= nlev - 1; l++){
		int w = J.cols;
		int h = J.rows;
		pyrDown(J, J, Size(w / 2, h / 2));
		pyr[l] = J.clone();
	}
	return pyr;
}

// 拉普拉斯金字塔
vector<Mat> laplacian_pyramid(Mat src, int nlev){
	int r = src.rows;
	int c = src.cols;

	/*Mat gray = GrayTrans(src);
	imshow("gray", gray);*/
	// compute the highest possible pyramid
	if (nlev == -1)
		nlev = (int)floor(log((double)(min(r, c))) / log(2.0));


	cout << nlev << endl;

	// recursively build pyramid
	vector<Mat> pry(nlev);

	Mat J = src.clone();
	//J.convertTo(J, CV_64F);
	//cout << J.elemSize() << endl;
	cv::Mat pyrDownImage, pyrUpImage;

	for (int l = 0; l < nlev - 1; l++){
		// apply low pass filter, and downsample
		int w = J.cols;
		int h = J.rows;
		// 下采样过程
		pyrDown(J, pyrDownImage, cv::Size(w / 2, h / 2));

		pyrUp(pyrDownImage, pyrUpImage, J.size());
		//cout << J.size() << " "<< pyrUpImage.size() << endl;
		//resize(J, J, pyrUpImage.size() );
		pry[l] = J - pyrUpImage;
		pyrDownImage.copyTo(J);
	}
	pry[nlev - 1] = J.clone();
	return pry;
}

// 拉普拉斯金字塔重建
Mat reconstruct_laplacian_pyramid(vector<Mat> pyr){
	int r = pyr[0].rows;
	int c = pyr[0].cols;

	int nlev = pyr.size();
	// start with low pass residual

	Mat R = pyr[nlev - 1];
	for (int l = nlev - 2; l >= 0; l--){
		Mat pyrUpImage;
		int w = R.cols;
		int h = R.rows;
		pyrUp(R, pyrUpImage, pyr[l].size());
		R = pyr[l] + pyrUpImage;
	}
	return R;
}

// 频域滤波 低通滤波 参数中的value 用于保存计算后的数值结果  返回值可直接显示出来CV_8UC1 
Mat getDFTBlur(Mat img, Mat& value){
	cv::Mat dftInput1, dftImage1, inverseDFT;
	img.convertTo(dftInput1, CV_64F);
	cv::dft(dftInput1, dftImage1, cv::DFT_COMPLEX_OUTPUT);	 // Applying DFT

	//cout << dftImage1 << endl;

	//生成频域滤波核
	Mat gaussianBlur = Mat::zeros(img.rows, img.cols, CV_64FC2);
	double D0 = 2 * 10000 * 10000;

	for (int i = 0; i < img.rows; i++)
	{
		double *q = gaussianBlur.ptr<double>(i);
		for (int j = 0; j < img.cols; j++)
		{
			double d = pow((double)(i - img.rows / 2), 2.0) + pow((double)(j - img.cols / 2), 2.0);
			/*if (d >= 70){
				q[2 * j] = 1.0;
				q[2 * j+1] = 1.0;
				continue;
				}*/
			q[2 * j] = 1 - expf(-d / D0);
			q[2 * j + 1] = 1 - expf(-d / D0);
		}
	}

	//	//高斯低通滤波， 高斯高通滤波
	multiply(dftImage1, gaussianBlur, gaussianBlur);


	// Reconstructing original imae from the DFT coefficients
	cv::idft(gaussianBlur, value, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT


	value.convertTo(inverseDFT, CV_8U);
	//cout << inverseDFTconverted.channels() << endl;

	Mat result = getContrastStretchImage(value);
	return result;
}

// 【绘制】靶标图像 给定靶标图像大小和格子的像素数
Mat drawChessboardImage(Size size, int gridWidth, int gridHeight, int offsetX, int offsetY, bool LeftTopBlack){
	Mat result = Mat::zeros(size, CV_8UC1);
	if (gridHeight == 0)
		gridHeight = gridWidth;
	for (int i = 0; i < size.height; i++){
		for (int j = 0; j < size.width; j++){
			if ((((i + offsetY) / gridHeight) % 2 + ((j + offsetX) / gridWidth) % 2) % 2)
				result.at<uchar>(i, j) = 255;
		}
	}
	if (!LeftTopBlack)
		result = inverseColor4(result);
	return result;
}


// 【绘制】靶标图像 给定靶标图像大小和格子数
Mat drawChessboardImage(Size size, Size gridCount, int offsetX, int offsetY, bool LeftTopBlack){
	Mat result = Mat::zeros(size, CV_8UC1);
	int gridHeight = size.height / gridCount.height;
	int gridWidth = size.width / gridCount.width;
	if (gridHeight == 0)
		gridHeight = gridWidth;
	for (int i = 0; i < size.height; i++){
		for (int j = 0; j < size.width; j++){
			if ((((i + offsetY) / gridHeight) % 2 + ((j + offsetX) / gridWidth) % 2) % 2)
				result.at<uchar>(i, j) = 255;
		}
	}
	if (!LeftTopBlack)
		result = inverseColor4(result);
	return result;
}

// 【绘制】靶标图像 给定靶标图像大小和格子数
Mat drawChessboardImage(Size gridCount, int gridWidth, int gridHeight, bool LeftTopBlack){
	Mat result = Mat::zeros(gridCount.height*gridHeight, gridCount.width*gridWidth, CV_8UC1);

	for (int i = 0; i < result.rows; i++){
		for (int j = 0; j < result.cols; j++){
			if (((i / gridHeight) % 2 + (j / gridWidth) % 2) % 2)
				result.at<uchar>(i, j) = 255;
		}
	}
	if (!LeftTopBlack)
		result = inverseColor4(result);
	return result;
}

// 合并两幅图 默认为横向合并 左右拼接 左右合并 上下拼接 上下合并
Mat combine2Img(Mat A, Mat B, bool CmbHor){
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

// 【图像压缩】游程编码 对压缩二值图像非常管用
bool runLengthCoding(Mat img, string outputPath){
	Mat gray = GrayTrans(img);
	uchar *data = gray.ptr<uchar>(0);

	FILE*ofp;
	if ((ofp = fopen(outputPath.c_str(), "wb")) == NULL){
		cout << "无法创建文件！" << endl;
		return false;
	}

	unsigned long inputSize, output = 0;
	inputSize = img.rows * img.cols;
	fwrite(&img.cols, sizeof(int), 1, ofp);
	fwrite(&img.rows, sizeof(int), 1, ofp);
	// 判断首位是0还是255
	int flag = (data[0]) ? 1 : 0;
	fwrite(&flag, sizeof(int), 1, ofp);

	for (int i = 0; i < inputSize - 1;){
		int s = 1;
		while (1){
			if (data[i + 1] == data[i]){
				i++;
				s++;
			}
			else{
				i++;
				break;
			}
		}
		//cout << "s = " << s << endl;
		fwrite(&s, sizeof(int), 1, ofp);
	}
	fclose(ofp);
	return true;
}

// 【图像解压缩】游程编码解压缩
Mat runLengthDecompress(string filepath)
{
	FILE*ifp;
	if ((ifp = fopen(filepath.c_str(), "rb")) == NULL){
		cout << "无法打开文件！" << endl;
		return Mat();
	}

	int w, h, flag;
	fread(&w, sizeof(int), 1, ifp);
	fread(&h, sizeof(int), 1, ifp);
	fread(&flag, sizeof(int), 1, ifp);
	cout << "w = " << w << endl;
	cout << "h = " << h << endl;

	Mat result = Mat::zeros(h, w, CV_8UC1);
	uchar* data = result.ptr<uchar>(0);

	int value = (flag == 1) ? 255 : 0;

	int j = 0;
	while (!feof(ifp))
	{
		int temp = 0;
		fread(&temp, sizeof(int), 1, ifp);
		//printf("temp = %d\n", temp);
		memset(data + j, value, sizeof(uchar)*temp);
		value = 255 - value;
		j += temp;
	}
	//imshow("result", result);
	//waitKey(0);
	fclose(ifp);
	return result;
}

// 【图像压缩】JPEG图像压缩 输入为JPEG彩色图像
Mat JPEGCompress(Mat src, int level){
	CV_Assert(level > 0 && level <= 100);
	cout << "origin image size: " << src.dataend - src.datastart << endl;
	cout << "height: " << src.rows << endl << "width: " << src.cols << endl << "depth: " << src.channels() << endl;
	cout << "height*width*depth: " << src.rows*src.cols*src.channels() << endl << endl;

	// (1)jpeg压缩
	vector<uchar> buff;		// buff for coding
	vector<int> param = vector<int>(2);
	param[0] = CV_IMWRITE_JPEG_QUALITY;
	param[1] = level; // default(95)(0-100)

	imencode(".jpg", src, buff, param);

	cout << "coded file size(jpg): " << buff.size() << endl;	// 自动拟合大小

	Mat jpegimage = imdecode(Mat(buff), CV_LOAD_IMAGE_COLOR);

	double psnr = PSNR(src, jpegimage);
	double bpp = 8.0*buff.size() / (jpegimage.size().area());	// bit/pixel
	printf("quality:%03d, %.1fdB, %.2fbpp\n", level, psnr, bpp);

	return jpegimage;
}

// 寻找Mat的最大几个元素 TopEles  寻找前num个像素 vec[0]代表横坐标，vec[1]代表纵坐标,vec[3]代表值
vector<Vec3d> findTopElements(Mat input, int num, bool useFilter){
	vector<Vec3d> result;

	double Dthresh = 5;

	for (int i = 0; i < num; i++){
		//cout << "i = " << i << endl;
		Vec3d temp;
		Point loc;
		minMaxLoc(input, NULL, &temp[2], NULL, &loc);

		bool flag = false;
		for (int j = 0; j < result.size(); j++){
			double d = sqrt((loc.x - result[j][0])*(loc.x - result[j][0]) + (loc.y - result[j][1])*(loc.y - result[j][1]));
			//cout << "d = " << d << endl;
			if (d < Dthresh){
				flag = true;
				input.at<double>(loc.y, loc.x) = 0;
				break;
			}
		}

		if (flag){
			i--;
			continue;
		}

		cout << loc << endl;
		temp[0] = loc.x;
		temp[1] = loc.y;
		result.push_back(temp);
		input.at<double>(loc.y, loc.x) = 0;
	}
	return result;
}

// 区域生长 type 可为4或者8
Mat RegionGrow(Mat& src, Point seed, int type)
{
	CV_Assert(type == 4 || type == 8);
	vector<Point> pts;
	pts.push_back(seed);
	Mat mask = Mat::zeros(src.size(), CV_8UC1);

	while (pts.size() > 0){
		Point p = pts[pts.size() - 1];
		//cout << pts.size() << endl;
		bool flag = (mask.at<uchar>(p.y, p.x) == 255);

		if (p.y + 1 < src.rows)
			flag = flag && (mask.at<uchar>(p.y + 1, p.x) == 255 || src.at<uchar>(p.y + 1, p.x) == 255);
		if (p.y - 1 >= 0)
			flag = flag && (mask.at<uchar>(p.y - 1, p.x) == 255 || src.at<uchar>(p.y - 1, p.x) == 255);
		if (p.x + 1 < src.cols)
			flag = flag && (mask.at<uchar>(p.y, p.x + 1) == 255 || src.at<uchar>(p.y, p.x + 1) == 255);
		if (p.x - 1 >= 0)
			flag = flag && (mask.at<uchar>(p.y, p.x - 1) == 255 || src.at<uchar>(p.y, p.x - 1) == 255);
		////////////////////////////////////////////////////////////////////////////////////////////
		if (type == 8){
			if (p.y + 1 < src.rows && p.x + 1 < src.cols)
				flag = flag && (mask.at<uchar>(p.y + 1, p.x + 1) == 255 || src.at<uchar>(p.y + 1, p.x + 1) == 255);
			if (p.y - 1 >= 0 && p.x - 1 >= 0)
				flag = flag && (mask.at<uchar>(p.y - 1, p.x - 1) == 255 || src.at<uchar>(p.y - 1, p.x - 1) == 255);
			if (p.y + 1 < src.rows && p.x - 1 >= 0)
				flag = flag && (mask.at<uchar>(p.y + 1, p.x - 1) == 255 || src.at<uchar>(p.y + 1, p.x - 1) == 255);
			if (p.y - 1 >= 0 && p.x + 1 < src.cols)
				flag = flag && (mask.at<uchar>(p.y - 1, p.x + 1) == 255 || src.at<uchar>(p.y - 1, p.x + 1) == 255);
		}

		if (flag){
			pts.pop_back();
		}
		else {
			int n = grow(src, mask, p, pts, type);
			if (n == 0)
				pts.pop_back();
		}
	}

	bitwise_or(mask, src, mask);
	return mask;
}

// 区域生长 4邻域
int grow(Mat src, Mat& mask, Point p, vector<Point>& s, int type){
	CV_Assert(type == 4 || type == 8);
	mask.at<uchar>(p.y, p.x) = 255;
	int count = 0;
	if (p.y + 1 < src.rows){
		if (mask.at<uchar>(p.y + 1, p.x) == 0 && src.at<uchar>(p.y + 1, p.x) == 0){
			mask.at<uchar>(p.y + 1, p.x) = 255;
			s.push_back(Point(p.x, p.y + 1));
			count++;
		}
	}
	if (p.y - 1 >= 0){
		if (mask.at<uchar>(p.y - 1, p.x) == 0 && src.at<uchar>(p.y - 1, p.x) == 0){
			mask.at<uchar>(p.y - 1, p.x) = 255;
			s.push_back(Point(p.x, p.y - 1));
			count++;
		}
	}
	if (p.x + 1 < src.cols){
		if (mask.at<uchar>(p.y, p.x + 1) == 0 && src.at<uchar>(p.y, p.x + 1) == 0){
			mask.at<uchar>(p.y, p.x + 1) = 255;
			s.push_back(Point(p.x + 1, p.y));
			count++;
		}
	}
	if (p.x - 1 >= 0){
		if (mask.at<uchar>(p.y, p.x - 1) == 0 && src.at<uchar>(p.y, p.x - 1) == 0){
			mask.at<uchar>(p.y, p.x - 1) = 255;
			s.push_back(Point(p.x - 1, p.y));
			count++;
		}
	}
	//////////////////////////////////////////////////////////////////////////////////
	if (type == 8){
		if (p.y + 1 < src.rows && p.x + 1 < src.cols){
			if (mask.at<uchar>(p.y + 1, p.x + 1) == 0 && src.at<uchar>(p.y + 1, p.x + 1) == 0){
				mask.at<uchar>(p.y + 1, p.x + 1) = 255;
				s.push_back(Point(p.x + 1, p.y + 1));
				count++;
			}
		}
		if (p.y - 1 >= 0 && p.x + 1 < src.cols){
			if (mask.at<uchar>(p.y - 1, p.x + 1) == 0 && src.at<uchar>(p.y - 1, p.x + 1) == 0){
				mask.at<uchar>(p.y - 1, p.x + 1) = 255;
				s.push_back(Point(p.x + 1, p.y - 1));
				count++;
			}
		}
		if (p.y + 1 < src.rows && p.x - 1 >= 0){
			if (mask.at<uchar>(p.y + 1, p.x - 1) == 0 && src.at<uchar>(p.y + 1, p.x - 1) == 0){
				mask.at<uchar>(p.y + 1, p.x - 1) = 255;
				s.push_back(Point(p.x - 1, p.y + 1));
				count++;
			}
		}
		if (p.y - 1 >= 0 && p.x - 1 >= 0){
			if (mask.at<uchar>(p.y - 1, p.x - 1) == 0 && src.at<uchar>(p.y - 1, p.x - 1) == 0){
				mask.at<uchar>(p.y - 1, p.x - 1) = 255;
				s.push_back(Point(p.x - 1, p.y - 1));
				count++;
			}
		}
	}
	return count;
}

// 将图片四等分
vector<Mat> divideMatInto4(Mat input, Mat& draw, Rect rect){
	CV_Assert(input.rows == input.cols);
	CV_Assert(is2ofIntPow(input.rows));
	//cout << rect << endl;
	vector<Mat> result;
	int s = input.rows / 2;
	Mat m1 = input(Rect(0, 0, s, s)).clone();
	Mat m2 = input(Rect(s, 0, s, s)).clone();
	Mat m3 = input(Rect(0, s, s, s)).clone();
	Mat m4 = input(Rect(s, s, s, s)).clone();
	result.push_back(m1);
	result.push_back(m2);
	result.push_back(m3);
	result.push_back(m4);
	Mat D = draw(rect);
	D.col(s).setTo(MC_WHITE);
	D.row(s).setTo(MC_WHITE);
	return result;
}

// 四叉树分解
void quadtreeSubdivision(Mat input, Mat& draw, Mat cur, int divTimes, int minSize, double thresh,
	Rect rect){
	if (divTimes == 0){
		draw = Mat::zeros(input.size(), CV_8UC3);
		rect = Rect(0, 0, input.cols, input.rows);
	}
	if (cur.rows < minSize)
		return;
	vector<Mat> m = divideMatInto4(cur, draw, rect);
	//showManyImages(m, Size(200, 200));
	//imshow("draw", draw);
	//waitKey(0);
	divTimes++;
	//cout << "divTimes = " << divTimes << endl;
	for (int i = 0; i < 4; i++){
		double tmpV = getGrayRange(m[i]);
		//cout << "gray range: m[" << i << "]" << tmpV << endl;
		if (tmpV >thresh && m[i].rows > minSize){
			//printf("分解m[%d]\n", i);
			quadtreeSubdivision(input, draw, m[i], divTimes, minSize, thresh, Rect(rect.tl().x + (i % 2)*m[i].rows, rect.tl().y + (i / 2)*m[i].cols, rect.width / 2, rect.height / 2));
		}
	}
}

// 灰度极差  灰度最大值减去灰度最小值
double getGrayRange(Mat input){
	CV_Assert(input.data != NULL);
	Mat img = GrayTrans(input);

	double minV, maxV;
	minMaxLoc(img, &minV, &maxV, NULL, NULL);
	return (maxV - minV);
}

// 灰度共生矩阵/灰度共现矩阵
// 水平方向 GLCM_HOR 0
// 竖直方向 GLCM_VER 1
// 左斜方向 GLCM_TL 2
// 右斜方向 GLCM_TR 3
// 灰度共生矩阵、灰度共现矩阵 Gray-level co-occurrence matrix
Mat getGLCM(Mat input, int type, int grayLevel){
	Mat gray = GrayTrans(input);
	Mat result = Mat::zeros(grayLevel, grayLevel, CV_32SC1);

	if (type == GLCM_HOR){
		for (int i = 0; i < gray.rows; i++){
			uchar *data = gray.ptr<uchar>(i);
			for (int j = 0; j < gray.cols - 1; j++){
				int m = (int)(data[j] / 256.0*grayLevel);
				int n = (int)(data[j + 1] / 256.0*grayLevel);
				result.at<int>(m, n)++;
			}
		}
	}
	else if (type == GLCM_VER){
		for (int i = 0; i < gray.rows - 1; i++){
			uchar *data = gray.ptr<uchar>(i);
			for (int j = 0; j < gray.cols; j++){
				int m = (int)(data[j] / 256.0*grayLevel);
				int n = (int)(data[j + gray.cols] / 256.0*grayLevel);
				result.at<int>(m, n)++;
			}
		}
	}
	else if (type == GLCM_TL){
		for (int i = 1; i < gray.rows; i++){
			uchar *data = gray.ptr<uchar>(i);
			for (int j = 1; j < gray.cols; j++){
				int m = (int)(data[j] / 256.0*grayLevel);
				int n = (int)(data[j - gray.cols - 1] / 256.0*grayLevel);
				result.at<int>(m, n)++;
			}
		}
	}
	else if (type == GLCM_TR){
		for (int i = 1; i < gray.rows; i++){
			uchar *data = gray.ptr<uchar>(i);
			for (int j = 0; j < gray.cols - 1; j++){
				int m = (int)(data[j] / 256.0*grayLevel);
				int n = (int)(data[j - gray.cols + 1] / 256.0*grayLevel);
				result.at<int>(m, n)++;
			}
		}
	}
	convertScaleAbs(result, result);
	result = getContrastStretchImage(result);
	return result;
}

//把图像归一化为0-255，便于显示
// 可以对图像进行拉伸处理后在进行归一化处理 默认不进行拉伸
Mat norm_0_255(const Mat& src, cv::Size s/* = cv::Size(0, 0)*/)
{
	if (s.width != 0 && s.height != 0)
		resize(src, src, s);
	Mat dst;
	switch (src.channels())
	{
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}


// 绘制某行的灰度变化曲线 截面图
void drawGrayCurveInRow(Mat src, int rowIndex){
	CV_Assert(src.data != NULL);
	CV_Assert(src.rows > rowIndex && rowIndex >= 0);
	Mat gray = GrayTrans(src);
	gray.convertTo(gray, CV_8UC1);
	uchar *data = src.ptr<uchar>(rowIndex);

	Mat result = Mat::zeros(500, src.cols, CV_8UC3);
	for (int i = 0; i < src.cols - 1; i++){
		Point pt1 = Point(i, 2 * (*data++));
		Point pt2 = Point(i + 1, 2 * (*data));
		line(result, pt1, pt2, MC_YELLOW, 1, 8, 0);
	}
	imshow("灰度变化曲线", result);
	waitKey(0);
}


// 绘制某列的灰度变化曲线 界面图
void drawGrayCurveInCol(Mat src, int colIndex){
	CV_Assert(src.data != NULL);
	CV_Assert(src.cols > colIndex && colIndex >= 0);
	Mat gray = GrayTrans(src);
	gray.convertTo(gray, CV_8UC1);
	Mat c = src.col(colIndex).clone();
	uchar *data = c.ptr<uchar>(0);

	Mat result = Mat::zeros(src.rows, 500, CV_8UC3);
	for (int i = 0; i < src.rows - 1; i++){
		Point pt1 = Point(2 * (*data++), i);
		Point pt2 = Point(2 * (*data), i + 1);
		line(result, pt1, pt2, MC_YELLOW, 1, 8, 0);
	}
	imshow("灰度变化曲线", result);
	waitKey(0);
}

// 图像分割 编号默认按行排列
bool ImgSegm(Mat src, string outputPath, Size size, string prefix){
	if (size.width > src.cols || size.height > src.rows){
		cout << "分割图像大小应不大于源图像大小!" << endl;
		return false;
	}

	int curR = 0, curC = 0;
	int i = 1;
	while (1){
		Mat tmp;
		//cout << Rect(curC, curR, size.width, size.height) << endl;
		src(Rect(curC, curR, size.width, size.height)).copyTo(tmp);
		imwrite(outputPath + "\\" + prefix + int2string(i) + ".jpg", tmp);

		curC += size.width;

		if (curC + size.width > src.cols){
			curR += size.height;
			curC = 0;
		}
		i++;
		if (curR + size.height > src.rows){
			break;
		}

	}
}


// 批量修改文件夹内所有后缀为suffix图像的尺寸 批量修改尺寸 
bool resizeImgsInFolder(string folderPath, Size dstSize, string dstFolder, string suffix,
	string dstSuffix /*= ""*/, bool gray/* = false*/){	
	ofstream hello(folderPath);	// 此处为相对路径，也可以改为绝对路径 
	CFileFind finder;
	CString path = string2CString(folderPath+"\\*."+suffix);
	BOOL bContinue = finder.FindFile(path);
	
	string src_route_head = folderPath + "\\";  // 源图像的路径头
	string dst_route_head = dstFolder + "\\";	// 目标图像的路径头
	string filesuffix = (dstSuffix == "") ? suffix : dstSuffix;

	if (!isFolderValidate(dstFolder)){
		_mkdir(string2pChar(dstFolder));
	}

	Mat src, dst;

	while (bContinue){
		bContinue = finder.FindNextFileW();
		string name = CString2string(finder.GetFileName());

		if (!finder.IsDirectory()){
			string SourceRoute = src_route_head + name;

			src = imread(SourceRoute);   // 载入图像
			if (gray)
				src = GrayTrans(src);
			resize(src, dst, dstSize);
			imwrite(dst_route_head + getFileName(name) + "."+filesuffix, dst);        // 保存dst
		}
	}

	hello.close();
	return true;
}

//// 批量修改文件夹内所有后缀为suffix图像的尺寸 批量修改尺寸 
//bool resizeImgsInFolder(CString folderPath, Size dstSize, CString dstFolder, CString suffix,
//	CString dstSuffix /*= ""*/, bool gray/* = false*/){
//	WIN32_FIND_DATA p;   // 指向一个用于保存文件信息的结构体
//	HANDLE h = FindFirstFile(folderPath + _T("\\*.") + suffix, &p);		// FindFirstFile的返回值是一个句柄，第二个参数p是采用引用的方式，也就是
//
//	if (h == NULL){
//		cout << "hello" << endl;
//		return false;
//	}
//	// 说当这句话执行完毕后p就指向该文件*.jpg
//
//	// 由于p的成员变量只有文件名，而无文件路径，所以必须加上路镜头
//	string src_route_head = CString2string(folderPath + _T("\\"));  // 源图像的路径头
//	string dst_route_head = CString2string(dstFolder + _T("\\"));	// 目标图像的路径头
//	string SourceRoute = src_route_head + CString2string(p.cFileName);   // 包含了路镜头和文件名的全路径
//	string filename = CString2string(getFileName(p.cFileName));
//	string filesuffix = CString2string(getFileSuffix(p.cFileName));
//	if (dstSuffix != ""){
//		filesuffix = CString2string(dstSuffix);
//	}
//	string DestRoute = dst_route_head + filename + "." + filesuffix;
//	cout << "SourceRoute = " << SourceRoute << endl;
//
//	Mat src = imread(SourceRoute);// 载入图像
//	if (gray)
//		src = GrayTrans(src);
//	Mat dst = Mat::zeros(dstSize, src.type());   // 分配一个dstSize大小的目标图像，resize后的结果将存放在这里    
//
//	resize(src, dst, dstSize);
//	imwrite(DestRoute, dst);		// 保存dst
//
//	// 到目前为止，我们就已经完成了对目标文件夹中第一幅图像的resize处理与保存，接下来让该文件中其余图像也被处理
//
//	while (FindNextFile(h, &p))  // p指针不断后移，寻找下一个、下下一个*.jpg
//	{
//		SourceRoute = src_route_head + CString2string(p.cFileName);
//		src = imread(SourceRoute, 0);   // 载入图像
//
//		resize(src, dst, dstSize);
//
//		filename = CString2string(getFileName(p.cFileName));
//		dst_route_head + filename + "." + filesuffix;
//		imwrite(DestRoute, dst);        // 保存dst
//	}
//
//	return true;
//}



// Floyd-Steinberg 抖动算法
Mat floydSteinbergDithering(Mat input){
	if (!input.data){
		cout << "输入图像无数据！" << endl;
		return Mat();
	}
	Mat gray = GrayTrans(input);
	gray.convertTo(gray, CV_64FC1);

	// 加网标识
	uchar A = 255;
	uchar B = 0;
	// 保存加网图像
	Mat C = Mat::zeros(gray.size(), CV_8UC1);

	// 加网
	for (int i = 0; i < gray.rows; i++){
		for (int j = 0; j < gray.cols; j++){
			double error = 0;
			double I = gray.at<double>(i, j);
			if (I > 128){				// 加网阈值 
				C.at<uchar>(i, j) = A;
				error = I - 255.0;		// 保留误差
			}
			else {
				C.at<uchar>(i, j) = B;
				error = I;				// 误差
			}
			if (j < gray.cols - 1){
				double temp = gray.at<double>(i, j + 1);
				temp = temp + error*(7.0 / 16.0);			// 向右扩散误差的7/16
				gray.at<double>(i, j + 1) = temp;
			}
			if (i < gray.rows - 1){
				double temp = gray.at<double>(i + 1, j);
				temp = temp + error*(5.0 / 16.0);			// 向下扩散误差的5/16
				gray.at<double>(i + 1, j) = temp;
				if (j < gray.cols - 1){
					temp = gray.at<double>(i + 1, j + 1);
					temp = temp + error*(1.0 / 16.0);		// 向右下扩散误差的1/16
					gray.at<double>(i + 1, j + 1) = temp;
				}
				if (j>0){
					temp = gray.at<double>(i + 1, j - 1);
					temp = temp + error*(3.0 / 16.0);			// 向左下扩散误差的3/16
					gray.at<double>(i + 1, j - 1) = temp;
				}
			}
		}
	}
	return C;
}

// 求梯度图像 数据类型为CV_32FC1
bool Gradient(Mat src, Mat& dst){
	Mat sobelX = Mat::zeros(src.size(), CV_32FC1);
	Mat sobelY = Mat::zeros(src.size(), CV_32FC1);
	float *theimage1 = sobelX.ptr<float>(0);
	float *theimage2 = sobelY.ptr<float>(0);
	Sobel(src, sobelX, CV_32FC1, 1, 0, 3);
	Sobel(src, sobelY, CV_32FC1, 0, 1, 3);

	int width = src.cols;
	int height = src.rows;

	dst = Mat::zeros(src.size(), CV_32FC1);
	float *tempout = dst.ptr<float>(0);
	for (int i = 0; i < height*width; i++){
		*tempout++ = sqrt((*theimage1)*(*theimage1) + (*theimage2)*(*theimage2));
		theimage1++;
		theimage2++;
	}
	theimage1 = NULL;
	theimage2 = NULL;
	return true;
}

// 显示空间点坐标数值 数字 
Mat showPt3d(Point3d pt){
	Mat result(600, 900, CV_8UC1);
	drawString(result, "X : " + double2string(pt.x, 5), Point(0, 100), MC_WHITE, 100, false, false, "DS-Digital");
	drawString(result, "Y : " + double2string(pt.y, 5), Point(0, 300), MC_WHITE, 100, false, false, "DS-Digital");
	drawString(result, "Z : " + double2string(pt.z, 5), Point(0, 500), MC_WHITE, 100, false, false, "DS-Digital");
	return result;
}

// 生成加密图像 LUT的数据类型为CV_32SC1
Mat getEncryptImage(Mat input, Mat &lut, int type){
	Mat result = Mat::zeros(input.size(), input.type());
	if (type == ENCRYPT_ROWS){
		vector<int> tmp = randomVecInt(input.rows - 1);
		lut = Mat::zeros(input.rows, 1, CV_32SC1);
		for (int i = 0; i < input.rows; i++){
			input.row(tmp[i]).copyTo(result.row(i));
			lut.ptr<int>(0)[i] = tmp[i];
		}
		return result;
	}
	else if (type == ENCRYPT_COLS){
		vector<int> tmp = randomVecInt(input.cols - 1);
		lut = Mat::zeros(input.cols, 1, CV_32SC1);
		for (int i = 0; i < input.cols; i++){
			input.col(tmp[i]).copyTo(result.col(i));
			lut.ptr<int>(0)[i] = tmp[i];
		}
		return result;
	}
	else if (type == ENCRYPT_ROWS_AND_COLS){
		lut = Mat::zeros(input.rows + input.cols, 1, CV_32SC1);
		Mat lut1;
		Mat t = getEncryptImage(input, lut1, ENCRYPT_ROWS);
		lut1.copyTo(lut(Rect(0, 0, 1, input.rows)));
		Mat result = getEncryptImage(t, lut1, ENCRYPT_COLS);
		lut1.copyTo(lut(Rect(0, input.rows, 1, input.cols)));
		return result;
	}
	else if (type == ENCRYPT_ALL_PIXELS){
		lut = Mat::zeros(input.size().area()*input.channels(), 1, CV_32SC1);
		vector<int> tmp = randomVecInt(input.size().area()*input.channels());
		uchar *data = result.ptr<uchar>(0);
		for (int i = 0; i < input.size().area()*input.channels(); i++){
			*data++ = input.ptr<uchar>(0)[tmp[i]];
			lut.ptr<int>(0)[i] = tmp[i];
		}
		return result;
	}
	else if (type == ENCRYPT_ADD_RANDOM){ //需先进行灰度化处理 lut也为CV_64FC1
		Mat gray = GrayTrans(input);
		gray.convertTo(gray, CV_64FC1);

		lut = Mat::zeros(input.size(), CV_64FC1);
		unsigned int optional_seed = (unsigned int)time(NULL);
		cv::RNG rng(optional_seed);
		Mat a = (Mat_<double>(1, 1) << 0);
		Mat b = (Mat_<double>(1, 1) << 255);
		rng.fill(lut, cv::RNG::UNIFORM, a, b);

		result.convertTo(result, CV_64FC1);
		result = lut*0.9 + gray*0.1;
		return result;
	}
}

// 解密图像 LUT的数据类型为CV_32SC1
Mat getDecodeImage(Mat input, Mat lut, int type){
	Mat result = Mat::zeros(input.size(), input.type());
	if (type == ENCRYPT_ROWS){
		for (int i = 0; i < input.rows; i++){
			input.row(i).copyTo(result.row(lut.ptr<int>(0)[i]));
		}
		return result;
	}
	else if (type == ENCRYPT_COLS){
		for (int i = 0; i < input.cols; i++){
			input.col(i).copyTo(result.col(lut.ptr<int>(0)[i]));
		}
		return result;
	}
	else if (type == ENCRYPT_ROWS_AND_COLS){
		Mat lutC = lut(Rect(0, input.rows, 1, input.cols)).clone();
		Mat lutR = lut(Rect(0, 0, 1, input.rows)).clone();
		Mat x = getDecodeImage(input, lutC, ENCRYPT_COLS);
		result = getDecodeImage(x, lutR, ENCRYPT_ROWS);
		return result;
	}
	else if (type == ENCRYPT_ALL_PIXELS){
		uchar* data = input.ptr<uchar>(0);
		int* it = lut.ptr<int>(0);
		for (int i = 0; i < input.size().area()*input.channels(); i++){
			result.ptr<uchar>(0)[*it++] = *data++;
		}
		return result;
	}
	else if (type == ENCRYPT_ADD_RANDOM){	// 只能对灰度CV_64FC1图像进行解密 lut也为CV_64FC1
		Mat tmp = (input - 0.9*lut)*10.0;
		tmp.convertTo(result, CV_8UC1);
		return result;
	}
}

// 二维离散小波变换（单通道浮点图像）
// 输入图像要求必须是单通道浮点图像，对图像大小也有要求
// （1层变换：w, h必须是2的倍数；2层变换：w, h必须是4的倍数；3层变换：w, h必须是8的倍数......）
Mat DWT(Mat src, int nLayer)
{
	Mat input = src.clone();

	if (input.channels() == 1 && (input.type() == CV_32FC1 || input.type() == CV_32F)
		&& ((input.cols >> nLayer) << nLayer) == input.cols
		&& ((input.rows >> nLayer) << nLayer) == input.rows){
		int     i, x, y, n;
		float   fValue = 0;
		float   fRadius = sqrt(2.0f);
		int     nWidth = input.cols;
		int     nHeight = input.rows;
		int     nHalfW = nWidth / 2;
		int     nHalfH = nHeight / 2;
		float **pData = new float*[input.rows];
		float  *pRow = new float[input.cols];
		float  *pColumn = new float[input.rows];
		for (i = 0; i < input.rows; i++)
		{
			pData[i] = input.ptr<float>(i);
		}
		// 多层小波变换
		for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2)
		{
			// 水平变换
			for (y = 0; y < nHeight; y++)
			{
				// 奇偶分离
				memcpy(pRow, pData[y], sizeof(float)* nWidth);
				for (i = 0; i < nHalfW; i++)
				{
					x = i * 2;
					pData[y][i] = pRow[x];
					pData[y][nHalfW + i] = pRow[x + 1];
				}
				// 提升小波变换
				for (i = 0; i < nHalfW - 1; i++)
				{
					fValue = (pData[y][i] + pData[y][i + 1]) / 2;
					pData[y][nHalfW + i] -= fValue;
				}
				fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
				pData[y][nWidth - 1] -= fValue;
				fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
				pData[y][0] += fValue;
				for (i = 1; i < nHalfW; i++)
				{
					fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
					pData[y][i] += fValue;
				}
				// 频带系数
				for (i = 0; i < nHalfW; i++)
				{
					pData[y][i] *= fRadius;
					pData[y][nHalfW + i] /= fRadius;
				}
			}
			// 垂直变换
			for (x = 0; x < nWidth; x++)
			{
				// 奇偶分离
				for (i = 0; i < nHalfH; i++)
				{
					y = i * 2;
					pColumn[i] = pData[y][x];
					pColumn[nHalfH + i] = pData[y + 1][x];
				}
				for (i = 0; i < nHeight; i++)
				{
					pData[i][x] = pColumn[i];
				}
				// 提升小波变换
				for (i = 0; i < nHalfH - 1; i++)
				{
					fValue = (pData[i][x] + pData[i + 1][x]) / 2;
					pData[nHalfH + i][x] -= fValue;
				}
				fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
				pData[nHeight - 1][x] -= fValue;
				fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
				pData[0][x] += fValue;
				for (i = 1; i < nHalfH; i++)
				{
					fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
					pData[i][x] += fValue;
				}
				// 频带系数
				for (i = 0; i < nHalfH; i++)
				{
					pData[i][x] *= fRadius;
					pData[nHalfH + i][x] /= fRadius;
				}
			}
		}
		delete[] pData;
		delete[] pRow;
		delete[] pColumn;
	}
	else if (input.channels() == 3 && input.type() == CV_32FC3){
		vector<Mat> vec_img;
		split(input, vec_img);
		for (int ch = 0; ch < 3; ch++){
			vec_img[ch] = DWT(vec_img[ch], nLayer);
		}
		Mat result;
		merge(vec_img, result);
		return result;
	}

	return input;
}

// 二维离散小波恢复（单通道浮点图像）
Mat IDWT(Mat src, int nLayer)
{
	Mat input = src.clone();

	if (input.channels() == 1 && (input.depth() == CV_32F || input.depth() == CV_32FC1) 
		&& ((input.cols >> nLayer) << nLayer) == input.cols
		&& ((input.rows >> nLayer) << nLayer) == input.rows)
	{
		int     i, x, y, n;
		float   fValue = 0;
		float   fRadius = sqrt(2.0f);
		int     nWidth = input.cols >> (nLayer - 1);
		int     nHeight = input.rows >> (nLayer - 1);
		int     nHalfW = nWidth / 2;
		int     nHalfH = nHeight / 2;
		float **pData = new float*[input.rows];
		float  *pRow = new float[input.cols];
		float  *pColumn = new float[input.rows];
		for (i = 0; i < input.rows; i++)
		{
			pData[i] = input.ptr<float>(i);
		}
		// 多层小波恢复
		for (n = 0; n < nLayer; n++, nWidth *= 2, nHeight *= 2, nHalfW *= 2, nHalfH *= 2)
		{
			// 垂直恢复
			for (x = 0; x < nWidth; x++)
			{
				// 频带系数
				for (i = 0; i < nHalfH; i++)
				{
					pData[i][x] /= fRadius;
					pData[nHalfH + i][x] *= fRadius;
				}
				// 提升小波恢复
				fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
				pData[0][x] -= fValue;
				for (i = 1; i < nHalfH; i++)
				{
					fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
					pData[i][x] -= fValue;
				}
				for (i = 0; i < nHalfH - 1; i++)
				{
					fValue = (pData[i][x] + pData[i + 1][x]) / 2;
					pData[nHalfH + i][x] += fValue;
				}
				fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
				pData[nHeight - 1][x] += fValue;
				// 奇偶合并
				for (i = 0; i < nHalfH; i++)
				{
					y = i * 2;
					pColumn[y] = pData[i][x];
					pColumn[y + 1] = pData[nHalfH + i][x];
				}
				for (i = 0; i < nHeight; i++)
				{
					pData[i][x] = pColumn[i];
				}
			}
			// 水平恢复
			for (y = 0; y < nHeight; y++)
			{
				// 频带系数
				for (i = 0; i < nHalfW; i++)
				{
					pData[y][i] /= fRadius;
					pData[y][nHalfW + i] *= fRadius;
				}
				// 提升小波恢复
				fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
				pData[y][0] -= fValue;
				for (i = 1; i < nHalfW; i++)
				{
					fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
					pData[y][i] -= fValue;
				}
				for (i = 0; i < nHalfW - 1; i++)
				{
					fValue = (pData[y][i] + pData[y][i + 1]) / 2;
					pData[y][nHalfW + i] += fValue;
				}
				fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
				pData[y][nWidth - 1] += fValue;
				// 奇偶合并
				for (i = 0; i < nHalfW; i++)
				{
					x = i * 2;
					pRow[x] = pData[y][i];
					pRow[x + 1] = pData[y][nHalfW + i];
				}
				memcpy(pData[y], pRow, sizeof(float)* nWidth);
			}
		}
		delete[] pData;
		delete[] pRow;
		delete[] pColumn;
	}
	else if (input.channels() == 3 && input.type() == CV_32FC3){
		vector<Mat> vec_img;
		split(input, vec_img);
		for (int ch = 0; ch < 3; ch++){
			vec_img[ch] = IDWT(vec_img[ch], nLayer);
		}
		Mat result;
		merge(vec_img, result);
		return result;
	}
	return input;
}

// 生成掩膜图像 灰度掩膜 [threshLow,threshHigh]  中间值取255，否则为0
// inverse可以取反 即中间值置零，否则为255
Mat generateGrayMask(Mat input, uchar threshLow, uchar threshHight, bool inverse/* = false*/){
	Mat mask = Mat::zeros(input.size(), CV_8UC1);
	Mat gray = GrayTrans(input);
	uchar * data = gray.ptr<uchar>(0);
	for (int i = 0; i < gray.total(); i++){
		if (!inverse){
			if (*data >= threshLow && *data <= threshHight){
				mask.ptr<uchar>(0)[i] = 255;
			}
		}
		else {
			if (*data < threshLow || *data > threshHight){
				mask.ptr<uchar>(0)[i] = 255;
			}
		}
		data++;
	}
	return mask;
}

// 图像的主成分分析 PCA变换
// number_principal_compent 保留最大的主成分数 默认为0 保留所有成分分量
Mat PCATrans(Mat src, int number_principal_compent/* = 0*/){
	src = GrayTrans(src);
	src.convertTo(src, CV_32FC1);
	PCA pca(src, Mat(), CV_PCA_DATA_AS_ROW, number_principal_compent);
	//cout << pca.mean<<endl; // 均值
	//cout << pca.eigenvalues << endl;// 特征值
	Mat dst = pca.project(src);   // 映射新空间
	dst = pca.backProject(dst);
	
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

// 制作GIF动画
void makeGIF(cv::Size size, string gifName, int fps/* = 15*/){
	// 【文件夹】选择一个文件夹
	string path = CString2string(selectFolder());
	renameFilesInFolder(path, "std", false);
	Sleep(50);
	resizeImgsInFolder(path, size, path + "\\gif", "bmp", "bmp");
	resizeImgsInFolder(path, size, path + "\\gif", "jpg", "bmp");
	resizeImgsInFolder(path, size, path + "\\gif", "png", "bmp");
	if (gifName == "")
		makeGIF(path + "\\gif", "std", path + "\\hello.gif", size, fps);
	else 
		makeGIF(path + "\\gif", "std", path + "\\" + gifName, size, fps);
	deleteFolder(path + "\\gif");
}

// 制作GIF动画
// 需要将C:\\FFMEPG\\bin文件夹中的exe文件都放在
// fps为帧率
void makeGIF(string folderPath, string prefix, string savePath, cv::Size size/* = cv::Size(0, 0)*/, int fps/* = 7*/){
	if (size.width == 0)
		system(string2pChar("ffmpeg -f image2 -framerate "+int2string(fps)+" -i " 
			+ folderPath + "\\"+prefix+"%d.bmp " + savePath));
	else 
		system(string2pChar("ffmpeg -s " + int2string(size.height) + "x" + int2string(size.width)
		+ " -f image2 -framerate " + int2string(fps)
		+ " -i " + folderPath + "\\" + prefix + "%d.bmp " + savePath));

	/*cout << "ffmpeg -s " + int2string(size.height) + "x" + int2string(size.width)
		+ " -f image2 -framerate " + int2string(fps)
		+ " -i " + folderPath + "\\" + prefix + "%d.bmp " + savePath << endl;*/
	//system(string2pChar("ffmpeg - f image2 - framerate 5 - i F:\std%d.bmp D:\c.gif"));
}

// 交换左上右下块 和 右上左下块
Mat fftshift(Mat input){
	//CV_Assert(input.rows % 2 == 0 && input.cols % 2 == 0);
	int cx = input.cols / 2;
	int cy = input.rows / 2;

	Mat dftResultImage = input.clone();

	Mat tmp;
	// Top-left——为每一个象限创建ROI
	Mat q0(dftResultImage, Rect(0, 0, cx, cy));
	// Top-Right
	Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
	// Bottom-Left
	Mat q2(dftResultImage, Rect(0, cy, cx, cy));
	// Bottom——Right
	Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
	// 交换象限 (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	// 交换象限(Top-Right with Bottom-Left)
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	return dftResultImage;
}


void initGDI(){
	
	
	
}

// 截屏函数
Mat screenshot(){
	int nWidth = GetSystemMetrics(SM_CXSCREEN);//得到屏幕的分辨率的x    
	int nHeight = GetSystemMetrics(SM_CYSCREEN);//得到屏幕分辨率的y    
	LPVOID    screenCaptureData = new char[nWidth*nHeight * 4];
	memset(screenCaptureData, 0, nWidth);
	
	// Get desktop DC, create a compatible dc, create a comaptible bitmap and select into compatible dc.    
	HDC hDDC = GetDC(GetDesktopWindow());//得到屏幕的dc    
	HDC hCDC = CreateCompatibleDC(hDDC);//    
	HBITMAP hBitmap = CreateCompatibleBitmap(hDDC, nWidth, nHeight);//得到位图    
	SelectObject(hCDC, hBitmap); //好像总得这么写。    

	BitBlt(hCDC, 0, 0, nWidth, nHeight, hDDC, 0, 0, SRCCOPY);

	GetBitmapBits(hBitmap, nWidth*nHeight * 4, screenCaptureData);//得到位图的数据，并存到screenCaptureData数组中。    
	Mat img = Mat::zeros(nHeight, nWidth, CV_8UC4);				 // 创建一个rgba格式的Mat，内容为空
	memcpy(img.data, screenCaptureData, nWidth*nHeight * 4);//这样比较浪费时间，但写的方便。这里必须得是*4，上面的链接写的是*3，这是不对的。    
	return img;
}

// 为图像添加透明度通道 输入为单通道或三通道图像
// alpha为透明度，1为不透明，0为透明
cv::Mat addAlphaChannel(cv::Mat input, double alpha){
	Mat result;
	vector<Mat> s;
	if (input.channels() == 1){
		Mat tmp = input.clone();
		tmp.setTo((uchar)(255 * alpha));
		s.push_back(tmp);
		merge(s, result);
	}
	else if (input.channels() == 3){		
		split(input, s);
		Mat tmp = s[0].clone();
		tmp.setTo((uchar)(255 * alpha));
		s.push_back(tmp);
		merge(s, result);
	}
	return result;
}

std::vector<float> getRowDataUchar(const cv::Mat &img, int rowId)
{
	rowId = min(max(0, rowId), img.rows-1);
	if (rowId < 0 || rowId >= img.rows)
	{
		printf("rowId = %d\n", rowId);
	}
	std::vector<float> result(img.cols);
	for (int i = 0; i < img.cols; i++)
		result[i] = img.at<uchar>(rowId, i);
	return result;
}

std::vector<float> getColDataUchar(const cv::Mat &img, int colId)
{
	colId = min(max(0, colId), img.cols - 1);
	if (colId < 0 || colId >= img.cols)
	{
		printf("colId = %d\n", colId);
	}
	//printf("img.rows = %d   colId = %d\n", img.rows, colId);
	std::vector<float> result(img.rows);
	for (int i = 0; i < img.rows; i++)
		result[i] = img.at<uchar>(i, colId);
	return result;
}

std::vector<float> getRowDataFloat(const cv::Mat &img, int rowId)
{
	if (rowId < 0 || rowId >= img.rows)
		return std::vector<float>();
	std::vector<float> result(img.cols);
	for (int i = 0; i < img.cols; i++)
		result[i] = img.at<float>(rowId, i);
	return result;
}

std::vector<float> getColDataFloat(const cv::Mat &img, int colId)
{
	if (colId < 0 || colId >= img.cols)
		return std::vector<float>();
	std::vector<float> result(img.rows);
	for (int i = 0; i < img.rows; i++)
		result[i] = img.at<float>(i, colId);
	return result;
}

//// 将输入图像延扩到最佳的尺寸 扩展图像 增加图像边框
//int nRows = getOptimalDFTSize(img.rows);
//int nCols = getOptimalDFTSize(img.cols);
//cv::Mat resultImage;
//// 把灰度图像放在左上角，向右边和下边扩展图像
//// 将添加的像素初始化为0
////cv::Mat resultImage;
//copyMakeBorder(img, resultImage, 0, nRows - img.rows, 0, nCols - img.cols,
//	BORDER_CONSTANT, Scalar::all(0));