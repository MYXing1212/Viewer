#include"stdafx.h"
//#include"stdafx.h"
#include"MyImage.h"
#include <direct.h>  

using namespace cv;

// DCT��������
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

CRITICAL_SECTION cs;						// �ٽ����ṹ����


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

	EnterCriticalSection(&cs);					// �����ٽ���
	string path = *(l->it + threadImgNum);
	vector<Mat>::iterator image = l->iMat + threadImgNum;
	vector<Mat>::iterator imageRGB = l->iMatRGB + threadImgNum;
	++threadImgNum;
	cout << path << " ����ɹ�!" << endl;
	LeaveCriticalSection(&cs);					// �뿪�ٽ���	

	*image = imread(path, 0);
	*imageRGB = imread(path);

	if (!image->data){
		printf("---ERROR!!!---\n����ͼ�����!!!\n");
		return false;
	}

	EnterCriticalSection(&cs);					// �����ٽ���
	threadEndNum++;
	//cout << "threadEndNum = " << threadEndNum << endl;
	LeaveCriticalSection(&cs);					// �뿪�ٽ���
	return 0;
}

// ͼ�����ݶȱ任
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



// ����ͼ��
bool loadImages(vector<Mat>& loadImages, vector<Mat>& loadImagesRGB, vector<CString>& FPths)
{
	vector<string> filepaths;
	InitializeCriticalSection(&cs);					// ��ʼ���ٽ���
	// ���ù�����
	char szFilter[] = "ͼƬ(*.bmp)|*.bmp|�����ļ�(*.*)|*.*||";

	AfxSetResourceHandle(GetModuleHandle(NULL));

	// ������ļ��Ի���   
	CFileDialog fileDlg(TRUE, _T("bmp"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | OFN_ALLOWMULTISELECT, Char2LPCTSTR(szFilter));

	// Ϊ��ʵ�ֶ��ļ�ͬʱ���
	DWORD max_file = 40000;	// ����own filename buffer �Ĵ�С
	TCHAR* lsf = new TCHAR[max_file];
	fileDlg.m_ofn.nMaxFile = max_file;
	fileDlg.m_ofn.lpstrFile = lsf;
	fileDlg.m_ofn.lpstrFile[0] = NULL;	// ��ʼ���Ի���

	int iReturn = fileDlg.DoModal();
	//system("cls");
	int nCount = 0;
	// ��ʾ���ļ��Ի���   
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

	HANDLE *hThread = new HANDLE[nCount];					// �߳̾��
	vector<string>::iterator istr = filepaths.begin();
	vector<Mat>::iterator im = loadImages.begin();
	vector<Mat>::iterator imrgb = loadImagesRGB.begin();
	loadImgParas iter(istr, im, imrgb);

	for (int i = 0; i < nCount; i++){
		hThread[i] = CreateThread(NULL, 0, ThreadReadImages, &iter, 0, NULL);
	}

	// �ȴ������߳̽���
	WaitForMultipleObjects(nCount, hThread, TRUE, INFINITE);
	printf("��1.������ͼ��ɹ�������~~~\n");

	threadImgNum = 0;
	threadEndNum = 0;
	for (int i = 0; i < 25; i++)
		CloseHandle(hThread[i]);
	DeleteCriticalSection(&cs);              // �ͷ��ٽ���
	return true;
}


// �������
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
			if (tmp.channels() == 1) //�Ҷ�ͼ
			{
				if (type == NOISE_SALT) tmp.at<uchar>(j, i) = 255;
				else if (type == NOISE_PEPPER) tmp.at<uchar>(j, i) = 0;
			}
			else if (tmp.channels() == 3) // ��ɫͼ
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
		// �ж�ͼ���������
		if (resultImage.isContinuous()){
			nCols = nCols*nRows;
			nRows = 1;
		}
		for (int i = 0; i < nRows; i++){
			for (int j = 0; j < nCols; j++){
				// ��Ӹ�˹����
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

// ���ɸ�˹����
double generateGaussianNoise(double mu, double sigma){
	// ����Сֵ
	const double epsilon = 1e-7;
	//const double epsilon = std::numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	// flag Ϊ�ٹ����˹�������X
	if (!flag)
		return z1*sigma + mu;
	double u1, u2;
	// �����������
	do{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	// flag Ϊ�湹���˹�������X
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}


//
// ͼ��ת
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

// ��ɫ����ģ��
void MyImage::colorReduce(Mat &image, int div)
{
	int nl = image.rows;// ����
	int nc = image.cols;// ����

	// ͼ���������洢����
	if (image.isContinuous())
	{
		// û�ж��н����
		nc = nc*nl;
		nl = 1;  // һά����
	}
	int n = static_cast<int>(
		log(static_cast<double>(div)) / log(2.0));
	// ����������ֵ����ȡ���Ķ�������Ĥ
	uchar mask = 0xFF << n;//e.g.for div = 16,mask = 0xF0
	// for all pixels
	for (int j = 0; j < nl; j++)
	{
		// ��j�еĵ�ַ
		uchar *data = image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++)
		{
			// ����ÿ������-------------
			*data++ = *data&mask + div / 2;
			*data++ = *data&mask + div / 2;
			*data++ = *data&mask + div / 2;
			// ���ش������
		}// �д������
	}
}

// ����������˹��ͼ����
void MyImage::sharpen(const Mat &image, Mat &result)
{
	if (image.channels() == 3)
	{
		printf("������һ���Ҷ�ͼ�񣡣���\n");
		return;
	}
	// ���б�Ҫ�����ͼ��
	result.create(image.size(), image.type());
	for (int j = 1; j < image.rows - 1; j++)
	{
		// ������˵�һ�к����һ��֮���������
		const uchar* previous =
			image.ptr<const uchar>(j - 1);// ��һ��
		const uchar* current =
			image.ptr<const uchar>(j);  // ��ǰ��
		const uchar* next =
			image.ptr<const uchar>(j + 1);// ��һ��
		uchar *output = result.ptr<uchar>(j);// �����
		for (int i = 1; i < image.cols - 1; i++)
		{
			*output++ = saturate_cast<uchar>(
				5 * current[i] - current[i - 1]
				- current[i + 1] - previous[i] - next[i]);
		}
	}
	// ��δ�������������Ϊ0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

void MyImage::sharpen2D(const Mat &image, Mat &result)
{
	// ����ˣ��������ʼ��Ϊ0��
	Mat kernel(3, 3, CV_32F, Scalar(0));
	// �Ժ�Ԫ�ؽ��и�ֵ
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	// ��ͼ������˲�
	filter2D(image, result, image.depth(), kernel);
}

// �õ�ͼ��ĸ�Ƭ
Mat MyImage::getInvert(const Mat& image)
{
	int dim(256);
	Mat lut(1,            // 1D
		&dim,         // 256��
		CV_8U);       // uchar
	for (int i = 0; i < 256; i++)
		lut.at<uchar>(i) = 255 - i;
	return applyLookUp(image, lut);
}

// ��ͼ��Ӧ�ò����������ͼ��
Mat MyImage::applyLookUp(const Mat& image, const Mat& lookup)
{
	// ���ͼ��
	Mat result;
	// Ӧ�ò��ұ�
	LUT(image, lookup, result);
	return result;
}

// ���غ��� f(x,y) = [f(x+1,y)-f(x-1,y)]/2 , x�������У�y��������
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

// �������� f(x,y) = [f(x,y+1)-f(x,y-1)]/2 , x�������У�y��������
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

// �����ݶȷ�ֵ ע�ⷵ�ص���CV_32FC1��ʽ �����ݶȷ�ֵ����thresh���ݶȷ�����beginAngl��endAngl֮��ġ��ݶȷ�ֵ��
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


// �����ݶȷ��� ע�ⷵ�ص���CV_8UC1��ʽ��ģ
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

/** �Ľ�������ͼ��Ϊ��ֵͼ�� 0��255
* @brief ������ͼ�����ϸ��
* @param[in] srcΪ����ͼ��,��cvThreshold�����������8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param[out] dstΪ��srcϸ��������ͼ��,��ʽ��src��ʽ��ͬ������ǰ��Ҫ����ռ䣬Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param[in] maxIterations���Ƶ���������������������ƣ�Ĭ��Ϊ-1���������Ƶ���������ֱ��������ս��
*/
cv::Mat thinImage(const cv::Mat & src, const int maxIterations)
{
	assert(src.type() == CV_8UC1);

	src /= 255;

	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //��¼��������  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������  
			break;
		std::vector<uchar *> mFlag; //���ڱ����Ҫɾ���ĵ�  
		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
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
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}

		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
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
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}
	}
	return dst * 255;
}

// ����ͼ��
cv::Mat loadImage(std::string filepath, bool grayScale){
	Mat result;
	if (grayScale)
		result = imread(filepath, 0);
	else
		result = imread(filepath);
	return result;
}

// ����ͼ�� ����ͬһ�ļ����� ����������ͬ��ͼƬ
bool loadImage(std::vector<cv::Mat>& images, std::string folderpath, std::string prefix, int num, std::string suffix, int initIndex){
	for (int i = 0; i < num; i++){
		Mat tempImg = imread(folderpath + "\\" + prefix + int2string(i + initIndex) + "." + suffix, 0);
		images.push_back(tempImg);
	}
	return true;
}

// ����ͼ����һ��ͼ������ʾvector<Point2f>
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

// ����ͼ����һ��ͼ������ʾvector<Point2f>
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

// ����ͼ����һ��ͼ������ʾvector<Point2f>
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

// ����ͼ����һ��ͼ������ʾvector<Point2f>
Mat drawVecPoints(vector<Point2f> points, Size size, Scalar color, int thickness){
	Mat result = Mat::zeros(size, CV_8UC3);
	for (int i = 0; i < (int)points.size(); i++){
		drawCross(result, points[i], 5, color, thickness);
		//result.at<uchar>(points[i].y, points[i].x) = 255;
	}
	return result;
}

// ����ͼ����һ��ͼ���ϻ���vector<Point2f> �����̶���СΪSize(800*600) 
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


// ����ͼ����һ��ͼ������ʾvector<Point2f> rc�Ǹõ������
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

// ����ͼ����ͼ������ʾvector<Point2f> ˳�ν�����ֱ���������� ��ά������
void drawSeqPoints(Mat& canvas, vector<Point2f> points, Scalar color/* = MC_WHITE*/, int lineWidth/* = 1*/){
	for (int i = 0; i < (int)points.size()-1; i++){
		line(canvas, Point(points[i].x, points[i].y),
			Point(points[i + 1].x, points[i + 1].y),
			color, lineWidth, 8, 0);
	}
}

//�����ơ� ��ͼ���ϻ���ʮ��  lenΪĳ���ߵĳ���
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

// ������Բ x1�� y1 �������Ͻǵ�����  x2 ,y2�������½ǵ�����
void drawEllipse(Mat& src, double x1, double y1, double x2, double y2, Scalar color, int thickness){
	RotatedRect r(Point2f((x1 + x2) / 2, (y1 + y2) / 2), Size(fabs(x2 - x1), fabs(y2 - y1)), 0);
	ellipse(src, r, color, thickness);
}

// �����ơ� ��ͼ������A��B֮����Ƽ�ͷ
void drawArrow(Mat& src, cv::Point2f A, cv::Point2f B, Scalar& color, int thickness,
	int lineType, int shift){
	float lenRatio = 0.32f;
	float angle = 20;
	float D = dist(A, B);
	float len = (float)(D*lenRatio*tan(angle / 180.0 * CV_PI));
	Point2f O(B.x - (B.x - A.x)*lenRatio, B.y + (A.y - B.y)*lenRatio);
	Vec2f OP((B.y - A.y) / D*len, (B.x - A.x) / D*len);		// ����������AB��ֱ������
	line(src, A, Point(B.x - (B.x - A.x)*lenRatio, B.y + (A.y - B.y)*lenRatio), color, thickness, lineType, shift);
	triangle(src, Point((int)B.x, (int)B.y), Point((int)(O.x - OP[0]), (int)(O.y + OP[1])),
		Point((int)(O.x + OP[0]), (int)(O.y - OP[1])), color);
}

// �����ơ� ��ͼ���ϻ��Ƽ�ͷ����ͷ�����ĵ�����cen���ͷָ��angle
// ����涨ˮƽ����Ϊ0�㣬��ֱ����Ϊ90�㣬ˮƽ����Ϊ180�� angleΪ�Ƕ�ֵ
void drawArrow(cv::Mat& src, cv::Point cen, double angle, double len, cv::Scalar& color, int thickness/* = 1*/, int lineType/* = 8*/, int shift/* = 0*/){
	Point2f A, B;
	A.x = cen.x - cos(angle / 180.0*CV_PI)*len / 2.0;
	A.y = cen.y - sin(angle / 180.0*CV_PI)*len / 2.0;
	B.x = cen.x + cos(angle / 180.0*CV_PI)*len / 2.0;
	B.y = cen.y + sin(angle / 180.0*CV_PI)*len / 2.0;
	drawArrow(src, A, B, color, thickness, lineType, shift);
}

// �����ơ���ͼ���ϻ���������
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

// �����ơ���ͼ���ϻ����������� rΪ�����εĳߴ� rΪ���������Բ�뾶
// thickness = -1 Ϊʵ��
void triangle(cv::Mat& src, cv::Point p, double r, cv::Scalar& color, int thickness/* = 1*/){
	double tmpR = r*cos(30.0 / 180 * CV_PI);
	double x1 = p.x - tmpR;
	double x2 = p.x + tmpR;
	double y_top = p.y - r;
	double y_bottom = p.y + r*sin(30.0/180*CV_PI);
	triangle(src, Point(x1, y_bottom), Point(x2, y_bottom), Point(p.x, y_top), color, thickness);
}

// �����ơ�������� ��ͼ���ϻ���������� ,rΪ������ε����Բ�뾶 
// angleOff Ϊƫ�ýǶ� ˮƽ���ҷ���Ϊ0�㣬��ʱ��ת��angleOffΪ��ֵ
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

// �����ơ������ ��ͼ���ϻ�������ǣ�rΪ����ǵ����Բ�ߴ�
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

// �����ơ����� ��ͼ���ϻ��Ʒ��飬rΪ����ı߳�
void drawSquare(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness/* = 1*/){
	rectangle(src, Rect(p.x - r / 2.0, p.y - r / 2.0, r, r), color, thickness);
}

// �����ơ��� ���Ʋ�� rΪĳ�ߵĳ���
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

// ������������ƫ�� ��ʮ�ֲ�˿����ʾ
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
	// ������
	drawArrow(canvas, Point2d(100, 700), Point2d(100, 20), MC_BLACK, 1, 8, 0);
	drawArrow(canvas, Point2d(100, 700), Point2d(980, 700), MC_BLACK, 1, 8, 0);

	double stepx = 0.5;
	double stepy = 0.4;
	// ������
	for (int i = 0; ymin + i*stepy <= ymax + stepy; i++){
		int tmpx = 25;
		int tmpy = (int)(700 - i*stepy*ratioy);
		cv::putText(canvas, double2string(ymin + i*stepy), Point2d(tmpx, tmpy), CV_FONT_ITALIC, 0.7, MC_BLACK, 2);
	}

	// ������
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

// �������ؽǵ�
vector<Point2f> getGoodFeaturePoints(const Mat& src, int maxCornerNum,
	double qualityLevel, double minDist, Mat mask, int blockSize, double k){
	vector<Point2f> corners;
	if (!src.data) {
		printf("��MyImage::getGoodFeaturePoints(src, maxCornerNum, qualityLevel, minDist, blockSize, k)����src����ͼƬ����Ϊ�գ�������");
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

// ���жϡ��������Ƿ���ֵ��low �� high֮�� ��ȷ��srcΪ�Ҷ�ͼ��
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

// ���жϡ����Ƿ������ģ
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

// ��ɸѡ�����Ϲ�صĶ�ά��
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

// ͳ����ģ��ֵ���ص����
int countTrueNums(Mat mask){
	if (mask.channels() > 1){
		printf("��ͳ����ģ��ֵ���ص������EROOR:ͼ��ǵ�ͨ���Ҷ�ͼ��\n");
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

// ��ͼƬ�������ߵ���������
void setBorderZero(Mat& mat){
	mat.row(0).setTo(0);
	mat.row(mat.rows - 1).setTo(0);
	mat.col(0).setTo(0);
	mat.col(mat.cols - 1).setTo(0);
}

// �������ص�ת��Ϊvector<Point>
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

// ����ͼ��Խ��߳ߴ�
float digLength(Mat mat){
	return sqrt((float)(mat.rows*mat.rows + mat.cols*mat.cols));
}

// ��Ѷ�ֵ����ֵѡȡ
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

	//ͳ��ÿ���Ҷȵ�����
	for (i = 0; i < width; i++){
		for (j = 0; j < height; j++){
			tmp = dst.at<uchar>(i, j);
			hst[tmp]++;
		}
	}

	//����ÿ���Ҷȼ�ռͼ���еĸ���
	for (i = 0; i < 256; i++)
		pro_hst[i] = (double)hst[i] / (double)(width*height);


	//����ƽ���Ҷ�ֵ
	u = 0.0;
	for (i = 0; i < 256; i++)
		u += i*pro_hst[i];

	double det = 0.0;
	for (i = 0; i < 256; i++)
		det += (i - u)*(i - u)*pro_hst[i];

	//ͳ��ǰ���ͱ�����ƽ���Ҷ�ֵ����������䷽��

	for (i = 0; i < 256; i++){
		w0 = 0.0; w1 = 0.0; u0 = 0.0; u1 = 0.0; uk = 0.0;
		for (j = 0; j < i; j++){
			uk += j*pro_hst[j];
			w0 += pro_hst[j];
		}
		u0 = uk / w0;

		w1 = 1 - w0;
		u1 = (u - uk) / (1 - w0);

		//������䷽��
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

//OTSU��������ͼ���ֵ��������Ӧ��ֵ 
/*
OTSU �㷨����˵������Ӧ���㵥��ֵ������ת���Ҷ�ͼ��Ϊ��ֵͼ�񣩵ļ򵥸�Ч������
����Ĵ��������� Ryan Dibble�ṩ���˺󾭹�����Joerg.Schulenburg, R.Z.Liu ���޸ģ�������

ת�ԣ�http://forum.assuredigit.com/display_topic_threads.asp?ForumID=8&TopicID=3480

�㷨������ĻҶ�ͼ���ֱ��ͼ���з�������ֱ��ͼ�ֳ��������֣�
ʹ��������֮��ľ�����󡣻��ֵ������õ���ֵ��

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
	int thresholdValue = 1;	// ��ֵ
	int ihist[256];			// ͼ��ֱ��ͼ��256����

	// ��ֱ��ͼ����
	memset(ihist, 0, sizeof(ihist));

	int gmin = 255, gmax = 0;
	// ����ֱ��ͼ
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
		sum += (double)k*(double)ihist[k];		// x*f(x) ������
		n += ihist[k];							// f(x) ����
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

/* �޸�BMPͼ��ߴ�*/
void Resize(CBitmap* src, CBitmap *dst, cv::Size size){
	CDC dcScreen;
	dcScreen.Attach(::GetDC(NULL));

	// ȡ��ԭʼ�D�n�� dc
	CDC dcMemory;
	dcMemory.CreateCompatibleDC(&dcScreen);
	CBitmap *pOldOrgBitmap = dcMemory.SelectObject(src);

	// �����µĽ��ͼ�� (ָ����С)
	dst->CreateCompatibleBitmap(&dcScreen, size.width, size.height);

	CDC dcFixMemory;
	dcFixMemory.CreateCompatibleDC(&dcScreen);
	CBitmap *pOldReslutBitmap = dcFixMemory.SelectObject(dst);

	// ��ԭʼͼ�����Ż��� Memory DC����
	BITMAP bmpInfo;
	src->GetBitmap(&bmpInfo); // ȡ�� ԭʼͼ�εĿ����߶�
	int mode = SetStretchBltMode(dcFixMemory, COLORONCOLOR); //���ò�ʧ������
	StretchBlt(dcFixMemory, 0, 0, size.width, size.height, dcMemory, 0, 0, bmpInfo.bmWidth, bmpInfo.bmHeight, SRCCOPY);
	//DC2.StretchBlt(0, 0, 200, 200, &DC1, 0, 0, info.bmWidth, info.bmHeight, SRCCOPY);
	SetStretchBltMode(dcFixMemory, mode);

	// Set Back
	dcMemory.SelectObject(pOldOrgBitmap);
	dcFixMemory.SelectObject(pOldReslutBitmap);
}
// ����ͼ��
bool loadImage(Mat& image, CString& filepath, bool grayScale)
{
	// ���ù�����
	char szFilter[] = "ͼƬ(*.bmp)|*.bmp|�����ļ�(*.*)|*.*||";
	AfxSetResourceHandle(GetModuleHandle(NULL));

	// ������ļ��Ի���   
	CFileDialog fileDlg(TRUE, _T("bmp"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, Char2LPCTSTR(szFilter));

	int iReturn = fileDlg.DoModal();

	// ��ʾ���ļ��Ի���   
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

// ���غ����ݶ�ͼ
Mat gradX(Mat src){
	Mat grad_x, abs_grad_x;
	Scharr(src, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	return abs_grad_x;
}

// ���������ݶ�ͼ
Mat gradY(Mat src){
	Mat grad_y, abs_grad_y;
	Scharr(src, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	return abs_grad_y;
}

// ���������ݶ�ͼ
Mat gradXY(Mat src){
	Mat gx = gradX(src);
	Mat gy = gradY(src);
	Mat result;
	addWeighted(gx, 0.5, gy, 0.5, 0, result);
	return result;
}

// ���ƽ����ת���� ������ angleΪ�Ƕ��ƣ���ʱ��ת��angleΪ��ֵ
Mat getRotatedImg(const Mat& img, double angle, double scale)
{
	Mat tmp = img.clone();          // ������ʱ�����õ�ͼƬ
	Point2f center = Point2f((float)tmp.cols / 2, (float)tmp.rows / 2);// ��ת����  
	Mat matrix2D = getRotationMatrix2D(center, angle, scale);
	Mat rotateImg;
	warpAffine(tmp, rotateImg, matrix2D, tmp.size());
	return rotateImg;
}

// ��ת�任,ԭʼͼ���������� ��������Ϊ��ɫ ������С��Χ�ı���
// ˳ʱ��Ϊ��
cv::Mat angleRotate(cv::Mat& src, int angle){
	float theta = angle * CV_PI / 180.0f;

	int oldWidth = src.cols;
	int oldHeight = src.rows;

	// Դͼ���ĸ��ǵ����꣨��ͼ������Ϊ����ϵԭ�㣩
	float fSrcX1 = (float)(-(oldWidth - 1) / 2);
	float fSrcY1 = (float)((oldHeight - 1) / 2);

	float fSrcX2 = (float)((oldWidth - 1) / 2);
	float fSrcY2 = (float)((oldHeight - 1) / 2);

	float fSrcX3 = (float)(-(oldWidth - 1) / 2);
	float fSrcY3 = (float)(-(oldHeight - 1) / 2);

	float fSrcX4 = (float)((oldWidth - 1) / 2);
	float fSrcY4 = (float)(-(oldHeight - 1) / 2);

	// ��ת���ĸ��ǵ����꣨��ͼ������Ϊ����ϵԭ�㣩
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

//	SurfFeatureDetector surf(2500.);	// ��ֵ

//	surf.detect(A, keypoints1);
//	surf.detect(B, keypoints2);

	KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
	KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

	Mat fundemental = findFundamentalMat(Mat(selPoints1), Mat(selPoints2), CV_FM_7POINT);

	// ����ͼ�л��ƶ�Ӧ�ļ���
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

// �Ҷ�ͼ���Ϊ��ͨ��ͼ��������ͨ�����Ƶ�һͨ��
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

// ����H-Sֱ��ͼ ����ͼ��Ϊ��ɫͼ��
Mat getH_SHistgram(Mat src, MatND& hist, int hueBinNum){
	if (src.channels() == 1){
		return Mat();
	}
	Mat hsvImage;
	cvtColor(src, hsvImage, CV_BGR2HSV);

	//int hueBinNum = 30;	// ɫ����ֱ��ͼֱ������
	int saturationBinNum = 32;		// ���Ͷȵ�ֱ��ͼֱ������
	int histSize[] = { hueBinNum, saturationBinNum };

	// ����ɫ���ı仯��ΧΪ0��179
	float hueRanges[] = { 0, 180 };
	// ���履�Ͷȵı仯��ΧΪ0���ڡ��ס��ң���255����������ɫ��
	float saturationRanges[] = { 0, 256 };
	const float* ranges[] = { hueRanges, saturationRanges };

	// ����׼��,calcHist�����н������0ͨ���͵�1ͨ����ֱ��ͼ
	int channels[] = { 0, 1 };

	calcHist(&hsvImage,		// �����ͼ��
		1,					// �������Ϊ1
		channels,			// ͨ������
		Mat(),				// ��ʹ����Ĥ
		hist,			// �����Ŀ��ֱ��ͼ
		2,					// ��Ҫ�����ֱ��ͼ��ά��Ϊ2
		histSize,			// ���ÿ��ά�ȵ�ֱ��ͼ�ߴ������
		ranges,				// ÿһά��ֵ��ȡֵ��Χ����
		true,				// ָʾֱ��ͼ�Ƿ���ȵı�ʶ����true��ʾ���ȵ�ֱ��ͼ
		false);				// �ۼƱ�ʶ����false��ʾֱ��ͼ�����ý׶λᱻ����

	// Ϊ����ֱ��ͼ׼������
	double maxValue = 0;	// ���ֵ
	minMaxLoc(hist, 0, &maxValue, 0, 0);		// ����������������ȫ����Сֵ
	int scale = 10;

	Mat histImg = Mat::zeros(saturationBinNum*scale, hueBinNum * 10, CV_8UC3);

	for (int hue = 0; hue < hueBinNum; hue++){
		for (int saturation = 0; saturation < saturationBinNum; saturation++){
			float binValue = hist.at<float>(hue, saturation);	// ֱ��ͼֱ����ֵ
			int intensity = cvRound(binValue * 255 / maxValue);		// ǿ��

			rectangle(histImg, Point(hue*scale, saturation*scale),
				Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1),
				Scalar::all(intensity), CV_FILLED);
		}
	}

	return histImg;

}

// ����RGB��ɫֱ��ͼ ����ͼ��Ϊ��ɫͼ��
Mat getRGBHistgram(Mat src, MatND& hist, int bins){
	int hist_size[] = { bins };
	float range[] = { 0, (float)bins };
	const float* ranges[] = { range };
	MatND redHist, greenHist, blueHist;
	int channels_r[] = { 0 };

	// ����ֱ��ͼ�ļ��㣨��ɫ�������֣�
	calcHist(&src, 1, channels_r, Mat(),		// ��ʹ����Ĥ
		redHist, 1, hist_size, ranges, true, false);

	// ����ֱ��ͼ�ļ��㣨��ɫ�������֣�
	int channels_g[] = { 1 };
	calcHist(&src, 1, channels_g, Mat(),		// ��ʹ����Ĥ
		greenHist, 1, hist_size, ranges, true, false);

	// ����ֱ��ͼ�ļ��㣨��ɫ�������֣�
	int channels_b[] = { 2 };
	calcHist(&src, 1, channels_b, Mat(),		// ��ʹ����Ĥ
		blueHist, 1, hist_size, ranges, true, false);

	// ���Ƴ���ɫֱ��ͼ
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


// ����ֱ��ͼ ����ͼ��Ϊ��ɫͼ���Ҷ�ͼ�� ��Ϊ��ɫͼ���ת��Ϊ�Ҷ�ͼ��
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

// ��������ͼ��histֱ��ͼ�ȽϵĽ��
// ���ֱȽϷ���Ϊ 
//��CV_COMP_CORREL��
//��CV_COMP_CHISQR��
//��CV_COMP_INTERSECT��
//��CV_COMP_BHATTACHARYYA�� 
double calCompareH_SHist(Mat A, Mat B, int method){
	MatND histA, histB;
	getH_SHistgram(A, histA);
	getH_SHistgram(B, histB);
	return compareHist(histA, histB, method);
}

// ������ͶӰ����ȡͼ����Բο�ͼ��ķ���ͶӰͼ
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

	// ���㷴��ͶӰ
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

// ģ��ƥ����� method��ƥ�䷽�� ��������
// ��CV_TM_SQDIFF��
// ��CV_TM_SQDIFF_NORMED��
// ��CV_TM_CCORR��
// ��CV_TM_CCORR_NORMED��
// ��CV_TM_CCOEFF��
// ��CV_TM_CCOEFF_NORMED��
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


// ������ͼ�񡿱���BMPͼ��
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

	// ��д�ļ�ͷ
	BITMAPINFOHEADER head;
	head.biBitCount = byte * 8;
	head.biHeight = height;
	head.biWidth = width;
	head.biCompression = 0;	// 0 ��ʾ��ѹ��
	head.biSizeImage = 0;	// ��ѹ������¿���Ϊ0
	head.biClrImportant = 0;
	head.biClrUsed = 0;
	head.biPlanes = 1;
	head.biSize = 40;
	head.biXPelsPerMeter = 0;
	head.biYPelsPerMeter = 0;
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

	// ��ɫ����
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

	// ��չÿ���ֽ�����׼��ͼ�����ݲ�����
	unsigned char*buf = new unsigned char[height * lineByte];
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width*byte; j++){
			*(buf + i*lineByte + j) = *(imgBuf + (height - 1 - i)*width * byte + j);
		}
	}
	fwrite(buf, height*lineByte, 1, fp);

	fclose(fp);

	delete[]buf;		// �ͷ���Դ
	return 1;
}

// ������ | ��ȡ ͼ��BMPͼ������  ����Ϊ unsigned char*  coordinateleft1.bin
// ����Ϊ�������ļ�
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

// ������ͼ��BMPͼ������ ����Ϊ float*         coordinateleft1.bin
// Mat����Ϊ�������ļ�
bool BmpSerialize(string fileName, Mat data, bool bRead)
{
	if (bRead)
	{
		CFile loadF;
		if (FALSE == loadF.Open(string2CString(fileName), CFile::modeRead)){
			AfxMessageBox(_T("Serialize pic open error"));
			return false;
		}

		// ���ȶ�ȡ�����Ϣ ���ж�ȡ
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
		// ���ȴ洢�����Ϣ
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

// ������ͼ��BMPͼ������ ����Ϊ float*         coordinateleft1.bin
// ����Ϊ�������ļ�
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

// ���Mat�Ļ�����Ϣ
void printMatInfo(Mat input)
{
	// ��ȡ����������
	std::cout << "Input row: " << input.rows << std::endl;
	std::cout << "Input col: " << input.cols << std::endl;

	cout << input.step.buf[0] << endl;
}

// ƽ�Ʋ�����ͼ���С����
cv::Mat imageTranslation1(cv::Mat& srcImage, int xOffset, int yOffset){
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	cv::Mat resultImage(srcImage.size(), srcImage.type());

	if (srcImage.channels() == 3){
		// ����ͼ��
		for (int i = 0; i < nRows; ++i){
			for (int j = 0; j < nCols; ++j){
				// ӳ��任
				int x = j - xOffset;
				int y = i - yOffset;
				// �߽��ж�
				if (x >= 0 && y >= 0 && x < nCols && y < nRows)
					resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
			}
		}
	}
	else if (srcImage.channels() == 1){
		// ����ͼ��
		for (int i = 0; i < nRows; ++i){
			for (int j = 0; j < nCols; ++j){
				// ӳ��任
				int x = j - xOffset;
				int y = i - yOffset;
				// �߽��ж�
				if (x >= 0 && y >= 0 && x < nCols && y < nRows)
					resultImage.at<uchar>(i, j) = srcImage.ptr<uchar>(y)[x];
			}
		}
	}

	return resultImage;
}


// ƽ�Ʋ�����ͼ���С�ı�
cv::Mat imageTranslation2(cv::Mat &srcImage, int xOffset, int yOffset){
	// ����ƽ�Ƴߴ�
	int nRows = srcImage.rows + abs(yOffset);
	int nCols = srcImage.cols + abs(xOffset);

	cv::Mat resultImage(nRows, nCols, srcImage.type());

	// ͼ�����
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols; j++){
			// ӳ��任
			int x = j - xOffset;
			int y = i - yOffset;
			// �߽��ж�
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

// ���ڵȼ����ȡͼ������
cv::Mat imageReduction1(cv::Mat &srcImage, float kx, float ky){
	if (kx > 1.0f || ky > 1.0f){
		printf("��Error��imageReduction1���������˺����޷�ִ�зŴ����\n");
		return Mat();
	}
	// ��ȡ���ͼ��ֱ���
	int nRows = cvRound(srcImage.rows*kx);
	int nCols = cvRound(srcImage.cols*ky);

	cv::Mat resultImage(nRows, nCols, srcImage.type());
	for (int i = 0; i < nRows; ++i){
		for (int j = 0; j < nCols; ++j){
			// ����ˮƽ���Ӽ�������
			int x = static_cast<int>((i + 1) / kx + 0.5) - 1;
			// ���ݴ�ֱ���Ӽ�������
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

// ���������ӿ� ���ػҶ�ƽ��ֵ
cv::Vec3b areaAverage(const cv::Mat& srcImage, Point_<int> leftPoint, Point_<int> rightPoint){
	int temp1 = 0, temp2 = 0, temp3 = 0;
	// ���������ӿ����ص����
	int nPix = (rightPoint.x - leftPoint.x + 1)*(rightPoint.y - leftPoint.y + 1);
	// �������ӿ����ͨ��������ֵ���
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
	// ��ÿ��ͨ�����ֵ
	Vec3b vecTemp;
	vecTemp[0] = temp1 / nPix;
	vecTemp[1] = temp2 / nPix;
	vecTemp[2] = temp3 / nPix;
	return vecTemp;
}


// ���������ӿ���ȡͼ������
// �����ӿ���ȡͼ��������ͨ����Դͼ����������ӿ黮�֣�Ȼ����ȡ�ӿ�������ֵ��Ϊ���������Թ�����ͼ����ʵ�ֵġ�
cv::Mat imageReduction2(const Mat& srcImage, double kx, double ky){
	// ��ȡ���ͼ��ֱ���
	int nRows = cvRound(srcImage.rows*kx);
	int nCols = cvRound(srcImage.cols*ky);

	cv::Mat resultImage(nRows, nCols, srcImage.type());

	// �����ӿ�����Ͻ���������
	int leftRowCoordinate = 0;
	int leftColCoordinate = 0;

	for (int i = 0; i < nRows; ++i){
		// ����ˮƽ���Ӽ�������
		int x = static_cast<int>((i + 1) / kx + 0.5) - 1;
		for (int j = 0; j < nCols; ++j){
			// ���ݴ�ֱ���Ӽ�������
			int y = static_cast<int>((j + 1) / ky + 0.5) - 1;

			Vec3b tempV = areaAverage(srcImage,
				Point_<int>(leftRowCoordinate, leftColCoordinate),
				Point_<int>(x, y));

			// ��������ӿ�ľ�ֵ
			if (srcImage.channels() == 3){
				resultImage.at<Vec3b>(i, j) = tempV;
			}
			else if (srcImage.channels() == 1){
				resultImage.at<uchar>(i, j) = tempV[0];
			}
			// �������ӿ����Ͻǵ������꣬�����겻��
			leftColCoordinate = y + 1;
		}
		leftColCoordinate = 0;
		// �������ӿ����Ͻǵ�������
		leftRowCoordinate = x + 1;
	}
	return resultImage;
}

// ��÷���任ͼ��
cv::Mat getAffineTransformImage(cv::Mat srcImage, const Point2f srcPts[], const Point2f dstPts[]){
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	// �������任����2*3
	cv::Mat warpMat(cv::Size(2, 3), CV_32F);
	cv::Mat resultImage = cv::Mat::zeros(nRows, nCols, srcImage.type());
	// �������任���󣬼�����任��2*3����
	warpMat = cv::getAffineTransform(srcPts, dstPts);
	// ���ݷ���������ͼ�����任
	cv::warpAffine(srcImage, resultImage, warpMat, resultImage.size());
	return resultImage;
}

// ���б��ͼ��
// ��бΪ��  ��бΪ�� �Ƕ���
cv::Mat getSkewImage(cv::Mat srcImage, float angle){
	// �Ƕ�ת��
	float alpha = (float)fabs(angle * CV_PI / 180);

	int nRows = srcImage.rows;
	int nCols = (int)(srcImage.rows * tan(alpha) + srcImage.cols);

	// �������任���� б��
	Mat warpMat = (Mat_<float>(2, 3) << 1, tan(alpha), 0, 0, 1, 0);
	cv::Mat resultImage = cv::Mat::zeros(nRows, nCols, srcImage.type());
	// ���ݷ���������ͼ�����任
	cv::warpAffine(srcImage, resultImage, warpMat, resultImage.size());

	if (angle < 0)
		return getFlipImage(resultImage, FLIP_HORIZONTAL);
	else
		return resultImage;
}

// ��Ƶ��������
// ����PSNR��ֵ����ȣ�������ֵΪ30~50dB,ֵԽ��Խ��
double PSNR(const Mat& I1, const Mat& I2){
	cv::Mat s1;
	// ����ͼ���|I1 - I2|
	absdiff(I1, I2, s1);
	// ת��32����������ƽ������
	s1.convertTo(s1, CV_32F);
	// s1*s1, ��|I1 - I2|^2
	s1 = s1.mul(s1);
	// �ֱ����ÿ��ͨ����Ԫ�أ�����s��
	cv::Scalar s = sum(s1);
	// ��������ͨ��Ԫ�غ�
	double sse = s.val[0] + s.val[1] + s.val[2];
	cout << "sse = " << sse << endl;
	// ��Ԫ�غ�Сʱ����0ֵ
	if (sse <= 1e-10)
		return 0;
	else {
		// ���ݹ�ʽ���㵱ǰI1��I2�ľ������
		double mse = sse / (double)(I1.channels() * I1.total());
		// �����ֵ�����
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

// ����MSSIM�ṹ�����ԣ�����ֵ��0��1��ֵԽ��Խ��
cv::Scalar MSSIM(const Mat& i1, const Mat& i2){
	const double C1 = 6.5025, C2 = 58.5225;
	cv::Mat I1, I2;
	// ת����32����������ƽ������
	i1.convertTo(I1, CV_32F);
	i2.convertTo(I2, CV_32F);
	// I2^2
	cv::Mat I2_2 = I2.mul(I2);
	cv::Mat I1_2 = I1.mul(I1);
	cv::Mat I1_I2 = I1.mul(I2);

	cv::Mat mu1, mu2;
	// ��˹��Ȩ����ÿһ���ڵľ�ֵ�������Լ�Э����
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	// ��˹ƽ��
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	// ���ݹ�ʽ������Ӧ����
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
	// ��ƽ��ֵ��Ϊ��ͼ��Ľṹ�����Զ���
	cv::Scalar mssim = mean(ssim_map);
	return mssim;
}

// MatIterator_ ��������ɫ����
cv::Mat inverseColor4(cv::Mat srcImage){
	cv::Mat tempImage = srcImage.clone();
	if (srcImage.channels() == 3){
		// ��ʼ��Դͼ�������
		cv::MatConstIterator_<cv::Vec3b> srcIterStart = srcImage.begin<cv::Vec3b>();
		cv::MatConstIterator_<cv::Vec3b> srcIterEnd = srcImage.end<cv::Vec3b>();
		// ��ʼ�����ͼ�������
		cv::MatIterator_<cv::Vec3b> resIterStart = tempImage.begin<cv::Vec3b>();
		cv::MatIterator_<cv::Vec3b> resIerEnd = tempImage.end<cv::Vec3b>();

		// ����ͼ��ɫ����
		while (srcIterStart != srcIterEnd){
			(*resIterStart)[0] = 255 - (*srcIterStart)[0];
			(*resIterStart)[1] = 255 - (*srcIterStart)[1];
			(*resIterStart)[2] = 255 - (*srcIterStart)[2];
			// ����������
			srcIterStart++;
			resIterStart++;
		}
	}
	else if (srcImage.channels() == 1){
		// ��ʼ��Դͼ�������
		cv::MatConstIterator_<uchar> srcIterStart = srcImage.begin<uchar>();
		cv::MatConstIterator_<uchar> srcIterEnd = srcImage.end<uchar>();
		// ��ʼ�����ͼ�������
		cv::MatIterator_<uchar> resIterStart = tempImage.begin<uchar>();
		cv::MatIterator_<uchar> resIerEnd = tempImage.end<uchar>();

		// ����ͼ��ɫ����
		while (srcIterStart != srcIterEnd){
			(*resIterStart) = 255 - (*srcIterStart);
			// ����������
			srcIterStart++;
			resIterStart++;
		}
	}
	return tempImage;
}

// isContinuous ��ɫ����
cv::Mat inverseColor5(cv::Mat srcImage){
	int row = srcImage.rows;
	int col = srcImage.cols;
	Mat tempImage = srcImage.clone();
	// �ж�ͼ���Ƿ�������ͼ�񣬼��Ƿ����������
	if (srcImage.isContinuous() && tempImage.isContinuous()){
		row = 1;
		// ������չ��
		col = col * srcImage.rows * srcImage.channels();
	}
	// ����ͼ���ÿ������
	for (int i = 0; i < row; i++){
		// �趨ͼ������Դָ�뼰���ͼ������ָ��
		const uchar* pSrcData = srcImage.ptr<uchar>(i);
		uchar* pResultData = tempImage.ptr<uchar>(i);
		for (int j = 0; j < col; j++){
			*pResultData++ = 255 - *pSrcData++;
		}
	}
	return tempImage;
}

// ����2-28 LUT ���ɫ����
cv::Mat inverseColor6(cv::Mat srcImage){
	int row = srcImage.rows;
	int col = srcImage.cols;
	cv::Mat tempImage = srcImage.clone();
	// ����LUT��ɫtable
	uchar LutTable[256];
	for (int i = 0; i < 256; i++){
		LutTable[i] = 255 - i;
	}
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar *pData = lookUpTable.data;
	// ����ӳ���
	for (int i = 0; i < 256; i++){
		pData[i] = LutTable[i];
	}
	// Ӧ����������в���
	cv::LUT(srcImage, lookUpTable, tempImage);
	return tempImage;
}

// ��������ʾ���ͼ��
void showManyImages(const std::vector<cv::Mat> &srcImages, cv::Size imgSize){
	int nNumImages = srcImages.size();
	//cout << "nNumImage  = " << nNumImages << std::endl;
	cv::Size nSizeWindows;
	if (nNumImages > 12){
		std::cout << "Not more than 12 images!" << std::endl;
		return;
	}
	// ����ͼƬ����������ȷ���ָ�С���ڵ���̬
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
	// ����Сͼ��ߴ硢��϶���߽�
	Size nShowImageSize = imgSize;
	int nSplitLineSize = 15;
	int nAroundLineSize = 50;
	// �������ͼ��ͼ���С��������Դ��ȷ��
	const int imagesHeight = nShowImageSize.height *nSizeWindows.height + nAroundLineSize + (nSizeWindows.height - 1)*nSplitLineSize;
	const int imagesWidth = nShowImageSize.width*nSizeWindows.width + nAroundLineSize + (nSizeWindows.width - 1)*nSplitLineSize;
	//std::cout << imagesWidth << " " << imagesHeight << std::endl;

	cv::Mat showWindowImages(imagesHeight, imagesWidth, CV_8UC3, cv::Scalar::all(0));
	// ��ȡ��ӦСͼ������Ͻ�����X��Y
	int posX = (showWindowImages.cols - (nShowImageSize.width*nSizeWindows.width + (nSizeWindows.width - 1)*nSplitLineSize)) / 2;
	int posY = (showWindowImages.rows - (nShowImageSize.height*nSizeWindows.height + (nSizeWindows.height - 1)*nSplitLineSize)) / 2;

	//std::cout << posX << " " << posY << std::endl;
	int tempPosX = posX;
	int tempPosY = posY;
	// ��ÿһС��ͼ�����ϳɴ�ͼ��
	for (int i = 0; i < nNumImages; i++){
		cout << srcImages[i].size() << endl;
		// Сͼ������ת��
		if ((i%nSizeWindows.width == 0) && (tempPosX != posX)){
			tempPosX = posX;
			tempPosY += (nSplitLineSize + nShowImageSize.height);
		}
		//printf("tempPosX = %d tempPosY = %d\n", tempPosX, tempPosY);
		putText(showWindowImages, "PIC " + int2string(i + 1), Point2d(tempPosX + 120, tempPosY + nShowImageSize.height + 12), CV_FONT_ITALIC, 0.5, MC_YELLOW, 1);
		// ����Rect����Сͼ�����ڴ�ͼ�����Ӧ����
		cv::Mat tempImage = showWindowImages(cv::Rect(tempPosX, tempPosY, nShowImageSize.width, nShowImageSize.height));
		// ����resize����ʵ��ͼ������
		Mat tmp = convert2BGR(srcImages[i]);
		resize(tmp, tempImage, nShowImageSize);
		tempPosX += (nSplitLineSize + nShowImageSize.width);
	}
	cv::imshow("showWindowImages", showWindowImages);
}

// ��ȡHSVͼ��
cv::Mat getHSVImage(const Mat& image, Mat& image_H, Mat& image_S, Mat& image_V){
	cv::Mat image_hsv;
	cvtColor(image, image_hsv, CV_BGR2HSV);

	// ����HSV����ͨ��
	std::vector<cv::Mat> hsvChannels;
	cv::split(image_hsv, hsvChannels);
	// 0 ͨ��ΪH����, 1ͨ��ΪS������ 2ͨ��ΪV����
	image_H = hsvChannels[0];
	image_S = hsvChannels[1];
	image_V = hsvChannels[2];
	return image_hsv;
}

// ����Ӧ��ֵ��
cv::Mat getAdaptiveThresholdImage(const Mat& image, double maxValue, int blockSize, double C,
	int adaptiveMethod, int thresholdType){
	Mat input = GrayTrans(image);
	Mat dstImage;
	cv::adaptiveThreshold(input, dstImage, maxValue, adaptiveMethod, thresholdType, blockSize, C);
	return dstImage;
}

// ˫��ֵ��
cv::Mat getDoubleThreshImage(const Mat& image, double lowthresh, double highthresh, double maxValue){
	if (!image.data)
	{
		printf("MyImage.cpp getDoubleThreshImage: ����ͼ������Ϊ��!\n");
		return Mat();
	}
	Mat srcGray = GrayTrans(image);
	Mat dstTempImage1, dstTempImage2, dstImage;
	// С��ֵ��Դ�Ҷ�ͼ�������ֵ������
	cv::threshold(srcGray, dstTempImage1, lowthresh, maxValue, cv::THRESH_BINARY);
	// ����ֵ��Դ�Ҷ�ͼ�������ֵ������
	cv::threshold(srcGray, dstTempImage2, highthresh, maxValue, cv::THRESH_BINARY_INV);
	// ����������õ���ֵ�����
	cv::bitwise_and(dstTempImage1, dstTempImage2, dstImage);
	return dstImage;
}

// ����ֵ��
cv::Mat getHalfThreshImage(const Mat& image, double thresh){
	if (!image.data)
	{
		printf("MyImage.cpp getHalfThreshImage: ����ͼ������Ϊ��!\n");
		return Mat();
	}
	Mat srcGray = GrayTrans(image);
	Mat dstTempImage, dstImage;
	// ��ֵ��Դ�Ҷ�ͼ������ֵ������
	cv::threshold(srcGray, dstTempImage, thresh, 255, cv::THRESH_BINARY);
	// ����������õ���ֵ�����
	cv::bitwise_and(srcGray, dstTempImage, dstImage);
	return dstImage;
}

// ֱ��ͼ���⻯
cv::Mat getEqualHistImage(const Mat& image, bool useRGB){
	if (!image.data)
	{
		printf("MyImage.cpp getEqualHistImage: ����ͼ������Ϊ��!\n");
		return Mat();
	}
	if (useRGB && image.channels() == 3){
		Mat colorHeqImage;
		std::vector<cv::Mat> BGR_plane;
		// ��BGRͨ�����з���
		cv::split(image, BGR_plane);
		// �ֱ��BGR����ֱ��ͼ���⻯
		for (int i = 0; i < (int)BGR_plane.size(); i++){
			cv::equalizeHist(BGR_plane[i], BGR_plane[i]);
		}
		// �ϲ���Ӧ�ĸ���ͨ��
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

// ֱ��ͼ�任��������
cv::Mat getHistogramTransLUT(const Mat& srcImage, int segThreshold){
	// ��һ���� ����ͼ���ֱ��ͼ
	Mat srcGray = GrayTrans(srcImage);
	MatND hist;
	getHistgram(srcImage, hist);

	// �ڶ���������Ԥ�����ͳ�ƻҶȼ��任
	// �ɵ͵��߽��в���
	int iLow = 0;
	for (; iLow < 256; iLow++){
		if (hist.at<float>(iLow) > segThreshold){
			break;
		}
	}
	// �ɸߵ��ͽ��в���
	int iHigh = 255;
	for (; iHigh >= 0; iHigh--){
		if (hist.at<float>(iHigh) > segThreshold){
			break;
		}
	}
	// ���������������ұ�
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
	// ���Ĳ�: ͨ�����ұ����ӳ��任
	cv::Mat histTransResult;
	cv::LUT(srcGray, lookUpTable, histTransResult);

	return histTransResult;
}

// ֱ��ͼ�任�����ۼ�
// ֱ��ͼ�任�ۼƷ���ʵ�ֵ�˼·��
// ��1�� ��Դͼ��ת��Ϊ�Ҷ�ͼ������ͼ��ĻҶ�ֱ��ͼ
// ��2�� ����ӳ�����ֱ��ͼ���������ۻ�
// ��3�� ����ӳ������Ԫ��ӳ��õ����յ�ֱ��ͼ�任
cv::Mat getHistogramTransAggregate(const Mat& srcImage){
	Mat srcGray = GrayTrans(srcImage);
	MatND hist;
	getHistgram(srcImage, hist);
	float table[256];
	int nPix = srcGray.cols * srcGray.rows;
	// ����ӳ���
	for (int i = 0; i < 256; i++){
		float temp[256];
		// ���ر任
		temp[i] = hist.at<float>(i) / nPix * 255;
		if (i != 0){
			// �����ۼ�
			table[i] = table[i - 1] + temp[i];
		}
		else {
			table[i] = temp[i];
		}
	}

	// ͨ��ӳ����б����
	cv::Mat lookUpTable(cv::Size(1, 256), CV_8U);
	for (int i = 0; i < 256; i++){
		lookUpTable.at<uchar>(i) = static_cast<uchar>(table[i]);
	}

	cv::Mat histTransResult;
	cv::LUT(srcGray, lookUpTable, histTransResult);
	return histTransResult;
}

// ֱ��ͼƥ��
// (1) �ֱ����Դͼ����Ŀ��ͼ����ۼƸ��ʷֲ�
// (2) �ֱ��Դͼ����Ŀ��ͼ�����ֱ��ͼ���⻯����
// (3) ������ӳ���ϵʹԴͼ��ֱ��ͼ���չ涨���б任
cv::Mat getHistgramMatchImage(const Mat& srcImage, Mat target){
	if (!srcImage.data || !target.data){
		printf("MyImage.cpp getHistgramMatchImage ����ͼ��Ϊ��!\n");
		return Mat();
	}
	resize(target, target, srcImage.size(), 0, 0, CV_INTER_LINEAR);

	// ��ʼ���ۼƷֲ�����
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
	// ���Դͼ����Ŀ��ͼ����ۼ�ֱ��ͼ
	for (int nrow = 0; nrow < srcImage.rows; nrow++){
		for (int ncol = 0; ncol < srcImage.cols; ncol++){
			srcAddTemp[(int)srcImage.at<uchar>(nrow, ncol)]++;
			dstAddTemp[(int)target.at<uchar>(nrow, ncol)]++;
		}
	}
	// ���Դͼ����Ŀ��ͼ����ۼƸ��ʷֲ�
	for (int i = 0; i < 256; i++){
		sumSrcTemp += srcAddTemp[i];
		srcCdfArr[i] = sumSrcTemp / nSrcPix;
		sumDstTemp += dstAddTemp[i];
		dstCdfArr[i] = sumDstTemp / nDstPix;
	}

	// ֱ��ͼƥ��ʵ��
	for (int i = 0; i < 256; i++){
		float minMatchPara = 20;
		for (int j = 0; j < 256; j++){
			// �жϵ�ǰֱ��ͼ�ۼƲ���
			if (minMatchPara > abs(srcCdfArr[i] - dstCdfArr[j])){
				minMatchPara = abs(srcCdfArr[i] - dstCdfArr[j]);
				matchFlag = j;
			}
		}
		histMatchMap[i] = matchFlag;
	}

	// ��ʼ��ƥ��ͼ��
	cv::Mat HistMatchImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
	cv::cvtColor(srcImage, HistMatchImage, CV_BGR2GRAY);
	// ͨ��mapӳ���ƥ��ͼ��
	for (int i = 0; i < HistMatchImage.rows; i++){
		for (int j = 0; j < HistMatchImage.cols; j++){
			HistMatchImage.at<uchar>(i, j) = histMatchMap[(int)HistMatchImage.at<uchar>(i, j)];
		}
	}
	return HistMatchImage;
}

// ����任
/*
	����Rosenfeld and Pfaltz����ľ���任���ۣ����ڶ�ֵͼ��ǰ��Ŀ��Ϊ1������Ϊ0��������任ʵ��ͼ���ÿ������
	�����ǰ��Ŀ���ͼ��߽�ľ��롣����任�Ĳ������£�
	��1�� ��ͼ������תΪ��ֵͼ��ǰ��Ŀ��Ϊ1������Ϊ0.
	��2�� ��һ��ˮƽɨ������Ͻǿ�ʼ�����δ�������ɨ�裬ɨ����һ���Զ�ת����һ�е�����˼���ɨ�裬���б���ͼ��
	��Ĥģ��maskΪmaskL��Ӧ������Ĺ�ʽ���м���
	f(p)  = min[f(p), D(p,q) + f(q)]   q����maskL
	����DΪ���룬����ŷʽ���롢���̾����������룬f(p)Ϊ���ص�p������ֵ
	��3�� �ڶ���ˮƽɨ������½ǿ�ʼ�����δ�����������ɨ�裬ɨ����һ������ת����һ�е����Ҷ˼���ɨ�裬���б���ͼ��
	��Ĥģ��maskΪmaskR������ͬ���裨2����
	��4�� ����ģ��maskL��maskR��ɨ�����õ����յľ���任ͼ��
	*/
cv::Mat getDistTransImage(Mat& srcImage, int thresh){
	CV_Assert(srcImage.data != NULL);
	cv::Mat srcGray = GrayTrans(srcImage);
	cv::Mat srcBinary;
	// ת���ɶ�ֵͼ��
	threshold(srcGray, srcBinary, thresh, 255, cv::THRESH_BINARY);

	imshow("binary", srcBinary);

	int rows = srcBinary.rows;
	int cols = srcBinary.cols;
	uchar* pDataOne;
	uchar* pDataTwo;
	float disPara = 0;
	float fDisMin = 0;
	// ��һ�����ͼ������ģ���������ֵ
	for (int i = 1; i < rows - 1; i++){
		// ͼ��ָ���ȡ
		pDataOne = srcBinary.ptr<uchar>(i);
		for (int j = 1; j < cols; j++){

			//	printf("(%d , %d)\n", i, j);

			// �ֱ��������ģ���������ؾ���
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

	// �ڶ������ͼ������ģ���������ֵ
	for (int i = rows - 2; i > 0; i--){
		// ͼ��ָ���ȡ
		pDataOne = srcBinary.ptr<uchar>(i);
		for (int j = cols - 1; j >= 0; j--){
			// �ֱ��������ģ���������ؾ���
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

// ����opencv�Դ��ľ���任����
cv::Mat getDistTransImage2(Mat &srcImage, int thresh){
	if (!srcImage.data){
		return Mat();
	}
	// ת��Ϊ�Ҷ�ͼ��
	cv::Mat srcGray = GrayTrans(srcImage);
	// ת��Ϊ��ֵͼ��
	cv::Mat srcBinary;
	threshold(srcGray, srcBinary, thresh, 255, cv::THRESH_BINARY);
	imshow("binary", srcBinary);
	// ����任
	cv::Mat dstImage;
	cv::distanceTransform(srcBinary, dstImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	// ��һ������
	cv::normalize(dstImage, dstImage, 0, 1., cv::NORM_MINMAX);
	return dstImage;
}

// GammaУ�� ٤��У��
// һ������£���Gamma������ֵ����1ʱ��ͼ��ĸ߹ⲿ�ֱ�ѹ�����������ֱ���չ
// ��Gamma����ֵС��1ʱ��ͼ��ĸ߹ⲿ�ֱ���չ���������ֱ�ѹ����
cv::Mat getGammaTransformImage(cv::Mat& srcImage, float kFactor){
	// �������LUT
	unsigned char LUT[256];
	for (int i = 0; i < 256; i++){
		// Gamma�任���ʽ
		LUT[i] = saturate_cast<uchar>(pow((float)(i / 255.0), kFactor)*255.0f);
	}

	cv::Mat resultImage = srcImage.clone();
	// ���ͨ��Ϊ��ͨ��ʱ��ֱ�ӽ��б任
	if (srcImage.channels() == 1){
		cv::MatIterator_<uchar> iterator = resultImage.begin<uchar>();
		cv::MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
		for (; iterator != iteratorEnd; iterator++)
			*iterator = LUT[(*iterator)];
	}
	else {
		// ����ͨ��Ϊ3ͨ��ʱ�����ÿ��ͨ���ֱ���б任
		cv::MatIterator_<cv::Vec3b> iterator = resultImage.begin<Vec3b>();
		cv::MatIterator_<cv::Vec3b> iteratorEnd = resultImage.end<Vec3b>();
		// ͨ�����ұ���б任
		for (; iterator != iteratorEnd; iterator++){
			(*iterator)[0] = LUT[((*iterator)[0])];
			(*iterator)[1] = LUT[((*iterator)[1])];
			(*iterator)[2] = LUT[((*iterator)[2])];
		}
	}
	return resultImage;
}

// ͼ�����Ա任����
cv::Mat getLinearTransformImage(cv::Mat& srcImage, float a, int b){
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	// ͼ��Ԫ�ر���
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols; j++){
			if (srcImage.channels() == 3){
				for (int c = 0; c < 3; c++){
					// ����at����������±��ֹԽ��
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

// ͼ������任����1
// ͼ������任�ǽ�ͼ�������з�Χ��խ�ĵͻҶ�ֵӳ�������нϿ�Χ�ĻҶ�ֵ��
// ��������չͼ���б�ѹ���ģ��Ҷ�ֵ�ϸ�����ģ�������ֵ��
cv::Mat getLogTransform1(cv::Mat srcImage, float c){
	// ����ͼ���ж�
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	// ����1+r
	cv::add(srcImage, cv::Scalar(1.0), srcImage);
	// ת��Ϊ32λ������
	srcImage.convertTo(srcImage, CV_32F);
	// ����log(1+r)
	log(srcImage, resultImage);
	resultImage = c * resultImage;
	// ��һ������
	cv::normalize(resultImage, resultImage, 0, 255, NORM_MINMAX);
	cv::convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

// ͼ������任����2 ��̫��ʹ �������û����תCV_32F
// ͼ������任�ǽ�ͼ�������з�Χ��խ�ĵͻҶ�ֵӳ�������нϿ�Χ�ĻҶ�ֵ��
// ��������չͼ���б�ѹ���ģ��Ҷ�ֵ�ϸ�����ģ�������ֵ��
cv::Mat getLogTransform2(cv::Mat srcImage, float c){
	// ����ͼ���ж�
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	double gray = 0;
	// ͼ�����,�ֱ����ÿ�����ص�Ķ����任
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
	// ��һ������
	cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

// ͼ������任����3
// ͼ������任�ǽ�ͼ�������з�Χ��խ�ĵͻҶ�ֵӳ�������нϿ�Χ�ĻҶ�ֵ��
// ��������չͼ���б�ѹ���ģ��Ҷ�ֵ�ϸ�����ģ�������ֵ��
cv::Mat getLogTransform3(cv::Mat srcImage, float c){
	// ����ͼ���ж�
	if (srcImage.empty()){
		std::cout << "No data!" << std::endl;
	}
	cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	// ͼ������ת��
	srcImage.convertTo(resultImage, CV_32F);
	// ͼ�����Ԫ�ؼ�1����
	resultImage = resultImage + 1;
	// ͼ���������
	cv::log(resultImage, resultImage);
	resultImage = c*resultImage;
	// ��һ������
	cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

// �Աȶ��������
cv::Mat getContrastStretchImage(cv::Mat srcImage){
	cv::Mat resultImage = srcImage.clone();
	if (srcImage.channels() == 1){
		int nRows = resultImage.rows;
		int nCols = resultImage.cols;
		// ͼ���������ж�
		if (resultImage.isContinuous()){
			nCols = nCols * nRows;
			nRows = 1;
		}
		// ͼ��ָ�����
		uchar* pDataMat;
		double pixMin = 0, pixMax = 255;
		minMaxLoc(resultImage, &pixMax, &pixMax);
		// �Աȶ�����ӳ��
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

// �Ҷȼ��ֲ�
// ������ȡ�ĸ���Ȥ����ĻҶ�ֵӳ������С������������Ȥ�ĻҶ�ֵ����ԭ��ֵ���䣬�������ͼ����Ϊ�Ҷ�ͼ��
cv::Mat getGrayLayeredImage(cv::Mat srcImage, int controlMin, int controlMax){
	cv::Mat resultImage = GrayTrans(srcImage);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	// ͼ���������ж�
	if (resultImage.isContinuous()){
		nCols = nCols * nRows;
		nRows = 1;
	}
	// ͼ��ָ�����
	uchar *pDataMat;
	// ����ͼ��ĻҶȼ��ֲ�
	for (int j = 0; j < nRows; j++){
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++){
			// ����ӳ��
			if (pDataMat[i] > controlMin && pDataMat[i] < controlMax)
				pDataMat[i] = controlMax;
		}
	}
	return resultImage;
}

// ��ûҶȱ���ƽ������
std::vector<cv::Mat> getMBitPlans(cv::Mat srcImage){
	// ���ǻҶ�ͼ��ת��Ϊ�Ҷ�ͼ
	Mat srcGray = GrayTrans(srcImage);
	int nRows = srcGray.rows;
	int nCols = srcGray.cols;
	// ͼ���������ж�
	if (srcGray.isContinuous()){
		nCols = nCols * nRows;
		nRows = 1;
	}
	// ͼ��ָ�����
	uchar *pSrcMat;
	uchar *pResultMat;
	cv::Mat resultImage = srcGray.clone();
	std::vector<cv::Mat> bitPlanes;
	bitPlanes.resize(8);
	int pixMax = 0, pixMin = 0;
	for (int n = 1; n <= 8; n++){
		// ����ƽ��ֲ����ع���
		pixMin = (int)pow(2.0, n - 1);
		pixMax = (int)pow(2.0, n);
		for (int j = 0; j < nRows; j++){
			// ��ȡͼ������ָ��
			pSrcMat = srcGray.ptr<uchar>(j);
			pResultMat = resultImage.ptr<uchar>(j);
			for (int i = 0; i < nCols; i++){
				//printf("pSrcMat(%d) = %d\n", i, pSrcMat[i]);
				// ��Ӧ����ƽ����ֵ��
				if (pSrcMat[i] >= pixMin && pSrcMat[i] < pixMax)
					pResultMat[i] = 255;
				else
					pResultMat[i] = 0;
			}
		}
		// ����ƽ������
		//char windowsName[20];
		//sprintf(windowsName, "BitPlane %d", n);
		bitPlanes[n - 1] = resultImage.clone();
		//imshow(windowsName, resultImage);
	}
	return bitPlanes;
}

// �������ֵ�ָ�
float calculateCurrentEntropy(cv::Mat hist, int threshold){
	float BackgroundSum = 0, targetSum = 0;
	const float* pDataHist = (float*)hist.ptr<float>(0);
	for (int i = 0; i < 256; i++){
		// �ۼƱ���ֵ
		if (i < threshold){
			BackgroundSum += pDataHist[i];
		}
		else {	// �ۼ�Ŀ��ֵ
			targetSum += pDataHist[i];
		}
	}
	// std::cout<< BackgroundSum <<" "<<targetSum<<std::endl;
	float BackgroundEntropy = 0, targetEntropy = 0;
	for (int i = 0; i < 256; i++){
		// ���㱳����
		if (i < threshold){
			if (pDataHist[i] == 0)
				continue;
			float ratio1 = pDataHist[i] / BackgroundSum;
			// ���㵱ǰ������
			BackgroundEntropy += -ratio1 * logf(ratio1);
		}
		else {	// ����Ŀ����
			if (pDataHist[i] == 0)
				continue;
			float ratio2 = pDataHist[i] / targetSum;
			targetEntropy += -ratio2 * logf(ratio2);
		}
	}
	return (targetEntropy + BackgroundEntropy);
}

// Ѱ���������ֵ���ָ�
cv::Mat maxEntropySegMentation(cv::Mat inputImage){
	Mat inputGray = GrayTrans(inputImage);
	cv::MatND hist;
	getHistgram(inputGray, hist);
	float maxentropy = 0;
	int max_index = 0;
	cv::Mat result;
	// �����õ��������ֵ�ָ�������ֵ
	for (int i = 0; i < 256; i++){

		float cur_entropy = calculateCurrentEntropy(hist, i);
		// ���㵱ǰ���ֵ��λ��
		if (cur_entropy > maxentropy){
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	printf("max_index=%d\n", max_index);
	// ��ֵ���ָ�
	threshold(inputGray, result, max_index, 255, CV_THRESH_BINARY);
	return result;
}

// ����ͼ�񲨷��
// ͶӰ���ߵĲ���/������ͨ���ж���һ�׵���Ϊ��㣬���׵���Ϊ����ֵ��ȷ���ģ�������һ�ײ��D��
// ���ǹ�ע����ͼ���ֵ�ֵ�Ĵ�С�����������Ҫ������з��Ż���Ȼ����ͨ��������ײ�ֵı仯��
// �ҵ�����б�ʵ�U�����������������������ɸ�������,�㼯U����ͶӰ���ߵĲ��岨��ֵ.
// ���ص�ͼ��Ϊ�ҵ���ͼ�񲨷�ͼ�� resultVec��¼�����еĲ�����������
cv::Mat findPeak(cv::Mat srcImage, vector<int>& resultVec, int thresh){
	cv::Mat verMat;
	cv::Mat resMat = srcImage.clone();
	// ��ֵ������
	//int threshType = 0;
	// Ԥ�����ֵ
	const int maxVal = 255;
	// �̶���ֵ������
	cv::threshold(srcImage, srcImage, thresh, maxVal, CV_THRESH_BINARY);
	imshow("threshold", srcImage);

	srcImage.convertTo(srcImage, CV_32FC1);
	// ���㴹ֱͶӰ
	cv::reduce(srcImage, verMat, 0, CV_REDUCE_SUM);
	//imshow("reduce", verMat);
	// std::cout<<verMat<<std::endl;
	// �������ַ��ź���
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
	// �Է��ź������б���
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
	// �����ж����
	for (vector<int>::size_type i = 0; i != tempVec.size() - 1; i++){
		if (tempVec[i + 1] - tempVec[i] == -2)
			resultVec.push_back(i + 1);
	}
	// �������λ��
	for (int i = 0; i < (int)resultVec.size(); i++){
		//std::cout << resultVec[i] << " ";
		// ����λ��Ϊ255
		resMat.col(resultVec[i]).setTo(Scalar::all(255));
	}
	return resMat;
}

// ��ô�ֱͶӰͼ�� ������а׵���� reduceMat �洢���Ǽ����� ��CV_32F��ʽ��
cv::Mat getVerticalProjImage(cv::Mat srcImage, Mat & reduceMat){
	// ����ͼ���ж�
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

// ���ˮƽͶӰͼ�� ������а׵���� reduceMat �洢���Ǽ����� ��CV_32F��ʽ��
cv::Mat getHorizontalProjImage(cv::Mat srcImage, Mat & reduceMat){
	// ����ͼ���ж�
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

// ͼ�����������
// �ϲ�ͼ�����²��ͨ�˲���ͨ���²����õ��ģ��������ԭ���Ĳ�ֵ��Ӧ���Ǹ�˹���������������Ϣ��
void Pyramid(cv::Mat srcImage){
	// ����ͼ��Դ�ߴ��ж��Ƿ���Ҫ����
	if (srcImage.rows > 400 && srcImage.cols > 400)
		cv::resize(srcImage, srcImage, cv::Size(), 0.5, 0.5);
	else // ����Ҫ��������
		cv::resize(srcImage, srcImage, cv::Size(), 1, 1);
	cv::imshow("srcImage", srcImage);
	cv::Mat pyrDownImage, pyrUpImage;
	// �²�������
	pyrDown(srcImage, pyrDownImage, cv::Size(srcImage.cols / 2, srcImage.rows / 2));
	cv::imshow("pyrDown", pyrDownImage);

	// �ϲ�������
	pyrUp(srcImage, pyrUpImage, cv::Size(srcImage.cols * 2, srcImage.rows * 2));
	cv::imshow("pyrUp", pyrUpImage);

	// ���²��������ع�
	cv::Mat pyrBuildImage;
	pyrUp(pyrDownImage, pyrBuildImage, cv::Size(pyrDownImage.cols * 2, pyrDownImage.rows * 2));
	cv::imshow("pyrBuildImage", pyrBuildImage);




	// �Ƚ��ع��������
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

// ͼ���������������ʵ��
// �������������������� ��ͨ�˲�
cv::Mat Myfilter2D(cv::Mat srcImage){
	const int nChannels = srcImage.channels();
	cv::Mat resultImage(srcImage.size(), srcImage.type());
	for (int j = 1; j < srcImage.rows - 1; j++){
		// ��ȡ����ָ��
		const uchar* previous = srcImage.ptr<uchar>(j - 1);
		const uchar* current = srcImage.ptr<uchar>(j);
		const uchar* next = srcImage.ptr<uchar>(j + 1);
		uchar * output = resultImage.ptr<uchar>(j);
		for (int i = nChannels; i < nChannels*(srcImage.cols - 1); ++i){
			// 4-�����ֵ�������
			*output++ = saturate_cast<uchar>((current[i - nChannels] + current[i + nChannels] + previous[i] + next[i]) / 4);
		}
	}
	// �߽紦��
	resultImage.row(0).setTo(Scalar(0));
	resultImage.row(resultImage.rows - 1).setTo(Scalar(0));
	resultImage.col(0).setTo(Scalar(0));
	resultImage.col(resultImage.cols - 1).setTo(Scalar(0));
	return resultImage;
}

// opencv�Դ���������� ��ͨ�˲�
cv::Mat filter2D_(cv::Mat srcImage){
	cv::Mat resultImage(srcImage.size(), srcImage.type());
	// ����˺�������
	Mat kern = (Mat_<float>(3, 3) << 0, 1, 0, 1, 0, 1, 0, 1, 0) / (float)(4);
	filter2D(srcImage, resultImage, srcImage.depth(), kern);
	return resultImage;
}

// ͼ����Ҷ�任
cv::Mat DFT(cv::Mat srcImage){
	cv::Mat srcGray = GrayTrans(srcImage);

	// ������ͼ����������ѵĳߴ�
	int nRows = getOptimalDFTSize((srcGray.rows + 1)/2*2);
	int nCols = getOptimalDFTSize((srcGray.cols+1)/2*2);
	cv::Mat resultImage;
	// �ѻҶ�ͼ��������Ͻǣ����ұߺ��±���չͼ��
	// ����ӵ����س�ʼ��Ϊ0
	copyMakeBorder(srcGray, resultImage, 0, nRows - srcGray.rows, 0, nCols - srcGray.cols,
		BORDER_CONSTANT, Scalar::all(0));
	// Ϊ����Ҷ�任�Ľ����ʵ�����鲿������洢�ռ�
	cv::Mat planes[] = { cv::Mat_<float>(resultImage), cv::Mat::zeros(resultImage.size(), CV_32F) };
	Mat completeI;
	// Ϊ�������ͼ������һ����ʼ��Ϊ0��ͨ��
	merge(planes, 2, completeI);
	// ������ɢ����Ҷ�任
	dft(completeI, completeI);
		// ������ת��Ϊ����
	split(completeI, planes);
	magnitude(planes[0], planes[1], planes[0]);

	//saveMat("planes.xml", planes[0]);
	cv::Mat dftResultImage = planes[0];
	// �����߶�(logarithmic scale ����
	dftResultImage += 1;
	log(dftResultImage, dftResultImage);
	dftResultImage = fftshift(dftResultImage);
	//// ���к��طֲ�����ͼ����
	//dftResultImage = dftResultImage(Rect(0, 0, srcGray.cols, srcGray.rows));
	//// ��һ��ͼ��
	//normalize(dftResultImage, dftResultImage, 0, 1, CV_MINMAX);

	//int cx = dftResultImage.cols / 2;
	//int cy = dftResultImage.rows / 2;
	//Mat tmp;
	//// Top-left����Ϊÿһ�����޴���ROI
	//Mat q0(dftResultImage, Rect(0, 0, cx, cy));
	//// Top-Right
	//Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
	//// Bottom-Left
	//Mat q2(dftResultImage, Rect(0, cy, cx, cy));
	//// Bottom����Right
	//Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
	//// �������� (Top-Left with Bottom-Right)
	//q0.copyTo(tmp);
	//q3.copyTo(q0);
	//tmp.copyTo(q3);
	//// ��������(Top-Right with Bottom-Left)
	//q1.copyTo(tmp);
	//q2.copyTo(q1);
	//tmp.copyTo(q2);
	return dftResultImage;
}

// ͼ����Ҷ�任
cv::Mat DCT(cv::Mat srcImage){
	Mat src = GrayTrans(srcImage);

	src.convertTo(src, CV_64FC1);

	//DCTϵ��������ͨ��    
	Mat dctImage(src.size(), CV_64FC1);

	//DCT�任    
	dct(src, dctImage);
	return dctImage;
}


//// ͼ����Ҷ��任
//cv::Mat INV_DFT(cv::Mat srcImage){
//	cv::Mat srcGray = GrayTrans(srcImage);
//
//	Mat image_Re = Mat::zeros(srcGray.size(), CV_64FC1);
//	Mat image_Im = Mat::zeros(srcGray.size(), CV_64FC1);
//
//	// ������ͼ����������ѵĳߴ�
//	int nRows = getOptimalDFTSize(srcGray.rows);
//	int nCols = getOptimalDFTSize(srcGray.cols);
//
//	cv::Mat resultImage;
//	// �ѻҶ�ͼ��������Ͻǣ����ұߺ��±���չͼ��
//	// ����ӵ����س�ʼ��Ϊ0
//	copyMakeBorder(srcGray, resultImage, 0, nRows - srcGray.rows, 0, nCols - srcGray.cols,
//		BORDER_CONSTANT, Scalar::all(0));
//	// Ϊ����Ҷ�任�Ľ����ʵ�����鲿������洢�ռ�
//	cv::Mat planes[] = { cv::Mat_<float>(resultImage), cv::Mat::zeros(resultImage.size(), CV_32F) };
//	Mat completeI;
//	// Ϊ�������ͼ������һ����ʼ��Ϊ0��ͨ��
//	merge(planes, 2, completeI);
//	// ������ɢ����Ҷ�任
//	dft(completeI, completeI);
//	// ������ת��Ϊ����
//	split(completeI, planes);
//	magnitude(planes[0], planes[1], planes[0]);
//	cv::Mat dftResultImage = planes[0];
//	// �����߶�(logarithmic scale ����
//	dftResultImage += 1;
//	log(dftResultImage, dftResultImage);
//	// ���к��طֲ�����ͼ����
//	dftResultImage = dftResultImage(Rect(0, 0, srcGray.cols, srcGray.rows));
//	// ��һ��ͼ��
//	normalize(dftResultImage, dftResultImage, 0, 1, CV_MINMAX);
//
//	int cx = dftResultImage.cols / 2;
//	int cy = dftResultImage.rows / 2;
//	Mat tmp;
//	// Top-left����Ϊÿһ�����޴���ROI
//	Mat q0(dftResultImage, Rect(0, 0, cx, cy));
//	// Top-Right
//	Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
//	// Bottom-Left
//	Mat q2(dftResultImage, Rect(0, cy, cx, cy));
//	// Bottom����Right
//	Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
//	// �������� (Top-Left with Bottom-Right)
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//	// ��������(Top-Right with Bottom-Left)
//	q1.copyTo(tmp);
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//	return dftResultImage;
//}

// ͼ��������
cv::Mat convolution(cv::Mat srcImage, cv::Mat kernel){
	Mat srcGray = GrayTrans(srcImage);
	srcGray.convertTo(srcGray, CV_32F);
	// ���ͼ����
	Mat dst = Mat::zeros(abs(srcGray.rows - kernel.rows) + 1, abs(srcGray.cols - kernel.cols) + 1, srcGray.type());
	cv::Size dftSize;
	// ���㸵��Ҷ�任�ߴ�
	dftSize.width = getOptimalDFTSize(srcGray.cols + kernel.cols - 1);
	dftSize.height = getOptimalDFTSize(srcGray.rows + kernel.rows - 1);

	// ������ʱͼ�񣬳�ʼ��Ϊ0
	cv::Mat tempA(dftSize, srcGray.type(), Scalar::all(0));
	cv::Mat tempB(dftSize, kernel.type(), Scalar::all(0));
	// ��������и���
	cv::Mat roiA(tempA, Rect(0, 0, srcGray.cols, srcGray.rows));
	srcGray.copyTo(roiA);
	cv::Mat roiB(tempB, Rect(0, 0, kernel.cols, kernel.rows));
	kernel.copyTo(roiB);

	// ����Ҷ�任
	dft(tempA, tempA, 0, srcGray.rows);
	dft(tempB, tempB, 0, kernel.rows);
	// ��Ƶ���е�ÿ��Ԫ�ؽ��гͷ�����
	mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT);
	// �任���,�����з���
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, dst.rows);
	// ���ƽ�������ͼ��
	tempA(Rect(0, 0, dst.cols, dst.rows)).copyTo(dst);
	normalize(dst, dst, 0, 1, CV_MINMAX);
	return dst;
}

// ��ֵ�˲�
cv::Mat getBlurImage(const Mat& src, Size ksize){
	if (!src.data) {
		printf("��MyBlur::getBlurImage(src, dst, ksize)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize.height % 2 != 1 || ksize.width % 2 != 1) {
		printf("��MyBlur::getBlurImage(src, dst, ksize)����ksize�ĳ��Ϳ����Ϊ����! ! !��");
		return Mat();
	}
	Mat dst;
	blur(src, dst, ksize);
	return dst;
}

// ��ֵ�˲�
cv::Mat getMedianBlurImage(const Mat& src, Size ksize){
	if (!src.data) {
		printf("��MyBlur::getMedianBlurImage(src, dst, ksize)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}

	if (ksize.height % 2 != 1 || ksize.width % 2 != 1) {
		printf("��MyBlur::getMedianBlurImage(src, dst, ksize)����ksize�ĳ��Ϳ����Ϊ����! ! !��");
		return Mat();
	}
	Mat dst;
	medianBlur(src, dst, ksize.width);
	return dst;
}

// ��˹�˲�
cv::Mat getGaussianBlurImage(const Mat& src, Size ksize, double sigmaX, double sigmaY){
	if (!src.data) {
		printf("��MyBlur::getGaussianBlurImage(src, dst, ksize, sigmaX, sigmaY)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize.height % 2 != 1 || ksize.width % 2 != 1) {
		printf("��MyBlur::getGaussianBlurImage(src, dst, ksize)����ksize�ĳ��Ϳ����Ϊ����! ! !��");
		return Mat();
	}
	Mat dst;
	GaussianBlur(src, dst, ksize, sigmaX, sigmaY);
	return dst;
}

// ˫���˲�
cv::Mat getBilateralFilterImage(const Mat& src, int d, double sigmaColor, double sigmaSpace){
	if (!src.data) {
		printf("��MyBlur::getBilateralFilterImage(src, dst, ksize, sigmaColor, sigmaSpace)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	Mat dst;
	bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);
	return dst;
}

// ͼ�����˲�
cv::Mat guidefilter(Mat &srcImage, int r, double eps){
	if (srcImage.empty()){
		printf("MyImage.cpp guidefilter ����ͼ��Ϊ��!\n");
		return Mat();
	}
	if (srcImage.channels() == 3){
		// ͨ������
		vector<Mat> vSrcImage, vResultImage;
		split(srcImage, vSrcImage);
		Mat resultMat;
		for (int i = 0; i < 3; i++){
			printf("i = %d\n", i);
			// ��ͨ��ת���ɸ���������
			Mat tempImage;
			vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
			Mat p = tempImage.clone();
			// �ֱ���е����˲�
			printf("convertTo success!\n");

			// ת��Դͼ����Ϣ
			tempImage.convertTo(tempImage, CV_64FC1);
			p.convertTo(p, CV_64FC1);
			printf("ת��ԭͼ����Ϣ�ɹ�!\n");

			int nRows = tempImage.rows;
			int nCols = tempImage.cols;
			cv::Mat boxResult;
			// ����һ�� �����ֵ
			cv::boxFilter(cv::Mat::ones(nRows, nCols, tempImage.type()), boxResult, CV_64FC1, cv::Size(r, r));
			// ���ɵ����ֵmean_I
			cv::Mat mean_I;
			cv::boxFilter(tempImage, mean_I, CV_64FC1, cv::Size(r, r));
			// ����ԭʼ��ֵmean_p
			cv::Mat mean_p;
			cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
			// ���ɻ���ؾ�ֵmean_Ip
			cv::Mat mean_Ip;
			cv::boxFilter(tempImage.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
			cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
			// ��������ؾ�ֵmean_II
			cv::Mat mean_II;
			// Ӧ�ú��˲���������ؾ�ֵ
			cv::boxFilter(tempImage.mul(tempImage), mean_II, CV_64FC1, cv::Size(r, r));
			printf("Step 1 done!\n");

			// �����: �������ϵ��
			cv::Mat var_I = mean_II - mean_I.mul(mean_I);
			cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
			printf("Step 2 done!\n");

			// ������: �������ϵ��a��b
			cv::Mat a = cov_Ip / (var_I + eps);
			cv::Mat b = mean_p - a.mul(mean_I);
			printf("Step 3 done!\n");
			// ������: ����ϵ��a��b�ľ�ֵ
			cv::Mat mean_a;
			cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
			mean_a = mean_a / boxResult;
			cv::Mat mean_b;
			cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
			mean_b = mean_b / boxResult;
			printf("Step 4 done!\n");
			// ������: �����������
			cv::Mat resultImage = mean_a.mul(tempImage) + mean_b;


			vResultImage.push_back(resultImage);
		}
		// ͨ������ϲ�
		merge(vResultImage, resultMat);
		return resultMat;
	}
	else {
		Mat tempImage;
		srcImage.convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
		Mat p = tempImage.clone();
		// �ֱ���е����˲�
		printf("convertTo success!\n");

		// ת��Դͼ����Ϣ
		tempImage.convertTo(tempImage, CV_64FC1);
		p.convertTo(p, CV_64FC1);
		printf("ת��ԭͼ����Ϣ�ɹ�!\n");

		int nRows = tempImage.rows;
		int nCols = tempImage.cols;
		cv::Mat boxResult;
		// ����һ�� �����ֵ
		cv::boxFilter(cv::Mat::ones(nRows, nCols, tempImage.type()), boxResult, CV_64FC1, cv::Size(r, r));
		// ���ɵ����ֵmean_I
		cv::Mat mean_I;
		cv::boxFilter(tempImage, mean_I, CV_64FC1, cv::Size(r, r));
		// ����ԭʼ��ֵmean_p
		cv::Mat mean_p;
		cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
		// ���ɻ���ؾ�ֵmean_Ip
		cv::Mat mean_Ip;
		cv::boxFilter(tempImage.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
		cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
		// ��������ؾ�ֵmean_II
		cv::Mat mean_II;
		// Ӧ�ú��˲���������ؾ�ֵ
		cv::boxFilter(tempImage.mul(tempImage), mean_II, CV_64FC1, cv::Size(r, r));
		printf("Step 1 done!\n");

		// �����: �������ϵ��
		cv::Mat var_I = mean_II - mean_I.mul(mean_I);
		cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
		printf("Step 2 done!\n");

		// ������: �������ϵ��a��b
		cv::Mat a = cov_Ip / (var_I + eps);
		cv::Mat b = mean_p - a.mul(mean_I);
		printf("Step 3 done!\n");
		// ������: ����ϵ��a��b�ľ�ֵ
		cv::Mat mean_a;
		cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
		mean_a = mean_a / boxResult;
		cv::Mat mean_b;
		cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
		mean_b = mean_b / boxResult;
		printf("Step 4 done!\n");
		// ������: �����������
		cv::Mat resultImage = mean_a.mul(tempImage) + mean_b;
		return resultImage;
	}
}

// ��ֱ�Ե���ʵ��
void diffOperation(const cv::Mat srcImage, cv::Mat& edgeXImage, cv::Mat& edgeYImage){
	if (srcImage.empty()){
		printf("MyImage.cpp diffOperation ����ͼ��������!!!\n");
	}
	cv::Mat tempImage = GrayTrans(srcImage);
	edgeXImage.create(tempImage.size(), tempImage.type());
	edgeYImage.create(tempImage.size(), tempImage.type());
	int nRows = tempImage.rows;
	int nCols = tempImage.cols;
	for (int i = 0; i < nRows - 1; i++){
		for (int j = 0; j < nCols - 1; j++){
			// ���㴹ֱ�߱�Ե
			edgeXImage.at<uchar>(i, j) = abs(tempImage.at<uchar>(i + 1, j) - tempImage.at<uchar>(i, j));
			// ����ˮƽ��Ե
			edgeYImage.at<uchar>(i, j) = abs(tempImage.at<uchar>(i, j + 1) - tempImage.at<uchar>(i, j));
		}
	}
}

// ͼ��Ǽ���ֵ����Sobel��Եʵ��
cv::Mat getSobelVerEdge(cv::Mat srcImage){
	CV_Assert(srcImage.channels() == 1);
	srcImage.convertTo(srcImage, CV_32FC1);
	// ˮƽ�����Sobel����
	cv::Mat sobelx = (cv::Mat_<float>(3, 3) << -0.125, 0, 0.125,
		-0.25, 0, 0.25,
		-0.125, 0, 0.125);
	cv::Mat ConResMat;
	// �������
	cv::filter2D(srcImage, ConResMat, srcImage.type(), sobelx);
	// �����ݶȵķ���
	cv::Mat graMagMat;
	cv::multiply(ConResMat, ConResMat, graMagMat);
	// �����ݶȷ��ȼ�����������ֵ
	int scaleVal = 4;
	double thresh = scaleVal * cv::mean(graMagMat).val[0];
	cv::Mat resultTempMat = cv::Mat::zeros(graMagMat.size(), graMagMat.type());
	float *pDataMag = (float*)graMagMat.data;
	float* pDataRes = (float*)resultTempMat.data;
	const int nRows = ConResMat.rows;
	const int nCols = ConResMat.cols;
	for (int i = 1; i != nRows - 1; i++){
		for (int j = 1; j != nCols - 1; j++){
			// ����õ��ݶ���ˮƽ��ֱ�ݶ�ֵ�ô�С���ȽϽ��
			bool b1 = (pDataMag[i*nCols + j] > pDataMag[i*nCols + j - 1]);
			bool b2 = (pDataMag[i*nCols + j] > pDataMag[i*nCols + j + 1]);
			bool b3 = (pDataMag[i*nCols + j] > pDataMag[(i - 1)*nCols + j]);
			bool b4 = (pDataMag[i*nCols + j] > pDataMag[(i + 1)*nCols + j]);

			// �ж������ݶ��Ƿ��������ˮƽ��ֱ�ݶȵ�����
			// ����������Ӧ��ֵ�������ж�ֵ��
			pDataRes[i*nCols + j] = (float)(255 * ((pDataMag[i*nCols + j] > thresh) && ((b1&&b2) || (b3&&b4))));
		}
	}
	resultTempMat.convertTo(resultTempMat, CV_8UC1);
	Mat resultImage = resultTempMat.clone();
	return resultImage;
}

// ͼ��ֱ�Ӿ��Sobel��Եʵ�� ģ����ֵΪ�ݶȷ�ֵ����ֵ
cv::Mat getsobelEdge(const cv::Mat& srcImage, uchar threshold){
	CV_Assert(srcImage.channels() == 1);
	// ��ʼ��ˮƽ������
	Mat sobelx = (Mat_<double>(3, 3) << 1, 0,
		-1, 2, 0, -2, 1, 0, -1);
	// ��ʼ����ֱ������
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
			// ��������ˮƽ�봹ֱ�ݶ�
			for (int i = -1; i <= 1; ++i){
				for (int j = -1; j <= 1; j++){
					edgeX += srcImage.at<uchar>(k + i, n + j)*sobelx.at<double>(1 + i, 1 + j);
					edgeY += srcImage.at<uchar>(k + i, n + j)*sobely.at<double>(1 + i, 1 + j);
				}
			}
			// �����ݶ�ģ��
			graMag = sqrt(pow(edgeY, 2) + pow(edgeX, 2));
			// ��ֵ��
			resultImage.at<uchar>(k - 1, n - 1) = ((graMag > threshold) ? 255 : 0);
		}
	}
	return resultImage;
}

// ͼ�����·Ǽ���ֵ����Sobelʵ��
// flag = 0 �����ݶ� 
// flag = 1 �����ݶ�
// flag = 2 ȫ����ݶ�
cv::Mat getsobelOptaEdge(const cv::Mat& srcImage, int flag){
	CV_Assert(srcImage.channels() == 1);
	// ��ʼ��Sobelˮƽ������
	cv::Mat sobelX = (cv::Mat_<double>(3, 3) << 1, 0, -1,
		2, 0, -2,
		1, 0, -1);
	// ��ʼ��Sobel��ֱ������
	cv::Mat sobelY = (cv::Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0,
		-1, -2, -1);
	// ����ˮƽ�봹ֱ���
	cv::Mat edgeX, edgeY;
	filter2D(srcImage, edgeX, CV_32F, sobelX);
	filter2D(srcImage, edgeY, CV_32F, sobelY);
	// ���ݴ������ȷ������ˮƽ��ֱ��Ե
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
	// ������ֵ
	int scaleVal = 4;
	double thresh = scaleVal * cv::mean(graMagMat).val[0];
	Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
	for (int i = 1; i < srcImage.rows - 1; i++){
		float *pDataEdgeX = edgeX.ptr<float>(i);
		float *pDataEdgeY = edgeY.ptr<float>(i);
		float *pDataGraMag = graMagMat.ptr<float>(i);
		// ��ֵ���ͼ���ֵ����
		for (int j = 1; j < srcImage.cols - 1; j++){
			// �жϵ�ǰ�����ݶ��Ƿ������ֵ�����ˮƽ��ֱ�ݶ�
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

// OpenCV�Դ���ͼ���Ե���� 
// flag��ȡ�����ֵ
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
		// ���Ա任��ת����������Ԫ��Ϊ8λ�޷�������
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

// ��ȡLaplace��Ե
cv::Mat getLaplaceEdge(cv::Mat srcImage){
	CV_Assert(!srcImage.empty());
	// ��˹ƽ��
	GaussianBlur(srcImage, srcImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cv::Mat dstImage;
	// ������˹�任
	Laplacian(srcImage, dstImage, CV_16S, 3);
	convertScaleAbs(dstImage, dstImage);
	return dstImage;
}

// Robert��Ե���
// Robert���������þֲ����Ѱ�ұ�Ե��һ�����ӣ�����򵥵ı�Ե������ӡ�Roberts�������öԽ���
// ��������������֮������ݶȷ�ֵ������Ե����ⴹֱ��Ե��Ч��Ҫ�������������Ե����λ���ȸߣ�
// ��������������������������Ե������Ӽ��ÿ�����ص����򲢶ԻҶȱ仯������������ͬʱҲ����
// �����ȷ����
cv::Mat getRobertsEdge(cv::Mat srcImage){
	cv::Mat dstImage = srcImage.clone();
	int nRows = dstImage.rows;
	int nCols = dstImage.cols;
	for (int i = 0; i < nRows - 1; i++){
		for (int j = 0; j < nCols - 1; j++){
			// ���ݹ�ʽ����
			int t1 = (srcImage.at<uchar>(i, j) - srcImage.at<uchar>(i + 1, j + 1)) *
				(srcImage.at<uchar>(i, j) - srcImage.at<uchar>(i + 1, j + 1));
			int t2 = (srcImage.at<uchar>(i + 1, j) - srcImage.at<uchar>(i, j + 1)) *
				(srcImage.at<uchar>(i + 1, j) - srcImage.at<uchar>(i, j + 1));
			// ����Խ������ز�
			dstImage.at<uchar>(i, j) = (uchar)sqrt((double)(t1 + t2));
		}
	}
	return dstImage;
}

// Prewitt��Ե���
// Prewitt������һ�ױ�Ե������ӣ������Ӷ��������������á�Prewitt���ӶԱ�Ե�Ķ�λ���Ȳ���Roberts���ӣ�
// Sobel���ӶԱ�Ե����׼ȷ�Ը�����Prewitt���ӡ�
cv::Mat getPrewittEdge(cv::Mat srcImage, bool verFlag){
	srcImage.convertTo(srcImage, CV_32FC1);
	cv::Mat prewitt_kernel = (cv::Mat_<float>(3, 3) << 0.1667, 0.1667, 0.1667,
		0, 0, 0,
		-0.1667, -0.1667, -0.1667);
	// ��ֱ��Ե
	if (verFlag){
		prewitt_kernel = prewitt_kernel.t();
		cv::Mat z1 = cv::Mat::zeros(srcImage.rows, 1, CV_32FC1);
		cv::Mat z2 = cv::Mat::zeros(1, srcImage.cols, CV_32FC1);
		// ��ͼ����ı���Ϊ0
		z1.copyTo(srcImage.col(0));
		z1.copyTo(srcImage.col(srcImage.cols - 1));
		z2.copyTo(srcImage.row(0));
		z2.copyTo(srcImage.row(srcImage.rows - 1));
	}
	cv::Mat edges;
	cv::filter2D(srcImage, edges, srcImage.type(), prewitt_kernel);
	cv::Mat mag;
	cv::multiply(edges, edges, mag);
	// ȥ����ֱ��Ե
	if (verFlag){
		cv::Mat black_region = srcImage < 0.03;
		cv::Mat se = cv::Mat::ones(5, 5, CV_8UC1);
		cv::dilate(black_region, black_region, se);
		mag.setTo(0, black_region);
	}
	// ����ģ��������ݶȵ���ֵ
	double thresh = 4.0f * cv::mean(mag).val[0];
	// ����ĳ���ݶȴ��ڷ����ֱ������ڵ��ݶ�ʱ
	// �����λ�õ����ֵΪ255
	// ��Ӧ����ֵthresh
	cv::Mat dstImage = cv::Mat::zeros(mag.size(), mag.type());
	float *dptr = (float*)mag.data;
	float *tptr = (float*)dstImage.data;
	int r = edges.rows, c = edges.cols;
	for (int i = 1; i != r - 1; ++i){
		for (int j = 1; j != c - 1; ++j){
			// �Ǽ���ֵ����
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

// Canny�⺯��ʵ�� �Ƽ��ĸ������ֵ��ֵ��2:1��3:1֮��
cv::Mat getCannyEdge(cv::Mat srcImage, int lowThresh, int highThresh){
	CV_Assert(!srcImage.empty());
	Mat resultImage;
	// Canny���
	Canny(srcImage, resultImage, lowThresh, highThresh, 3);
	return resultImage;
}

// �Ľ���Ե�������Marr-Hildreth LoG����
// ���Ѹ�˹ƽ���˲�����������˹���˲��������������ƽQ�����������ٽ��б�Ե���
cv::Mat getMarrEdge(const Mat src, int kerValue, double delta){
	// ����LOG����
	Mat kernel;
	// �뾶
	int kerLen = kerValue / 2;
	kernel = Mat_<double>(kerValue, kerValue);
	// ����
	for (int i = -kerLen; i <= kerLen; i++){
		for (int j = -kerLen; j <= kerLen; j++){
			// ���ɺ�����
			kernel.at<double>(i + kerLen, j + kerLen) =
				exp(-((pow(j, 2.0) + pow(i, 2.0)) /
				(pow(delta, 2.0) * 2)))
				*(((pow(j, 2.0) + pow(i, 2.0) - 2 *
				pow(delta, 2.0)) / (2 * pow(delta, 4.0))));
		}
	}
	// �����������
	int kerOffset = kerValue / 2;
	Mat laplacian = (Mat_<double>(src.rows - kerOffset * 2, src.cols - kerOffset * 2));
	Mat result = Mat::zeros(src.rows - kerOffset * 2, src.cols - kerOffset * 2, src.type());
	double sumLaplacian;
	// ����������ͼ���������˹����
	for (int i = kerOffset; i < src.rows - kerOffset; ++i){
		for (int j = kerOffset; j < src.cols - kerOffset; ++j){
			sumLaplacian = 0;
			for (int k = -kerOffset; k <= kerOffset; ++k){
				for (int m = -kerOffset; m <= kerOffset; ++m){
					// ����ͼ����
					sumLaplacian += src.at<uchar>(i + k, j + m)*kernel.at<double>(kerOffset + k, kerOffset + m);
				}
			}
			// ����������˹���
			laplacian.at<double>(i - kerOffset, j - kerOffset) = sumLaplacian;
		}
	}

	// ����㽻�棬Ѱ�ұ�Ե����
	for (int y = 1; y < result.rows - 1; y++){
		for (int x = 1; x < result.cols - 1; x++){
			result.at<uchar>(y, x) = 0;
			// �����ж�
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

// MoravecCorners�ǵ���
cv::Mat MoravecCorners(cv::Mat srcImage, vector<Point> & points, int kSize, int threshold){
	cv::Mat resMorMat = convert2BGR(srcImage);
	// ��ȡ��ʼ��������Ϣ
	int r = kSize / 2;
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	int nCount = 0;

	// ͼ�����
	for (int i = r; i < srcImage.rows - r; i++){
		for (int j = r; j < srcImage.cols - r; j++){
			int wV1, wV2, wV3, wV4;
			wV1 = wV2 = wV3 = wV4 = 0;
			// ����ˮƽ�����ڵ���Ȥֵ
			for (int k = -r; k < r; k++){
				wV1 += (srcImage.at<uchar>(i, j + k) - srcImage.at<uchar>(i, j + k + 1)) * (srcImage.at<uchar>(i, j + k) - srcImage.at<uchar>(i, j + k + 1));
			}
			// ���㴹ֱ�����ڵ���Ȥֵ
			for (int k = -r; k < r; k++){
				wV2 += (srcImage.at<uchar>(i + k, j) - srcImage.at<uchar>(i + k + 1, j)) * (srcImage.at<uchar>(i + k, j) - srcImage.at<uchar>(i + k + 1, j));
			}
			// ����45�㷽���ڵ���Ȥֵ
			for (int k = -r; k < r; k++){
				wV3 += (srcImage.at<uchar>(i + k, j + k) - srcImage.at<uchar>(i + k + 1, j + k + 1)) * (srcImage.at<uchar>(i + k, j + k) - srcImage.at<uchar>(i + k + 1, j + k + 1));
			}
			// ����135�㷽���ڵ���Ȥֵ
			for (int k = -r; k < r; k++){
				wV4 += (srcImage.at<uchar>(i + k, j - k) - srcImage.at<uchar>(i + k + 1, j - k - 1)) * (srcImage.at<uchar>(i + k, j - k) - srcImage.at<uchar>(i + k + 1, j - k - 1));
			}

			// ȡ���е���Сֵ��Ϊ�����ص��������Ȥֵ
			int value = min(min(wV1, wV2), min(wV3, wV4));
			// ����Ȥֵ������ֵ���򽫵���������������
			if (value > threshold){
				points.push_back(Point(j, i));
				nCount++;
			}
		}
	}
	drawVecPoints(resMorMat, points);
	return resMorMat;
}


// ����Harris�ǵ�
Mat getHarrisCornersImage(const Mat& srcImage, float thresh, int blockSize, int kSize, double k){
	CV_Assert(!srcImage.empty());
	Mat src = GrayTrans(srcImage);
	Mat result(src.size(), CV_32F);
	int depth = src.depth();
	// �����ģ�ߴ�
	double scale = (double)(1 << ((kSize > 0 ? kSize : 3) - 1))*blockSize;
	if (depth == CV_8U)
		scale *= 255.;
	scale = 1. / scale;
	// Sobel�˲�
	Mat dx, dy;
	Sobel(src, dx, CV_32F, 1, 0, kSize, scale, 0);
	Sobel(src, dy, CV_32F, 0, 1, kSize, scale, 0);
	Size size = src.size();
	cv::Mat cov(size, CV_32FC3);

	// ���ˮƽ����ֱ�ݶ�
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
	// ��ͼ����к��˲�����
	boxFilter(cov, cov, cov.depth(), Size(blockSize, blockSize), Point(-1, -1), false);
	// �ж�ͼ��������
	if (cov.isContinuous() && result.isContinuous()){
		size.width *= size.height;
		size.height = 1;
	}
	else
		size = result.size();
	// ������Ӧ����
	for (int i = 0; i < size.height; i++){
		// ��ȡͼ�����ָ��
		float *resultData = (float*)(result.data + i*result.step);
		const float* covData = (const float*)(cov.data + i*cov.step);
		for (int j = 0; j < size.width; j++){
			// ������Ӧ����
			float a = covData[3 * j];
			float b = covData[3 * j + 1];
			float c = covData[3 * j + 2];
			resultData[j] = (float)(a*c - b*b - k*(a + c) * (a + c));
		}
	}

	Mat drawing = convert2BGR(srcImage);
	// �����һ��
	normalize(result, result, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(result, result);
	//printf("drawing go!\n");

	// ���ƽǵ�����
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

// �õ��Զ����
// ����һ int   kshape	: ��ʾ�ں˵���״��������ѡ��<1> ���Σ� MORPH_RECT; <2> ������: MORPH_CROSS; <3> Բ��: MORPH_ELLIPSE
// ������ int   ksize	: ��ʾ�ں˵ĳߴ�
// ������ Point kpos		: ��ʾê���λ��
Mat getCustomKernel(int ksize, int kshape, Point kpos) {
	// Ĭ�ϵ�ê�����ں˵����ĵ�
	if (kpos.x == -1){
		kpos.x = ksize / 2;
		kpos.y = ksize / 2;
	}

	Mat element = getStructuringElement(kshape,
		Size(ksize, ksize),
		Point(kpos.x, kpos.y));
	return element;
}

// ��ñ���� Top Hat �ֳơ���ñ������, �õ���Ч��ͼͻ���˱�ԭͼ������Χ����������������򣬶�ñ������������������ڽ�����һЩ�İ߿顣
// ��һ��ͼ����д���ı�������΢С��Ʒ�Ƚ��й��ɵ�����£�����ʹ�ö�ñ������б�����ȡ.
// dst = tophat(src, dst) = src = open(src, element)
Mat getMorphTopHatImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("��MyImage::getMorphTopHatImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getMorphTopHatImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_TOPHAT, element);
	return dst;
}

// ��ñ���� Black Hat ,�õ���Ч��ͼͻ���˱�ԭͼ������Χ��������������򣬺�ñ��������������ڽ��㰵һЩ�İ߿飬Ч��ͼ���ŷǳ�������������
Mat getMorphBlackHatImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("��MyImage::getMorphBlackHatImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getMorphBlackHatImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_BLACKHAT, element);
	return dst;
}

// ��̬ѧ�ݶ�
// dst = morph-grad(src,element) = dilate(src,element) - erode(src,element)
Mat getMorphGradientImage(const Mat& src, int ksize, int kshape, Point kpos) {
	if (!src.data) {
		printf("��MyImage::getMorphGradientImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getMorphGradientImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_GRADIENT, element);
	return dst;
}

// ������
// dst = open(src, element) = dilate(erode(src, element))
// �����������������С���壬����ϸ�㴦�������壬������ƽ���ϴ�����ı߽��ͬʱ�����Ըı��������
Mat getOpeningOperationImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("��MyImage::getOpeningOperationImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getOpeningOperationImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_OPEN, element);
	return dst;
}

// ������
// dst = close(src, element) = erode(dilate(src, element))
// �������ܹ��ų�С�ͺڶ�����ɫ����
Mat getClosingOperationImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("��MyImage::getClosingOperationImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getClosingOperationImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}
	Mat element = getCustomKernel(ksize, kshape, kpos);
	Mat dst;
	morphologyEx(src, dst, MORPH_CLOSE, element);
	return dst;
}

// �õ���ʴ��ͼ��
Mat getErodeImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("��MyImage::getErodeImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getErodeImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}

	Mat ele = getStructuringElement(kshape, Size(ksize, ksize), kpos);
	Mat dst;
	erode(src, dst, ele);
	return dst;
}

// �õ����͵�ͼ��
Mat getDilateImage(const Mat& src, int ksize, int kshape, Point kpos){
	if (!src.data) {
		printf("��MyImage::getDilateImage(src, dst, ksize, shape, kpos)����src����ͼƬ����Ϊ�գ�������");
		return Mat();
	}
	if (ksize % 2 != 1 || ksize % 2 != 1) {
		printf("��MyImage::getDilateImage(src, dst, ksize, shape, kpos)����ksize����Ϊ����! ! !��");
		return Mat();
	}

	Mat ele = getStructuringElement(kshape, Size(ksize, ksize), kpos);
	Mat dst;
	dilate(src, dst, ele);
	return dst;
}

// ��ˮ��ͼ��ָ�
Mat getWatershedSegmentImage(Mat &srcImage, int& noOfSegments, Mat& markers){
	Mat grayMat = GrayTrans(srcImage);
	Mat otsuMat;
	//imshow("graymat", grayMat);
	// ��ֵ����
	threshold(grayMat, otsuMat, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	//imshow("otsuMat", otsuMat);

	// ��̬ѧ������
	morphologyEx(otsuMat, otsuMat, MORPH_OPEN, Mat::ones(7, 7, CV_8SC1), Point(4, 4), 2);
	//imshow("Mor-openMat", otsuMat);
	// ����任
	Mat disTranMat(otsuMat.size(), CV_32FC1);
	distanceTransform(otsuMat, disTranMat, CV_DIST_L2, 3);
	// ��һ��
	normalize(disTranMat, disTranMat, 0.0, 1, NORM_MINMAX);
	//imshow("DistranMat", disTranMat);
	// ��ֵ���ָ�ͼ��
	threshold(disTranMat, disTranMat, 0.1, 1, CV_THRESH_BINARY);
	// ��һ��ͳ��ͼ��0~255
	normalize(disTranMat, disTranMat, 0.0, 255.0, NORM_MINMAX);
	disTranMat.convertTo(disTranMat, CV_8UC1);
	//imshow("TDisTranMat", disTranMat);
	// �����ǵķָ��
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
	// ���������
	for (; idx >= 0; idx = hierarchy[idx][0], compCount++){
		drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
	}
	if (compCount == 0)
		return Mat();
	// �����㷨��ʱ�临�Ӷ�
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

// ��ʾ��ˮ��ָ��㷨���ͼ��
Mat showWaterSegResult(Mat markers){
	Mat wshed;
	markers.convertTo(wshed, CV_8U);
	wshed = getContrastStretchImage(wshed);
	imshow("wshed", wshed);
	return wshed;
}


// �ָ�ϲ�
void segMerge(Mat& image, Mat& segments, int & numSeg){
	// ��һ���ָ�ֽ�������ͳ��
	vector<Mat> samples;
	// ͳ�����ݸ���
	int newNumSeg = numSeg;
	// ��ʼ���ָ��
	for (int i = 0; i <= numSeg; i++){
		Mat sampleImage;
		//cout << "ok" << endl;
		samples.push_back(sampleImage);
	}
	// ͳ��ÿһ������
	for (int i = 0; i < segments.rows; i++){
		for (int j = 0; j < segments.cols; j++){
			// ���ÿ�����صĹ���
			//cout << "hh" << endl;
			int index = segments.at<int>(i, j);
			//cout << "hh" << endl;
			if (index >= 0 && index < numSeg)
				samples[index].push_back(image(Rect(j, i, 1, 1)));
		}
	}

	// ����ֱ��ͼ
	vector<MatND> hist_bases;
	Mat hsv_base;
	// ����ֱ��ͼ����
	int h_bins = 35;
	int s_bins = 30;
	int histSize[] = { h_bins, s_bins };
	// hue�任��Χ0~256, saturation�任��Χ0~180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };
	const float* ranges[] = { h_ranges, s_ranges };
	// ʹ�õ�0���1ͨ��
	int channels[] = { 0, 1 };
	// ����ֱ��ͼ
	MatND hist_base;
	for (int c = 1; c < numSeg; c++){
		if (samples[c].dims>0){
			// �����򲿷�ת����HSV
			cvtColor(samples[c], hsv_base, CV_BGR2HSV);
			// ֱ��ͼͳ��
			calcHist(&hsv_base, 1, channels, Mat(),
				hist_base, 2, histSize, ranges, true, false);
			// ֱ��ͼ��һ��
			normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
			// ��ӵ�ͳ�Ƽ�
			hist_bases.push_back(hist_base);
		}
		else {
			hist_bases.push_back(MatND());
		}
		hist_base.release();
	}
	cout << "ֱ��ͼok" << endl;

	double similarity = 0;
	vector<bool> mearged;
	for (int k = 0; k < (int)hist_bases.size(); k++){
		mearged.push_back(false);
	}
	// ͳ��ÿһ�����ֵ�ֱ��ͼ����
	for (int c = 0; c < (int)hist_bases.size(); c++){
		for (int q = c + 1; q < (int)hist_bases.size(); q++){
			if (!mearged[q]){
				// �ж�ֱ��ͼ��ά��
				if (hist_bases[c].dims > 0 && hist_bases[q].dims > 0){
					// ֱ��ͼ�Ա�
					similarity = compareHist(hist_bases[c], hist_bases[q], CV_COMP_BHATTACHARYYA);

					if (similarity > 0.99){
						mearged[q] = true;
						if (q != c){
							// �������򲿷�
							newNumSeg--;
							for (int i = 0; i < segments.rows; i++){
								for (int j = 0; j < segments.cols; j++){
									int index = segments.at<int>(i, j);
									// �ϲ�
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

// ��ɫͨ������
static void MergeSeg(Mat& img, const Scalar& colorDiff){
	CV_Assert(!img.empty());
	Mat img_copy = img.clone();
	RNG rng = theRNG();
	// ��������ͼ��
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	Mat mask1(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			if (mask.at<uchar>(y + 1, x + 1) == 0){
				// ������ɫ
				Scalar newVal(rng(256), rng(256), rng(256));
				// ����ϲ�
				//if (floodFill(img_copy, mask1, Point(x, y), newVal, 0, colorDiff, colorDiff) > 120){
				//cout << "hello" << endl;
				floodFill(img, mask, Point(x, y), newVal, 0, colorDiff, colorDiff);
				//}
				//cout << "area = " << area << endl;
			}
		}
	}
}


// �������FloodFillͼ��ָ�
Mat getFloodFillImage(const Mat&srcImage, Mat mask, Point pt, int& area, int ffillMode, int loDiff, int upDiff,
	int connectivity, bool useMask, Scalar color, int newMaskVal){
	// floodfill��������
	Point seed = pt;
	int lo = ffillMode == 0 ? 0 : loDiff;
	int up = ffillMode == 0 ? 0 : upDiff;
	int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	Mat dst = convert2BGR(srcImage);
	Rect ccomp;

	// ���ݱ�־λѡ�񷺺����
	if (useMask){
		// ��ֵ������
		cv::threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
		area = floodFill(dst, mask, seed, color, &ccomp, Scalar(lo, lo, lo), Scalar(up, up, up), flags);
		//imshow("mask", mask);
		return mask;
	}
	else {
		// �������
		area = floodFill(dst, seed, color, &ccomp, Scalar(lo, lo, lo), Scalar(up, up, up), flags);
		//imshow("image", dst);
		return dst;
	}
}

// MeanShiftͼ��ָ�
Mat getMeanShiftImage(const Mat &srcImage, int spatialRad, int colorRad, int maxPyrLevel){
	CV_Assert(!srcImage.empty());
	Mat resImg;
	// ��ֵƯ�Ʒָ�
	pyrMeanShiftFiltering(srcImage, resImg, spatialRad, colorRad, maxPyrLevel);

	imshow("resImg", resImg);

	// ��ɫͨ������ϲ�
	MergeSeg(resImg, Scalar::all(2));
	return resImg;
}

// Grabcutͼ��ָ� ����ǰ��ͼ��
Mat getGrabcutImage(const Mat& srcImage, Rect roi){
	CV_Assert(!srcImage.empty());
	// ����ǰ�������ͼ��;
	Mat srcImage2 = srcImage.clone();
	cv::Mat foreground(srcImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat result(srcImage.size(), CV_8UC1);
	// Grabcut�ָ�ǰ���뱳��
	cv::Mat fgMat, bgMat;

	// ��������
	int i = 20;
	std::cout << "20 iters" << std::endl;
	// ʵ��ͼ�����
	grabCut(srcImage, result, roi, bgMat, fgMat, i, GC_INIT_WITH_RECT);

	// ͼ��ƥ��
	compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);

	// ����ǰ��ͼ��
	srcImage.copyTo(foreground, result);

	return foreground;
}

// �߶ȱ任ʵ��
bool CreateScaleSpace(cv::Mat srcImage, std::vector<std::vector<Mat>> &ScaleSpace, std::vector<std::vector<Mat>> &DoG){
	if (!srcImage.data){
		return false;
	}
	cv::Size ksize(5, 5);
	double sigma;
	Mat srcBlurMat, up, down;
	// ��˹ƽ��
	GaussianBlur(srcImage, srcBlurMat, ksize, 0.5);
	// ������
	pyrUp(srcBlurMat, up);
	up.copyTo(ScaleSpace[0][0]);
	// ��˹ƽ��
	GaussianBlur(ScaleSpace[0][0], ScaleSpace[0][0], ksize, 1.0);
	// ͼ�����
	for (int i = 0; i < 4; i++){
		// ƽ������
		double sigma = 1.1412135;
		for (int j = 0; j < 5 + 2; j++){
			sigma = sigma*pow(2.0, j / 2.0);
			// ����һ�߶Ƚ��и�˹����
			GaussianBlur(ScaleSpace[i][j], ScaleSpace[i][j + 1], ksize, sigma);
			// ���ɶ�߶ȿռ�
			DoG[i][j] = ScaleSpace[i][j] - ScaleSpace[i][j + 1];
			// �����Ӧ�����ռ�߶�
			cout << "iave:" << i << " Scale:" << j << "size:" <<
				ScaleSpace[i][j].rows << "x" <<
				ScaleSpace[i][j].cols << endl;
		}

		// ���������ɣ��������н���������
		if (i < 3){
			// �������²���
			pyrDown(ScaleSpace[i][0], down);
			down.copyTo(ScaleSpace[i + 1][0]);
		}
	}
	return true;
}


// ����ͼʵ�� HOG��������ʵ��
// �������ͼ
std::vector<cv::Mat> CalculateIntegralHOG(Mat &srcMat){
	// Sobel��Ե���
	Mat sobelMatX, sobelMatY;
	Sobel(srcMat, sobelMatX, CV_32F, 1, 0);
	Sobel(srcMat, sobelMatY, CV_32F, 0, 1);
	std::vector<Mat> bins(NBINS);

	for (int i = 0; i < NBINS; i++){
		bins[i] = Mat::zeros(srcMat.size(), CV_32F);
	}

	Mat magnMat, angleMat;
	// ����ת��
	cartToPolar(sobelMatX, sobelMatY, magnMat, angleMat, true);
	// �Ƕȱ任
	add(angleMat, Scalar(180), angleMat, angleMat < 0);
	add(angleMat, Scalar(-180), angleMat, angleMat >= 180);
	angleMat /= THETA;
	for (int y = 0; y < srcMat.rows; y++){
		for (int x = 0; x < srcMat.cols; x++){
			// ����bins�·�ֵ
			int ind = angleMat.at<float>(y, x);
			bins[ind].at<float>(y, x) += magnMat.at<float>(y, x);
		}
	}
	// ���ɻ���ͼͼ��
	std::vector<Mat> integrals(NBINS);
	for (int i = 0; i < NBINS; i++){
		integral(bins[i], integrals[i]);
	}
	return integrals;
}

// �����������ֱ��ͼʵ��
// ���㵥��cell HOG����
void calHOGinCell(Mat& HOGCellMat, Rect roi, std::vector<Mat>& integrals){
	// ʵ�ֿ��ٻ���HOG
	int x0 = roi.x, y0 = roi.y;
	int x1 = x0 + roi.width;
	int y1 = y0 + roi.height;
	for (int i = 0; i < NBINS; i++){
		// ���ݾ�������������������
		Mat integral = integrals[i];
		float a = integral.at<double>(y0, x0);
		float b = integral.at<double>(y1, x1);
		float c = integral.at<double>(y0, x1);
		float d = integral.at<double>(y1, x0);
		HOGCellMat.at<float>(0, i) = (a + b) - (c + d);
	}
}

// ��ȡHOGֱ��ͼ
cv::Mat getHog(Point pt, std::vector<Mat> &integrals){
	// �жϵ�ǰ���λ���Ƿ��������
	if (pt.x - R_HOG < 0 || pt.y - R_HOG < 0 || pt.x + R_HOG >= integrals[0].cols || pt.y + R_HOG >= integrals[0].rows){
		return Mat();
	}

	// ֱ��ͼ
	Mat hist(Size(NBINS*BLOCKSIZE*BLOCKSIZE, 1), CV_32F);
	Point t1(0, pt.y - R_HOG);
	int c = 0;
	// ������
	for (int i = 0; i < BLOCKSIZE; i++){
		t1.x = pt.x - R_HOG;
		for (int j = 0; j < BLOCKSIZE; j++){
			// ��ȡ��ǰ���ڣ�����ֲ�ֱ��ͼ
			Rect roi(t1, t1 + Point(CELLSIZE, CELLSIZE));
			// ���㵱ǰbins��ֱ��ͼ
			Mat hist_temp = hist.colRange(c, c + NBINS);
			calHOGinCell(hist_temp, roi, integrals);
			// cell �����ߴ�
			t1.x += CELLSIZE;
			c += NBINS;
		}
		t1.y = CELLSIZE;
	}

	// ��һ��L2����
	normalize(hist, hist, 1, 0, NORM_L2);
	return hist;
}

// ����HOG���� �˺����������� �����֤
std::vector<Mat> calHOGFeature(cv::Mat srcImage){
	Mat grayImage = GrayTrans(srcImage);
	std::vector<Mat> HOGMatVector;
	grayImage.convertTo(grayImage, CV_8UC1);

	// ���ɻ���ͼ
	std::vector<Mat> integrals = CalculateIntegralHOG(grayImage);
	Mat image = grayImage.clone();
	// �Ҷ�ֵ��С
	image *= 0.5;
	// HOG��������
	cv::Mat HOGBlockMat(Size(NBINS, 1), CV_32F);
	// cell����
	for (int y = CELLSIZE / 2; y < grayImage.rows; y += CELLSIZE){
		for (int x = CELLSIZE / 2; x < grayImage.cols; x += CELLSIZE){
			// ��ȡ��ǰ����HOG
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
			// L2������һ��
			normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
			HOGMatVector.push_back(HOGBlockMat);

			Point center(x, y);

			//if (y % 7 != 0 || x % 7 != 0)
			//	continue;

			// ����HOG����ͼ
			for (int i = 0; i < NBINS; i++){
				// �ǶȻ�ȡ
				double theta = (i*THETA + 90.0) * CV_PI / 180.0;
				Point rd(CELLSIZE * 0.5*cos(theta), CELLSIZE*0.5*sin(theta));

				// ��ȡ��������
				Point rp = center - rd;
				Point lp = center - (-rd);
				// ����HOG������
				line(image, rp, lp, Scalar(255 * HOGBlockMat.at<float>(0, i), 255, 255));
			}
		}
	}
	imshow("out", image);
	return HOGMatVector;
}

// ����LBP����
cv::Mat getLBPImage(cv::Mat & srcImage){
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	srcImage = GrayTrans(srcImage);
	cv::Mat resultMat(srcImage.size(), srcImage.type());
	// ����ͼ������LBP����
	for (int y = 1; y < nRows - 1; y++){
		for (int x = 1; x < nCols - 1; x++){
			// ��������
			uchar neighbor[8] = { 0 };
			neighbor[0] = srcImage.at<uchar>(y - 1, x - 1);
			neighbor[1] = srcImage.at<uchar>(y - 1, x);
			neighbor[2] = srcImage.at<uchar>(y - 1, x + 1);
			neighbor[3] = srcImage.at<uchar>(y, x + 1);
			neighbor[4] = srcImage.at<uchar>(y + 1, x + 1);
			neighbor[5] = srcImage.at<uchar>(y + 1, x);
			neighbor[6] = srcImage.at<uchar>(y + 1, x - 1);
			neighbor[7] = srcImage.at<uchar>(y, x - 1);

			// ��ǰͼ��Ĵ�������
			uchar center = srcImage.at<uchar>(y, x);
			uchar temp = 0;
			// ����LBP��ֵ
			for (int k = 0; k < 8; k++){
				// �������ĵ�����
				temp += (neighbor[k] >= center)*(1 << k);
			}
			resultMat.at<uchar>(y, x) = temp;
		}
	}
	return resultMat;
}

// Haar������ȡ ����Haar����
double HaarExtract(double const **image, int type_, cv::Rect roi){
	double value;
	double wh1, wh2;
	double bk1, bk2;
	int x = roi.x;
	int y = roi.y;
	int width = roi.width;
	int height = roi.height;
	switch (type_){
		// Haarˮƽ��Ե
	case 0:	// HaarHEdege
		wh1 = calcIntegral(image, x, y, width, height);
		bk1 = calcIntegral(image, x + width, y, width, height);
		value = (wh1 - bk1) / static_cast<double>(width * height);
		break;
		// Haar��ֱ��Ե
	case 1:
		wh1 = calcIntegral(image, x, y, width, height);
		bk1 = calcIntegral(image, x, y + height, width, height);
		value = (wh1 - bk1) / static_cast<double>(width * height);
		break;
		// Haarˮƽ����
	case 2:
		wh1 = calcIntegral(image, x, y, width * 3, height);
		bk1 = calcIntegral(image, x + width, y, width, height);
		value = (wh1 - 3.0*bk1) / static_cast<double>(2 * width * height);
		break;
		// Haar��ֱ����
	case 3:
		wh1 = calcIntegral(image, x, y, width, height * 3);
		bk1 = calcIntegral(image, x, y + height, width, height);
		value = (wh1 - 3.0*bk1) / static_cast<double>(2 * width*height);
		break;
		// Haar������
	case 4:
		wh1 = calcIntegral(image, x, y, width * 2, height * 2);
		bk1 = calcIntegral(image, x + width, y, width, height);
		bk2 = calcIntegral(image, x, y + height, width, height);
		value = (wh1 - 2.0*(bk1 + bk2)) / static_cast<double>(2 * width*height);
		break;
		// Haar���İ�Χ��
	case 5:
		wh1 = calcIntegral(image, x, y, width * 3, height * 3);
		bk1 = calcIntegral(image, x + width, y + height, width, height);
		value = (wh1 - 9.0*bk1) / static_cast<double>(8 * width*height);
		break;
	}
	return value;
}

// ���㵥���ڵĻ���ͼ
double calcIntegral(double const** image, int x, int y, int width, int height){
	double term_1 = image[y - 1 + height][x - 1 + width];
	double term_2 = image[y - 1][x - 1];
	double term_3 = image[y - 1 + height][x - 1];
	double term_4 = image[y - 1][x - 1 + width];
	return (term_1 + term_2) - (term_3 + term_4);
}

// ��ͼƬ��������帨������
void GetStringSize(HDC hDC, const char* str, int* w, int* h)
{
	SIZE size;
	GetTextExtentPoint32A(hDC, str, strlen(str), &size);
	if (w != 0) *w = size.cx;
	if (h != 0) *h = size.cy;
}

// ��ͼƬ��������� ����ת
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


// ��ͼƬ��������� ֧�ֺ��� �ɻ��� �ɵ����� ��б��
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

	lf.lfItalic = italic;  //б��  
	lf.lfUnderline = underline;   //�»���  
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

	//�������  
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

// ������������
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

// ��ͼ���ĳһ�л�������
void drawDottedLineRow(Mat& src, int row, Scalar color, int thickness){
	CV_Assert(row < src.rows);
	CV_Assert(row >= 0);
	dottedLine(src, Point(0, row), Point(src.cols - 1, row), color, thickness);
}

// ��ͼ���ĳһ�л�������
void drawDottedLineCol(Mat& src, int col, Scalar color, int thickness){
	CV_Assert(col < src.cols);
	CV_Assert(col >= 0);
	dottedLine(src, Point(col, 0), Point(col, src.rows - 1), color, thickness);
}

// ˫���Բ�ֵ
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

// ����Աȶ� ����Mat��¼����ĶԱȶ� double
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

// ���㱥�Ͷ� ����Mat��¼����ı��Ͷ� double
Mat calculateSaturate(Mat src, bool useD){
	CV_Assert(src.channels() == 3);

	Mat result(src.size(), CV_64FC1);

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			//���Ͷȼ���Ϊ��ɫͨ���ı�׼��
			double R = src.at<Vec3b>(i, j)[2];
			double G = src.at<Vec3b>(i, j)[1];
			double B = src.at<Vec3b>(i, j)[0];
			double mu = (R + G + B) / 3.0;

			result.at<double>(i, j) = sqrt(((R - mu)*(R - mu) + (G - mu)*(G - mu) + (B - mu)*(B - mu)) / 3.0);
		}
	}
	if (!useD)
		convertScaleAbs(result, result);  //ȡabs����ֵ
	return result;
}

// �����ع�ʱ������
Mat calculateWellExpose(Mat src)
{
	double sig = 0.2;
	int imgs_row = src.rows; //��
	int imgs_col = src.cols; //��

	Mat img_wellExposedness(imgs_row, imgs_col, CV_64F);
	Mat srcD;
	src.convertTo(srcD, CV_64FC3);
	double * p = srcD.ptr<double>(0);
	double * pd = img_wellExposedness.ptr<double>(0);

	for (int r = 0; r < imgs_row * imgs_col; r++){
		//��˹��������wellExposedness
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

// ��˹������
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

// ������˹������
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
		// �²�������
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

// ������˹�������ؽ�
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

// Ƶ���˲� ��ͨ�˲� �����е�value ���ڱ����������ֵ���  ����ֵ��ֱ����ʾ����CV_8UC1 
Mat getDFTBlur(Mat img, Mat& value){
	cv::Mat dftInput1, dftImage1, inverseDFT;
	img.convertTo(dftInput1, CV_64F);
	cv::dft(dftInput1, dftImage1, cv::DFT_COMPLEX_OUTPUT);	 // Applying DFT

	//cout << dftImage1 << endl;

	//����Ƶ���˲���
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

	//	//��˹��ͨ�˲��� ��˹��ͨ�˲�
	multiply(dftImage1, gaussianBlur, gaussianBlur);


	// Reconstructing original imae from the DFT coefficients
	cv::idft(gaussianBlur, value, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT


	value.convertTo(inverseDFT, CV_8U);
	//cout << inverseDFTconverted.channels() << endl;

	Mat result = getContrastStretchImage(value);
	return result;
}

// �����ơ��б�ͼ�� �����б�ͼ���С�͸��ӵ�������
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


// �����ơ��б�ͼ�� �����б�ͼ���С�͸�����
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

// �����ơ��б�ͼ�� �����б�ͼ���С�͸�����
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

// �ϲ�����ͼ Ĭ��Ϊ����ϲ� ����ƴ�� ���Һϲ� ����ƴ�� ���ºϲ�
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

// ��ͼ��ѹ�����γ̱��� ��ѹ����ֵͼ��ǳ�����
bool runLengthCoding(Mat img, string outputPath){
	Mat gray = GrayTrans(img);
	uchar *data = gray.ptr<uchar>(0);

	FILE*ofp;
	if ((ofp = fopen(outputPath.c_str(), "wb")) == NULL){
		cout << "�޷������ļ���" << endl;
		return false;
	}

	unsigned long inputSize, output = 0;
	inputSize = img.rows * img.cols;
	fwrite(&img.cols, sizeof(int), 1, ofp);
	fwrite(&img.rows, sizeof(int), 1, ofp);
	// �ж���λ��0����255
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

// ��ͼ���ѹ�����γ̱����ѹ��
Mat runLengthDecompress(string filepath)
{
	FILE*ifp;
	if ((ifp = fopen(filepath.c_str(), "rb")) == NULL){
		cout << "�޷����ļ���" << endl;
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

// ��ͼ��ѹ����JPEGͼ��ѹ�� ����ΪJPEG��ɫͼ��
Mat JPEGCompress(Mat src, int level){
	CV_Assert(level > 0 && level <= 100);
	cout << "origin image size: " << src.dataend - src.datastart << endl;
	cout << "height: " << src.rows << endl << "width: " << src.cols << endl << "depth: " << src.channels() << endl;
	cout << "height*width*depth: " << src.rows*src.cols*src.channels() << endl << endl;

	// (1)jpegѹ��
	vector<uchar> buff;		// buff for coding
	vector<int> param = vector<int>(2);
	param[0] = CV_IMWRITE_JPEG_QUALITY;
	param[1] = level; // default(95)(0-100)

	imencode(".jpg", src, buff, param);

	cout << "coded file size(jpg): " << buff.size() << endl;	// �Զ���ϴ�С

	Mat jpegimage = imdecode(Mat(buff), CV_LOAD_IMAGE_COLOR);

	double psnr = PSNR(src, jpegimage);
	double bpp = 8.0*buff.size() / (jpegimage.size().area());	// bit/pixel
	printf("quality:%03d, %.1fdB, %.2fbpp\n", level, psnr, bpp);

	return jpegimage;
}

// Ѱ��Mat����󼸸�Ԫ�� TopEles  Ѱ��ǰnum������ vec[0]��������꣬vec[1]����������,vec[3]����ֵ
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

// �������� type ��Ϊ4����8
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

// �������� 4����
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

// ��ͼƬ�ĵȷ�
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

// �Ĳ����ֽ�
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
			//printf("�ֽ�m[%d]\n", i);
			quadtreeSubdivision(input, draw, m[i], divTimes, minSize, thresh, Rect(rect.tl().x + (i % 2)*m[i].rows, rect.tl().y + (i / 2)*m[i].cols, rect.width / 2, rect.height / 2));
		}
	}
}

// �Ҷȼ���  �Ҷ����ֵ��ȥ�Ҷ���Сֵ
double getGrayRange(Mat input){
	CV_Assert(input.data != NULL);
	Mat img = GrayTrans(input);

	double minV, maxV;
	minMaxLoc(img, &minV, &maxV, NULL, NULL);
	return (maxV - minV);
}

// �Ҷȹ�������/�Ҷȹ��־���
// ˮƽ���� GLCM_HOR 0
// ��ֱ���� GLCM_VER 1
// ��б���� GLCM_TL 2
// ��б���� GLCM_TR 3
// �Ҷȹ������󡢻Ҷȹ��־��� Gray-level co-occurrence matrix
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

//��ͼ���һ��Ϊ0-255��������ʾ
// ���Զ�ͼ��������촦����ڽ��й�һ������ Ĭ�ϲ���������
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


// ����ĳ�еĻҶȱ仯���� ����ͼ
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
	imshow("�Ҷȱ仯����", result);
	waitKey(0);
}


// ����ĳ�еĻҶȱ仯���� ����ͼ
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
	imshow("�Ҷȱ仯����", result);
	waitKey(0);
}

// ͼ��ָ� ���Ĭ�ϰ�������
bool ImgSegm(Mat src, string outputPath, Size size, string prefix){
	if (size.width > src.cols || size.height > src.rows){
		cout << "�ָ�ͼ���СӦ������Դͼ���С!" << endl;
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


// �����޸��ļ��������к�׺Ϊsuffixͼ��ĳߴ� �����޸ĳߴ� 
bool resizeImgsInFolder(string folderPath, Size dstSize, string dstFolder, string suffix,
	string dstSuffix /*= ""*/, bool gray/* = false*/){	
	ofstream hello(folderPath);	// �˴�Ϊ���·����Ҳ���Ը�Ϊ����·�� 
	CFileFind finder;
	CString path = string2CString(folderPath+"\\*."+suffix);
	BOOL bContinue = finder.FindFile(path);
	
	string src_route_head = folderPath + "\\";  // Դͼ���·��ͷ
	string dst_route_head = dstFolder + "\\";	// Ŀ��ͼ���·��ͷ
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

			src = imread(SourceRoute);   // ����ͼ��
			if (gray)
				src = GrayTrans(src);
			resize(src, dst, dstSize);
			imwrite(dst_route_head + getFileName(name) + "."+filesuffix, dst);        // ����dst
		}
	}

	hello.close();
	return true;
}

//// �����޸��ļ��������к�׺Ϊsuffixͼ��ĳߴ� �����޸ĳߴ� 
//bool resizeImgsInFolder(CString folderPath, Size dstSize, CString dstFolder, CString suffix,
//	CString dstSuffix /*= ""*/, bool gray/* = false*/){
//	WIN32_FIND_DATA p;   // ָ��һ�����ڱ����ļ���Ϣ�Ľṹ��
//	HANDLE h = FindFirstFile(folderPath + _T("\\*.") + suffix, &p);		// FindFirstFile�ķ���ֵ��һ��������ڶ�������p�ǲ������õķ�ʽ��Ҳ����
//
//	if (h == NULL){
//		cout << "hello" << endl;
//		return false;
//	}
//	// ˵����仰ִ����Ϻ�p��ָ����ļ�*.jpg
//
//	// ����p�ĳ�Ա����ֻ���ļ����������ļ�·�������Ա������·��ͷ
//	string src_route_head = CString2string(folderPath + _T("\\"));  // Դͼ���·��ͷ
//	string dst_route_head = CString2string(dstFolder + _T("\\"));	// Ŀ��ͼ���·��ͷ
//	string SourceRoute = src_route_head + CString2string(p.cFileName);   // ������·��ͷ���ļ�����ȫ·��
//	string filename = CString2string(getFileName(p.cFileName));
//	string filesuffix = CString2string(getFileSuffix(p.cFileName));
//	if (dstSuffix != ""){
//		filesuffix = CString2string(dstSuffix);
//	}
//	string DestRoute = dst_route_head + filename + "." + filesuffix;
//	cout << "SourceRoute = " << SourceRoute << endl;
//
//	Mat src = imread(SourceRoute);// ����ͼ��
//	if (gray)
//		src = GrayTrans(src);
//	Mat dst = Mat::zeros(dstSize, src.type());   // ����һ��dstSize��С��Ŀ��ͼ��resize��Ľ�������������    
//
//	resize(src, dst, dstSize);
//	imwrite(DestRoute, dst);		// ����dst
//
//	// ��ĿǰΪֹ�����Ǿ��Ѿ�����˶�Ŀ���ļ����е�һ��ͼ���resize�����뱣�棬�������ø��ļ�������ͼ��Ҳ������
//
//	while (FindNextFile(h, &p))  // pָ�벻�Ϻ��ƣ�Ѱ����һ��������һ��*.jpg
//	{
//		SourceRoute = src_route_head + CString2string(p.cFileName);
//		src = imread(SourceRoute, 0);   // ����ͼ��
//
//		resize(src, dst, dstSize);
//
//		filename = CString2string(getFileName(p.cFileName));
//		dst_route_head + filename + "." + filesuffix;
//		imwrite(DestRoute, dst);        // ����dst
//	}
//
//	return true;
//}



// Floyd-Steinberg �����㷨
Mat floydSteinbergDithering(Mat input){
	if (!input.data){
		cout << "����ͼ�������ݣ�" << endl;
		return Mat();
	}
	Mat gray = GrayTrans(input);
	gray.convertTo(gray, CV_64FC1);

	// ������ʶ
	uchar A = 255;
	uchar B = 0;
	// �������ͼ��
	Mat C = Mat::zeros(gray.size(), CV_8UC1);

	// ����
	for (int i = 0; i < gray.rows; i++){
		for (int j = 0; j < gray.cols; j++){
			double error = 0;
			double I = gray.at<double>(i, j);
			if (I > 128){				// ������ֵ 
				C.at<uchar>(i, j) = A;
				error = I - 255.0;		// �������
			}
			else {
				C.at<uchar>(i, j) = B;
				error = I;				// ���
			}
			if (j < gray.cols - 1){
				double temp = gray.at<double>(i, j + 1);
				temp = temp + error*(7.0 / 16.0);			// ������ɢ����7/16
				gray.at<double>(i, j + 1) = temp;
			}
			if (i < gray.rows - 1){
				double temp = gray.at<double>(i + 1, j);
				temp = temp + error*(5.0 / 16.0);			// ������ɢ����5/16
				gray.at<double>(i + 1, j) = temp;
				if (j < gray.cols - 1){
					temp = gray.at<double>(i + 1, j + 1);
					temp = temp + error*(1.0 / 16.0);		// ��������ɢ����1/16
					gray.at<double>(i + 1, j + 1) = temp;
				}
				if (j>0){
					temp = gray.at<double>(i + 1, j - 1);
					temp = temp + error*(3.0 / 16.0);			// ��������ɢ����3/16
					gray.at<double>(i + 1, j - 1) = temp;
				}
			}
		}
	}
	return C;
}

// ���ݶ�ͼ�� ��������ΪCV_32FC1
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

// ��ʾ�ռ��������ֵ ���� 
Mat showPt3d(Point3d pt){
	Mat result(600, 900, CV_8UC1);
	drawString(result, "X : " + double2string(pt.x, 5), Point(0, 100), MC_WHITE, 100, false, false, "DS-Digital");
	drawString(result, "Y : " + double2string(pt.y, 5), Point(0, 300), MC_WHITE, 100, false, false, "DS-Digital");
	drawString(result, "Z : " + double2string(pt.z, 5), Point(0, 500), MC_WHITE, 100, false, false, "DS-Digital");
	return result;
}

// ���ɼ���ͼ�� LUT����������ΪCV_32SC1
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
	else if (type == ENCRYPT_ADD_RANDOM){ //���Ƚ��лҶȻ����� lutҲΪCV_64FC1
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

// ����ͼ�� LUT����������ΪCV_32SC1
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
	else if (type == ENCRYPT_ADD_RANDOM){	// ֻ�ܶԻҶ�CV_64FC1ͼ����н��� lutҲΪCV_64FC1
		Mat tmp = (input - 0.9*lut)*10.0;
		tmp.convertTo(result, CV_8UC1);
		return result;
	}
}

// ��ά��ɢС���任����ͨ������ͼ��
// ����ͼ��Ҫ������ǵ�ͨ������ͼ�񣬶�ͼ���СҲ��Ҫ��
// ��1��任��w, h������2�ı�����2��任��w, h������4�ı�����3��任��w, h������8�ı���......��
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
		// ���С���任
		for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2)
		{
			// ˮƽ�任
			for (y = 0; y < nHeight; y++)
			{
				// ��ż����
				memcpy(pRow, pData[y], sizeof(float)* nWidth);
				for (i = 0; i < nHalfW; i++)
				{
					x = i * 2;
					pData[y][i] = pRow[x];
					pData[y][nHalfW + i] = pRow[x + 1];
				}
				// ����С���任
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
				// Ƶ��ϵ��
				for (i = 0; i < nHalfW; i++)
				{
					pData[y][i] *= fRadius;
					pData[y][nHalfW + i] /= fRadius;
				}
			}
			// ��ֱ�任
			for (x = 0; x < nWidth; x++)
			{
				// ��ż����
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
				// ����С���任
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
				// Ƶ��ϵ��
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

// ��ά��ɢС���ָ�����ͨ������ͼ��
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
		// ���С���ָ�
		for (n = 0; n < nLayer; n++, nWidth *= 2, nHeight *= 2, nHalfW *= 2, nHalfH *= 2)
		{
			// ��ֱ�ָ�
			for (x = 0; x < nWidth; x++)
			{
				// Ƶ��ϵ��
				for (i = 0; i < nHalfH; i++)
				{
					pData[i][x] /= fRadius;
					pData[nHalfH + i][x] *= fRadius;
				}
				// ����С���ָ�
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
				// ��ż�ϲ�
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
			// ˮƽ�ָ�
			for (y = 0; y < nHeight; y++)
			{
				// Ƶ��ϵ��
				for (i = 0; i < nHalfW; i++)
				{
					pData[y][i] /= fRadius;
					pData[y][nHalfW + i] *= fRadius;
				}
				// ����С���ָ�
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
				// ��ż�ϲ�
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

// ������Ĥͼ�� �Ҷ���Ĥ [threshLow,threshHigh]  �м�ֵȡ255������Ϊ0
// inverse����ȡ�� ���м�ֵ���㣬����Ϊ255
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

// ͼ������ɷַ��� PCA�任
// number_principal_compent �����������ɷ��� Ĭ��Ϊ0 �������гɷַ���
Mat PCATrans(Mat src, int number_principal_compent/* = 0*/){
	src = GrayTrans(src);
	src.convertTo(src, CV_32FC1);
	PCA pca(src, Mat(), CV_PCA_DATA_AS_ROW, number_principal_compent);
	//cout << pca.mean<<endl; // ��ֵ
	//cout << pca.eigenvalues << endl;// ����ֵ
	Mat dst = pca.project(src);   // ӳ���¿ռ�
	dst = pca.backProject(dst);
	
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

// ����GIF����
void makeGIF(cv::Size size, string gifName, int fps/* = 15*/){
	// ���ļ��С�ѡ��һ���ļ���
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

// ����GIF����
// ��Ҫ��C:\\FFMEPG\\bin�ļ����е�exe�ļ�������
// fpsΪ֡��
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

// �����������¿� �� �������¿�
Mat fftshift(Mat input){
	//CV_Assert(input.rows % 2 == 0 && input.cols % 2 == 0);
	int cx = input.cols / 2;
	int cy = input.rows / 2;

	Mat dftResultImage = input.clone();

	Mat tmp;
	// Top-left����Ϊÿһ�����޴���ROI
	Mat q0(dftResultImage, Rect(0, 0, cx, cy));
	// Top-Right
	Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
	// Bottom-Left
	Mat q2(dftResultImage, Rect(0, cy, cx, cy));
	// Bottom����Right
	Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
	// �������� (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	// ��������(Top-Right with Bottom-Left)
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	return dftResultImage;
}


void initGDI(){
	
	
	
}

// ��������
Mat screenshot(){
	int nWidth = GetSystemMetrics(SM_CXSCREEN);//�õ���Ļ�ķֱ��ʵ�x    
	int nHeight = GetSystemMetrics(SM_CYSCREEN);//�õ���Ļ�ֱ��ʵ�y    
	LPVOID    screenCaptureData = new char[nWidth*nHeight * 4];
	memset(screenCaptureData, 0, nWidth);
	
	// Get desktop DC, create a compatible dc, create a comaptible bitmap and select into compatible dc.    
	HDC hDDC = GetDC(GetDesktopWindow());//�õ���Ļ��dc    
	HDC hCDC = CreateCompatibleDC(hDDC);//    
	HBITMAP hBitmap = CreateCompatibleBitmap(hDDC, nWidth, nHeight);//�õ�λͼ    
	SelectObject(hCDC, hBitmap); //�����ܵ���ôд��    

	BitBlt(hCDC, 0, 0, nWidth, nHeight, hDDC, 0, 0, SRCCOPY);

	GetBitmapBits(hBitmap, nWidth*nHeight * 4, screenCaptureData);//�õ�λͼ�����ݣ����浽screenCaptureData�����С�    
	Mat img = Mat::zeros(nHeight, nWidth, CV_8UC4);				 // ����һ��rgba��ʽ��Mat������Ϊ��
	memcpy(img.data, screenCaptureData, nWidth*nHeight * 4);//�����Ƚ��˷�ʱ�䣬��д�ķ��㡣����������*4�����������д����*3�����ǲ��Եġ�    
	return img;
}

// Ϊͼ�����͸����ͨ�� ����Ϊ��ͨ������ͨ��ͼ��
// alphaΪ͸���ȣ�1Ϊ��͸����0Ϊ͸��
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

//// ������ͼ����������ѵĳߴ� ��չͼ�� ����ͼ��߿�
//int nRows = getOptimalDFTSize(img.rows);
//int nCols = getOptimalDFTSize(img.cols);
//cv::Mat resultImage;
//// �ѻҶ�ͼ��������Ͻǣ����ұߺ��±���չͼ��
//// ����ӵ����س�ʼ��Ϊ0
////cv::Mat resultImage;
//copyMakeBorder(img, resultImage, 0, nRows - img.rows, 0, nCols - img.cols,
//	BORDER_CONSTANT, Scalar::all(0));