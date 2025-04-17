#pragma once
#define _AFXDLL
#include<math.h>
#include<stdio.h>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
//#include<opencv2\nonfree\features2d.hpp>
//#include<opencv2/features2d.hpp>
//#include<opencv2/xfeatures2d.hpp>
//#include<opencv2\legacy\legacy.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2\features2d\features2d.hpp>
#include <afx.h>  
#include <afxdlgs.h> 
#include"MyString.h"
#include<iostream>
#include<cderr.h>	// for definition of FNERR_BUFFERTOOSMALL
#include"MyPoint2D.h"
#include"color.h"
#include"MyFloat.h"
#include<io.h>
#include"MyFile.h"
//#include"MyRotate.h"

using namespace std;

#define FLIP_VERTICAL	0
#define FLIP_HORIZONTAL 1
#define FLIP_ALL		2

#define NOISE_SALT 0
#define NOISE_PEPPER 1
#define NOISE_GAUSSIAN 2

#define EDGE_SOBEL_VER 0
#define EDGE_SOBEL_HOR 1
#define EDGE_SOBEL_ALL 2
#define EDGE_SCHARR_VER 3
#define EDGE_SCHARR_HOR 4
#define EDGE_SCHARR_ALL 5

#define NBINS 4
#define BLOCKSIZE 8
#define CELLSIZE 2
#define THETA 45
#define R_HOG 8

// �Ҷȹ�������
#define GLCM_HOR 0
#define GLCM_VER 1
#define GLCM_TL 2
#define GLCM_TR 3

// ����ͼ��
#define ENCRYPT_ROWS 0
#define ENCRYPT_COLS 1
#define ENCRYPT_ROWS_AND_COLS 2
#define ENCRYPT_ALL_PIXELS 3
#define ENCRYPT_ADD_RANDOM 4
//#define ENCRYPT_GRAYS

struct loadImgParas{
	vector<string>::iterator it;
	vector<cv::Mat>::iterator iMat;
	vector<cv::Mat>::iterator iMatRGB;

	loadImgParas(vector<string>::iterator it_,vector<cv::Mat>::iterator iMat_,vector<cv::Mat>::iterator iMatRGB_) :
		it(it_), iMat(iMat_),iMatRGB(iMatRGB_){}
};

// �Ҷ�ͼ���Ϊ��ͨ��ͼ��������ͨ�����Ƶ�һͨ��
cv::Mat convert2BGR(cv::Mat gray);



class MyImage
{
public:
	MyImage();
	~MyImage();

	cv::Scalar my_color[3];

	bool IfLoadImg;                                      // �Ƿ�������ͼƬ
	bool IfLoadForGray;                                  // �Ƿ��ԻҶȷ�ʽ����ͼ��

	
	void colorReduce(cv::Mat &image,int div);                // ����ͼ������ɫ����Ŀ
	void sharpen(const cv::Mat &image,cv::Mat &result);          // ����������˹���Ӷ�ͼ�������
	void sharpen2D(const cv::Mat &image,cv::Mat &result);        // ͼ���� Ч��ͬ��

	cv::Mat LoadImage(void);                                 // ����һ��ͼƬ
	cv::Mat getInvert(const cv::Mat& image);                     // �õ�ͼ��ĸ�Ƭ
	cv::Mat applyLookUp(const cv::Mat& image,const cv::Mat& lookup); // ���ò��ұ�������ͼ��
    
    
	
	
	cv::Mat getRadialGradient(const cv::Mat& img, cv::Point center, double scale);// ͼ�����ݶȱ任
};


cv::Mat addNoises(const cv::Mat& image,int num,int type, double mu = 2, double sigma = 0.8);    // �������

// ���ɸ�˹����
double generateGaussianNoise(double mu, double sigma);

cv::Mat GrayTrans(const cv::Mat image);                     // תΪ�Ҷ�ͼ��

cv::Mat getFlipImage(const cv::Mat& image,int type);         // ͼ��ת

/** �Ľ�������ͼ��Ϊ��ֵͼ�� 0��255
* @brief ������ͼ�����ϸ��
* @param[in] srcΪ����ͼ��,��cvThreshold�����������8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param[out] dstΪ��srcϸ��������ͼ��,��ʽ��src��ʽ��ͬ������ǰ��Ҫ����ռ䣬Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param[in] maxIterations���Ƶ���������������������ƣ�Ĭ��Ϊ-1���������Ƶ���������ֱ��������ս��
*/
cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);

// ���غ��� f(x,y) = [f(x+1,y)-f(x-1,y)]/2 , x�������У�y��������
cv::Mat gradientX(cv::Mat src, uchar thresh = 0);

// �������� f(x,y) = [f(x,y+1)-f(x,y-1)]/2 , x�������У�y��������
cv::Mat gradientY(cv::Mat src, uchar thresh = 0);

// �����ݶȷ�ֵ ע�ⷵ�ص���CV_32FC1��ʽ �����ݶȷ�ֵ����thresh���ݶȷ�����beginAngl��endAngl֮��ġ��ݶȷ�ֵ��
cv::Mat gradientAmpl(cv::Mat src, float thresh = 0.0, float beginAngl = 0.0, float endAngl = 360.0);

// �����ݶȷ��� ע�ⷵ�ص���CV_8UC1��ʽ��ģ ������ýǶ���
cv::Mat gradientAngl(cv::Mat src, float beginAngl=0.0, float endAngl = 360.0);

// ����ͼ��
cv::Mat loadImage(std::string filepath, bool grayScale=false);

// ����ͼ�� ����ͬһ�ļ����� ����������ͬ��ͼƬ
bool loadImage(std::vector<cv::Mat>& images, std::string folderpath, std::string prefix, int num, std::string suffix="bmp", int initIndex=1);

// ����ͼ��
bool loadImages(vector<cv::Mat>& loadImages, vector<cv::Mat>& loadImagesRGB, vector<CString>& FPths);

// ����ͼ����һ��ͼ������ʾvector<Point2f>
void drawVecPoints(cv::Mat &mat, vector<cv::Point2f> points, cv::Scalar color = MC_YELLOW, int thickness = 2, bool useCross = false);


// ����ͼ����һ��ͼ������ʾvector<Point2f>
void drawVecPoints(cv::Mat &mat, vector<cv::Point> points, cv::Scalar color = MC_YELLOW, int thickness = 2, bool useCross = false);


// ����ͼ����һ��ͼ������ʾvector<Point2f>         rc ������Ǹõ����ڵİб������
cv::Mat drawVecPoints(cv::Mat mat, vector<cv::Point2f> points, vector<cv::Point> rc, int col);

//�����ơ���һ��ͼ������ʾvector<Point2f>
cv::Mat drawVecPoints(vector<cv::Point2f> points, cv::Size size, cv::Scalar color = MC_BLUE, int thickness = 1);

// ����ͼ����һ��ͼ���ϻ���vector<Point2f> �����̶���СΪSize(800*600) 
cv::Mat drawVecPoints(vector<cv::Point2f> points, cv::Scalar color, int thickness);

// ��һ��ͼ������ʾvector<Point2f>
cv::Mat drawVecPoints(vector<cv::Point2f> points, vector<cv::Point> rc, cv::Size size);

// ����ͼ����ͼ������ʾvector<Point2f> ˳�ν�����ֱ���������� ��ά������
void drawSeqPoints(cv::Mat& canvas, vector<cv::Point2f> points, cv::Scalar color = MC_WHITE, int lineWidth = 1);

//�����ơ� ��ͼ���ϻ���ʮ��
void drawCross(cv::Mat& src, cv::Point center, int len, cv::Scalar& color, int thickness = 1,
	int lineType = 8, int shift = 0);

// ������Բ x1�� y1 �������Ͻǵ�����  x2 ,y2�������½ǵ�����
void drawEllipse(cv::Mat& src, double x1, double y1, double x2, double y2, cv::Scalar color = MC_YELLOW, int thickness = 2);

// �����ơ� ��ͼ������A��B֮����Ƽ�ͷ
void drawArrow(cv::Mat& src, cv::Point2f A, cv::Point2f B, cv::Scalar& color, int thickness = 1,
	int lineType = 8, int shift = 0);

// �����ơ� ��ͼ���ϻ��Ƽ�ͷ����ͷ�����ĵ�����cen���ͷָ��angle
// ����涨ˮƽ����Ϊ0�㣬��ֱ����Ϊ90�㣬ˮƽ����Ϊ180�� angleΪ�Ƕ�ֵ
void drawArrow(cv::Mat& src, cv::Point cen, double angle, double len, cv::Scalar& color,
	int thickness = 1, int lineType = 8, int shift = 0);

// �����ơ���ͼ���ϻ���������
void triangle(cv::Mat& src, cv::Point A, cv::Point B, cv::Point C, cv::Scalar& color, int thickness = 1);

// �����ơ���ͼ���ϻ����������� rΪ�����εĳߴ� rΪ���������Բ�뾶
// thickness = -1 Ϊʵ��
void triangle(cv::Mat& src, cv::Point p, double r, cv::Scalar& color, int thickness = 1);

// �����ơ�������� ��ͼ���ϻ���������� ,rΪ������ε����Բ�뾶 
// angleOff Ϊƫ�ýǶ� ˮƽ���ҷ���Ϊ0�㣬��ʱ��ת��angleOffΪ��ֵ
void regularPolygon(cv::Mat& src, cv::Point p, int n, double r, cv::Scalar& color, double angleOff = 0, int thickness = 1);

// �����ơ������ ��ͼ���ϻ�������ǣ�rΪ����ǵ����Բ�ߴ�
void drawStar(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness = 1);

// �����ơ����� ��ͼ���ϻ��Ʒ��飬rΪ����ı߳�
void drawSquare(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness = 1);

// �����ơ��� ���Ʋ�� rΪĳ�ߵĳ���
void drawSkewCross(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness = 1);

// �����ơ���������ƫ�� ��ʮ�ֲ�˿����ʾ
cv::Mat drawDisparity(vector<cv::Point2f> pts1, vector<cv::Point2f> pts2, cv::Scalar color = MC_BLACK);

// �����ơ���������ƫ�� ��ʮ�ֲ�˿����ʾ
cv::Mat drawDisparity(vector<vector<cv::Point2f>> pts1, vector<vector<cv::Point2f>>pts2);

// �����ơ�����ͼ
cv::Mat drawHistogram(vector<double> a, cv::Scalar color);

// �������ؽǵ�
// qualityLevel ��������ܵĽǵ�������� Ĭ�� 0.01
vector<cv::Point2f> getGoodFeaturePoints(const cv::Mat& src, int maxCornerNum,
	double qualityLevel = 0.01, double minDist = 10, cv::Mat mask = cv::Mat(),int blockSize = 3, double k = 0.04 );

// ���жϡ��������Ƿ���ֵ��low �� high֮�� ��ȷ��srcΪ�Ҷ�ͼ��
bool hasNbhdPointInRange(cv::Mat& src, cv::Point center, int radius, int low, int high);

// ���жϡ����Ƿ������ģ
bool isInMask(cv::Mat mask, cv::Point point);

// ��ɸѡ�����Ϲ�صĶ�ά��
void siftGoodPoints(vector<cv::Point2f>& points, cv::Mat src);

// ͳ����ģ��ֵ���ص����
int countTrueNums(cv::Mat mask);

// ��ͼƬ�������ߵ���������
void setBorderZero(cv::Mat& mat);

// �������ص�ת��Ϊvector<Point>
vector<cv::Point> getTruePoints(cv::Mat mask);

// ����ͼ��Խ��߳ߴ�
float digLength(cv::Mat mat);

// ��Ѷ�ֵ����ֵѡȡ
int otsu(cv::Mat dst);

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
int otsu(unsigned char *image, int rows, int cols, int x0, int y0, int dx, int dy, int vvv = 0);

/* �޸�BMPͼ��ߴ�*/
void Resize(CBitmap* src, CBitmap *dst, cv::Size size);

// ����ͼ��
bool loadImage(cv::Mat& image, CString& filepath, bool grayScale = false);

// ���غ����ݶ�ͼ
cv::Mat gradX(cv::Mat src);

// ���������ݶ�ͼ
cv::Mat gradY(cv::Mat src);

// ���������ݶ�ͼ
cv::Mat gradXY(cv::Mat src);

// ͼ����ת angleΪ�Ƕ�ֵ ��ʱ��Ϊ��
cv::Mat getRotatedImg(const cv::Mat& img, double angle, double scale = 1.0);// ͼ����ת

std::vector<float> getRowDataUchar(const cv::Mat &img, int rowId);
std::vector<float> getColDataUchar(const cv::Mat &img, int colId);
std::vector<float> getRowDataFloat(const cv::Mat &img, int rowId);
std::vector<float> getColDataFloat(const cv::Mat &img, int colId);

// ��ת�任,ԭʼͼ���������� ��������Ϊ��ɫ ������С��Χ�ı���
// ˳ʱ��Ϊ��
cv::Mat angleRotate(cv::Mat& src, int angle);

// ƥ��
cv::Mat match2Img(cv::Mat A, cv::Mat B);



// ����H-Sֱ��ͼ
cv::Mat getH_SHistgram(cv::Mat src, cv::MatND &hist, int hueBinNum = 30);

// ����RGB��ɫֱ��ͼ ����ͼ��Ϊ��ɫͼ��
cv::Mat getRGBHistgram(cv::Mat src, cv::MatND &hist, int bins = 256);

// ����ֱ��ͼ ����ͼ��Ϊ��ɫͼ���Ҷ�ͼ�� ��Ϊ��ɫͼ���ת��Ϊ�Ҷ�ͼ��
cv::Mat getHistgram(cv::Mat src, cv::MatND &hist, int bins = 256);

// ��������ͼ��histֱ��ͼ�ȽϵĽ��
// ���ֱȽϷ���Ϊ ��CV_COMP_CORREL����CV_COMP_CHISQR����CV_COMP_INTERSECT����CV_COMP_BHATTACHARYYA�� 
double calCompareH_SHist(cv::Mat A, cv::Mat B, int method = CV_COMP_CORREL);

// ������ͶӰ����ȡͼ����Բο�ͼ��ķ���ͶӰͼ
cv::MatND getBackProjImage(cv::Mat src, cv::Mat ref, cv::Mat& hueRefHist, int bins = 30, bool equalHist = false);

// ģ��ƥ����� method��ƥ�䷽�� ��������
// ��CV_TM_SQDIFF��
// ��CV_TM_SQDIFF_NORMED��
// ��CV_TM_CCORR��
// ��CV_TM_CCORR_NORMED��
// ��CV_TM_CCOEFF��
// ��CV_TM_CCOEFF_NORMED��
cv::Mat getTemplateMatchImage(cv::Mat src, cv::Mat ref, cv::Point &p, int method = CV_TM_CCOEFF);

// ������ͼ�񡿱���BMPͼ��  ���浥ͨ����bmpͼ�� byte=1�Ǻ�ʹ�ģ������Ĳ���ʹ
bool saveBmp(CString bmpName, unsigned char* imgBuf, int width, int height, int byte=1);

// ������ | ��ȡ ͼ��BMPͼ������  ����Ϊ unsigned char*
int BmpSerialize(CString bmpName, unsigned short *imgBuf, bool bRead, int width, int height, int PicCount);		// int iType

// ������ͼ��BMPͼ������ ����Ϊ float*         coordinateleft1.bin
// cv::Mat����Ϊ�������ļ�
bool BmpSerialize(string fileName, cv::Mat data, bool bRead);

// ������ | ��ȡ ͼ��BMPͼ������  ����Ϊ float*
int BmpSerialize(CString bmpName, float *imgBuf, bool bRead, int width, int height, int PicCount);		// int iType

// ���cv::Mat�Ļ�����Ϣ
void printMatInfo(cv::Mat input);

// ƽ�Ʋ�����ͼ���С����
cv::Mat imageTranslation1(cv::Mat& srcImage, int xOffset, int yOffset);

// ƽ�Ʋ�����ͼ���С�ı�
cv::Mat imageTranslation2(cv::Mat &srcImage, int xOffset, int yOffset);

// ���ڵȼ����ȡͼ������
cv::Mat imageReduction1(cv::Mat &srcImage, float kx, float ky);

// ���������ӿ� ���ػҶ�ƽ��ֵ
cv::Vec3b areaAverage(const cv::Mat& srcImage, cv::Point_<int> leftPoint, cv::Point_<int> rightPoint);

// ���������ӿ���ȡͼ������
// �����ӿ���ȡͼ��������ͨ����Դͼ����������ӿ黮�֣�Ȼ����ȡ�ӿ�������ֵ��Ϊ���������Թ�����ͼ����ʵ�ֵġ�
cv::Mat imageReduction2(const cv::Mat& srcImage, double kx, double ky);

// ��÷���任ͼ��
cv::Mat getAffineTransformImage(cv::Mat srcImage, const cv::Point2f srcPts[], const cv::Point2f dstPts[]);

// ���б��ͼ��
// ��бΪ��  ��бΪ�� �Ƕ���
cv::Mat getSkewImage(cv::Mat srcImage, float angle);

// ��Ƶ��������
// ����PSNR��ֵ����ȣ�������ֵΪ30~50dB,ֵԽ��Խ��
double PSNR(const cv::Mat& I1, const cv::Mat& I2);

// ����MSSIM�ṹ�����ԣ�����ֵ��0��1��ֵԽ��Խ��
cv::Scalar MSSIM(const cv::Mat& i1, const cv::Mat& i2);

// MatIterator_ ��������ɫ����
cv::Mat inverseColor4(cv::Mat srcImage);

// isContinuous ��ɫ����
cv::Mat inverseColor5(cv::Mat srcImage);

// ����2-28 LUT ���ɫ����
cv::Mat inverseColor6(cv::Mat srcImage);

// ��������ʾ���ͼ��
void showManyImages(const std::vector<cv::Mat> &srcImages, cv::Size imgSize);

// ��ȡHSVͼ��
cv::Mat getHSVImage(const cv::Mat& image, cv::Mat& image_H, cv::Mat& image_S, cv::Mat& image_V);

// ����Ӧ��ֵ��
cv::Mat getAdaptiveThresholdImage(const cv::Mat& image, double maxValue=255, int blockSize = 5, double C = 10,
	int adaptiveMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C, int thresholdType = cv::THRESH_BINARY_INV);

// ˫��ֵ��
cv::Mat getDoubleThreshImage(const cv::Mat& image, double lowthresh, double highthresh, double maxValue = 255);

// ����ֵ��
cv::Mat getHalfThreshImage(const cv::Mat& image, double thresh);

// ֱ��ͼ���⻯
cv::Mat getEqualHistImage(const cv::Mat& image, bool useRGB = false);

// ֱ��ͼ�任��������
// ֱ��ͼ�任���ҷ���ʵ�ֵ�˼·:
// (1) ��Դͼ��ת��Ϊ�Ҷ�ͼ������ͼ��ĻҶ�ֱ��ͼ
// (2) ����Ԥ����ֵ�����ɵ͵��߲���iLow,���ɸߵ��Ͳ���iHigh
// (3) �����ϲ��õ�ֱ��ͼiLow��iHigh�����в��ұ�任
// (4) ͨ�����ұ����ӳ��任�����ֱ��ͼ���ҷ����任��
cv::Mat getHistogramTransLUT(const cv::Mat& srcImage, int segThreshold = 50);

// ֱ��ͼ�任�����ۼ�
// ֱ��ͼ�任�ۼƷ���ʵ�ֵ�˼·��
// ��1�� ��Դͼ��ת��Ϊ�Ҷ�ͼ������ͼ��ĻҶ�ֱ��ͼ
// ��2�� ����ӳ�����ֱ��ͼ���������ۻ�
// ��3�� ����ӳ������Ԫ��ӳ��õ����յ�ֱ��ͼ�任
cv::Mat getHistogramTransAggregate(const cv::Mat& srcImage);

// ֱ��ͼƥ��
// (1) �ֱ����Դͼ����Ŀ��ͼ����ۼƸ��ʷֲ�
// (2) �ֱ��Դͼ����Ŀ��ͼ�����ֱ��ͼ���⻯����
// (3) ������ӳ���ϵʹԴͼ��ֱ��ͼ���չ涨���б任
cv::Mat getHistgramMatchImage(const cv::Mat& srcImage, cv::Mat target);

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
cv::Mat getDistTransImage(cv::Mat& srcImage, int thresh = 100);

// ����opencv�Դ��ľ���任����
cv::Mat getDistTransImage2(cv::Mat &srcImage, int thresh = 160);

// GammaУ�� ٤��У��
// һ������£���Gamma������ֵ����1ʱ��ͼ��ĸ߹ⲿ�ֱ�ѹ�����������ֱ���չ
// ��Gamma����ֵС��1ʱ��ͼ��ĸ߹ⲿ�ֱ���չ���������ֱ�ѹ����
cv::Mat getGammaTransformImage(cv::Mat& srcImage, float kFactor);

// ͼ�����Ա任����
cv::Mat getLinearTransformImage(cv::Mat& srcImage, float a, int b);

// ͼ������任����1
// ͼ������任�ǽ�ͼ�������з�Χ��խ�ĵͻҶ�ֵӳ�������нϿ�Χ�ĻҶ�ֵ��
// ��������չͼ���б�ѹ���ģ��Ҷ�ֵ�ϸ�����ģ�������ֵ��
cv::Mat getLogTransform1(cv::Mat srcImage, float c);

// ͼ������任����2
// ͼ������任�ǽ�ͼ�������з�Χ��խ�ĵͻҶ�ֵӳ�������нϿ�Χ�ĻҶ�ֵ��
// ��������չͼ���б�ѹ���ģ��Ҷ�ֵ�ϸ�����ģ�������ֵ��
cv::Mat getLogTransform2(cv::Mat srcImage, float c);

// ͼ������任����3
// ͼ������任�ǽ�ͼ�������з�Χ��խ�ĵͻҶ�ֵӳ�������нϿ�Χ�ĻҶ�ֵ��
// ��������չͼ���б�ѹ���ģ��Ҷ�ֵ�ϸ�����ģ�������ֵ��
cv::Mat getLogTransform3(cv::Mat srcImage, float c);

// �Աȶ��������
cv::Mat getContrastStretchImage(cv::Mat srcImage);

// �Ҷȼ��ֲ�
// ������ȡ�ĸ���Ȥ����ĻҶ�ֵӳ������С������������Ȥ�ĻҶ�ֵ����ԭ��ֵ���䣬�������ͼ����Ϊ�Ҷ�ͼ��
cv::Mat getGrayLayeredImage(cv::Mat srcImage, int controlMin, int controlMax);

// ��ûҶȱ���ƽ������
std::vector<cv::Mat> getMBitPlans(cv::Mat srcImage);

// �������ֵ�ָ�
float calculateCurrentEntropy(cv::Mat hist, int threshold);

// Ѱ���������ֵ���ָ�
cv::Mat maxEntropySegMentation(cv::Mat inputImage);

// ����ͼ�񲨷��
// ͶӰ���ߵĲ���/������ͨ���ж���һ�׵���Ϊ��㣬���׵���Ϊ����ֵ��ȷ���ģ�������һ�ײ��D��
// ���ǹ�ע����ͼ���ֵ�ֵ�Ĵ�С�����������Ҫ������з��Ż���Ȼ����ͨ��������ײ�ֵı仯��
// �ҵ�����б�ʵ�U�����������������������ɸ�������,�㼯U����ͶӰ���ߵĲ��岨��ֵ.
// ���ص�ͼ��Ϊ�ҵ���ͼ�񲨷�ͼ�� resultVec��¼�����еĲ�����������
cv::Mat findPeak(cv::Mat srcImage, vector<int>& resultVec, int thresh);
 
// ��ô�ֱͶӰͼ�� ������а׵���� reduceMat �洢���Ǽ����� ��CV_32F��ʽ��
cv::Mat getVerticalProjImage(cv::Mat srcImage, cv::Mat & reduceMat);


// ���ˮƽͶӰͼ�� ������а׵���� reduceMat �洢���Ǽ����� ��CV_32F��ʽ��
cv::Mat getHorizontalProjImage(cv::Mat srcImage, cv::Mat & reduceMat);

// ͼ�����
// �ϲ�ͼ�����²��ͨ�˲���ͨ���²����õ��ģ��������ԭ���Ĳ�ֵ��Ӧ���Ǹ�˹���������������Ϣ��
void Pyramid(cv::Mat srcImage);

// ��������������������
cv::Mat Myfilter2D(cv::Mat srcImage);

// opencv�Դ����������
cv::Mat filter2D_(cv::Mat srcImage);

// ͼ����Ҷ�任
cv::Mat DFT(cv::Mat srcImage);

// ͼ����ɢ���ұ任
cv::Mat DCT(cv::Mat srcImage);

// ͼ��������
cv::Mat convolution(cv::Mat srcImage, cv::Mat kernel);

// ��ֵ�˲�
cv::Mat getBlurImage(const cv::Mat& src, cv::Size ksize = cv::Size(3, 3));

// ��ֵ�˲�
cv::Mat getMedianBlurImage(const cv::Mat& src, cv::Size ksize = cv::Size(3, 3));

// ��˹�˲�
cv::Mat getGaussianBlurImage(const cv::Mat& src, cv::Size ksize = cv::Size(3, 3), double sigmaX = 1.5, double sigmaY = 1.5);

// ˫���˲�
cv::Mat getBilateralFilterImage(const cv::Mat& src, int d = 25, double sigmaColor = 50.0, double sigmaSpace = 12.5);

// ͼ�����˲�
cv::Mat guidefilter(cv::Mat &srcImage,int r, double eps);

// ��ֱ�Ե���ʵ��
void diffOperation(const cv::Mat srcImage, cv::Mat& edgeXImage, cv::Mat& edgeYImage);

// ͼ��Ǽ���ֵ����Sobel��Եʵ��
cv::Mat getSobelVerEdge(cv::Mat srcImage);

// ͼ��ֱ�Ӿ��Sobel��Եʵ�� ģ����ֵΪ�ݶȷ�ֵ����ֵ
cv::Mat getsobelEdge(const cv::Mat& srcImage, uchar threshold);

// ͼ�����·Ǽ���ֵ����Sobelʵ�� flag ��ȡ�����ֵ
//#define EDGE_SOBEL_VER 0
//#define EDGE_SOBEL_HOR 1
//#define EDGE_SOBEL_ALL 2
cv::Mat getsobelOptaEdge(const cv::Mat& srcImage, int flag = EDGE_SOBEL_ALL);

// OpenCV�Դ���ͼ���Ե���� 
// flag��ȡ�����ֵ
//#define EDGE_SOBEL_VER 0
//#define EDGE_SOBEL_HOR 1
//#define EDGE_SOBEL_ALL 2
//#define EDGE_SCHARR_VER 3
//#define EDGE_SCHARR_HOR 4
//#define EDGE_SCHARR_ALL 5
cv::Mat getSobelEdgeImage(const cv::Mat srcImage, int flag);

// ��ȡLaplace��Ե
cv::Mat getLaplaceEdge(cv::Mat srcImage);

// Robert��Ե���
// Robert���������þֲ����Ѱ�ұ�Ե��һ�����ӣ�����򵥵ı�Ե������ӡ�Roberts�������öԽ���
// ��������������֮������ݶȷ�ֵ������Ե����ⴹֱ��Ե��Ч��Ҫ�������������Ե����λ���ȸߣ�
// ��������������������������Ե������Ӽ��ÿ�����ص����򲢶ԻҶȱ仯������������ͬʱҲ����
// �����ȷ����
cv::Mat getRobertsEdge(cv::Mat srcImage);

// Prewitt��Ե���
// Prewitt������һ�ױ�Ե������ӣ������Ӷ��������������á�Prewitt���ӶԱ�Ե�Ķ�λ���Ȳ���Roberts���ӣ�
// Sobel���ӶԱ�Ե����׼ȷ�Ը�����Prewitt���ӡ�
cv::Mat getPrewittEdge(cv::Mat srcImage, bool verFlag = false);

// Canny�⺯��ʵ�� �Ƽ��ĸ������ֵ��ֵ��2:1��3:1֮��
cv::Mat getCannyEdge(cv::Mat srcImage, int lowThresh, int highThresh);

// �Ľ���Ե�������Marr-Hildreth
// ���Ѹ�˹ƽ���˲�����������˹���˲��������������ƽ�����������ٽ��б�Ե���
// ���� getMarrEdge(srcImage, 9, 1.6);
cv::Mat getMarrEdge(const cv::Mat src, int kerValue, double delta);

// MoravecCorners�ǵ���
cv::Mat MoravecCorners(cv::Mat srcImage, vector<cv::Point> & points, int kSize = 5, int threshold = 10000);


// ����Harris�ǵ� thresh ��ʾ�ǵ�����ǿ��
cv::Mat getHarrisCornersImage(const cv::Mat& srcImage, float thresh = 170, int blockSize = 2, int kSize = 3, double k = 0.04);

// �õ��Զ����
// ����һ int   kshape	: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
// ������ int   ksize	: ��ʾ�ں˵ĳߴ�
// ������ Point kpos		: ��ʾê���λ��
cv::Mat getCustomKernel(int ksize, int kshape, cv::Point kpos);

// ��ʴ
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getErodeImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// ����
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getDilateImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_ELLIPSE, cv::Point kpos = cv::Point(-1, -1));

// ������
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getOpeningOperationImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// ������
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getClosingOperationImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// ��̬ѧ�ݶ�
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getMorphGradientImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// ��ñ���� Top Hat �ֳơ���ñ������
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getMorphTopHatImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// ��ñ���� Black Hat 
// kshape: ��ʾ�ں˵���״��������ѡ��
//							<1> ���Σ�		MORPH_RECT;
//							<2> ������:		MORPH_CROSS;
//							<3> Բ��:		MORPH_ELLIPSE
cv::Mat getMorphBlackHatImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// ��ˮ��ͼ��ָ�
cv::Mat getWatershedSegmentImage(cv::Mat &srcImage, int& noOfSegments, cv::Mat& markers);

// ��ʾ��ˮ��ָ��㷨���ͼ��
cv::Mat showWaterSegResult(cv::Mat markers);

// �ָ�ϲ�
void segMerge(cv::Mat& image, cv::Mat& segments, int & numSeg);

// ��ɫͨ������
static void MergeSeg(cv::Mat& img, const cv::Scalar& colorDiff = cv::Scalar::all(1));

// �������FloodFillͼ��ָ�
cv::Mat getFloodFillImage(const cv::Mat&srcImage, cv::Mat mask, cv::Point pt, int& area, int ffillMode = 1, int loDiff = 20, int upDiff = 20,
	int connectivity = 4, bool useMask = false, cv::Scalar color = MC_GREY, int newMaskVal = 255);

// MeanShiftͼ��ָ�
cv::Mat getMeanShiftImage(const cv::Mat &srcImage, int spatialRad = 20, int colorRad = 20, int maxPyrLevel = 6);

// Grabcutͼ��ָ�
cv::Mat getGrabcutImage(const cv::Mat& srcImage, cv::Rect roi);

// �߶ȱ任ʵ��
bool CreateScaleSpace(cv::Mat srcImage, std::vector<std::vector<cv::Mat>> &ScaleSpace, std::vector<std::vector<cv::Mat>> &DoG);

// ����ͼʵ�� HOG��������ʵ��
// �������ͼ
std::vector<cv::Mat> CalculateIntegralHOG(cv::Mat &srcMat);

// �����������ֱ��ͼʵ��
// ���㵥��cell HOG����
void calHOGinCell(cv::Mat& HOGCellMat, cv::Rect roi, std::vector<cv::Mat>& integrals);

// ��ȡHOGֱ��ͼ
cv::Mat getHog(cv::Point pt, std::vector<cv::Mat> &integrals);


// ����HOG����
std::vector<cv::Mat> calHOGFeature(cv::Mat srcImage);

// ����LBP����
cv::Mat getLBPImage(cv::Mat & srcImage);

// Haar������ȡ ����Haar����
double HaarExtract(double const **image, int type_, cv::Rect roi);

// ���㵥���ڵĻ���ͼ
double calcIntegral(double const** image, int x, int y, int width, int height);

// ��ͼƬ��������帨������
void GetStringSize(HDC hDC, const char* str, int* w, int* h);

// ��ͼƬ��������� ����ת
void drawString(cv::Mat& dst, string text, cv::Point org, double angle, cv::Scalar color, int fontSize,
	bool italic = false, bool underline = false, bool black = false, const char* fontType = "����");

// ��ͼƬ��������� ֧�ֺ��� �ɻ��� �ɵ����� ��б�� �ɼӴ�
// ��Ӻ��� ������� 
void drawString(cv::Mat& dst, string text, cv::Point org, cv::Scalar color, int fontSize,
	bool italic = false, bool underline = false, bool black = false, const char* fontType = "����");

// ������������
void dottedLine(cv::Mat& src, cv::Point A, cv::Point B, cv::Scalar color, int thickness = 1);

// ��ͼ���ĳһ�л�������
void drawDottedLineRow(cv::Mat& src, int row, cv::Scalar color, int thickness);

// ��ͼ���ĳһ�л�������
void drawDottedLineCol(cv::Mat& src, int col, cv::Scalar color, int thickness);

// ˫���Բ�ֵ
cv::Vec3d bilinearlInterpolation(cv::Mat src, cv::Point2f pt);

// �����ع�ʱ������
cv::Mat calculateWellExpose(cv::Mat src);

// ���㱥�Ͷ�	  ����cv::Mat��¼����ı��Ͷ� double
cv::Mat calculateSaturate(cv::Mat src, bool useD = true);

// ����Աȶ� ����cv::Mat��¼����ĶԱȶ� double
cv::Mat calculateContrast(cv::Mat src, bool useD = false);

// ������˹������
vector<cv::Mat> laplacian_pyramid(cv::Mat src, int nlev=-1);

// ������˹�������ؽ�
cv::Mat reconstruct_laplacian_pyramid(vector<cv::Mat> pyr);

// ��˹������
vector<cv::Mat> gaussian_pyramid(cv::Mat src, int nlev=-1);

// Ƶ���˲� ��ͨ�˲�  ����ֵvalue
cv::Mat getDFTBlur(cv::Mat img, cv::Mat& value);

// �����ơ��б�ͼ��
cv::Mat drawChessboardImage(cv::Size size, int gridWidth, int gridHeight = 0, int offsetX = 0, int offsetY = 0, bool LeftTopBlack = true);

// �����ơ��б�ͼ��
cv::Mat drawChessboardImage(cv::Size size, cv::Size gridCount, int offsetX = 0, int offsetY = 0, bool LeftTopBlack = true);

// �����ơ��б�ͼ�� �����������͸��ӵ����سߴ� �Լ����Ͻ��Ƿ�Ϊ��
cv::Mat drawChessboardImage(cv::Size gridCount, int gridWidth, int gridHeight, bool LeftTopBlack);

// �ϲ�����ͼ Ĭ��Ϊ����ϲ� ����ƴ�� ���Һϲ� ����ƴ�� ���ºϲ�
cv::Mat combine2Img(cv::Mat A, cv::Mat B, bool CmbHor = true);

// ��ͼ��ѹ�����γ̱��� ��ѹ����ֵͼ��ǳ�����
bool runLengthCoding(cv::Mat img, string outputPath);

// �γ̱����ѹ��
cv::Mat runLengthDecompress(string filepath);

// JPEGͼ��ѹ�� ����Ϊ��ɫJPEGͼ��
cv::Mat JPEGCompress(cv::Mat src, int level = 95);

// Ѱ��cv::Mat����󼸸�Ԫ�� TopEles  Ѱ��ǰnum������ vec[0]��������꣬vec[1]����������,vec[3]����ֵ
vector<cv::Vec3d> findTopElements(cv::Mat input, int num, bool useFilter = false);

// �������� 4����
cv::Mat RegionGrow(cv::Mat& src, cv::Point seed, int type = 4);

// �������� 4����
int grow(cv::Mat src, cv::Mat& mask, cv::Point p, vector<cv::Point>& s, int type = 4);

// ��ͼƬ�ĵȷ�
vector<cv::Mat> divideMatInto4(cv::Mat input, cv::Mat& draw, cv::Rect rect);

// �Ĳ����ֽ�
void quadtreeSubdivision(cv::Mat input, cv::Mat& draw, cv::Mat cur, int divTimes = 0, int minSize = 5, double thresh = 10.0, cv::Rect rect = cv::Rect(0, 0, 0, 0));

// �Ҷȼ���  �Ҷ����ֵ��ȥ�Ҷ���Сֵ
double getGrayRange(cv::Mat input);

// �Ҷȹ�������/�Ҷȹ��־���
// ˮƽ���� GLCM_HOR 0
// ��ֱ���� GLCM_VER 1
// ��б���� GLCM_TL 2
// ��б���� GLCM_TR 3
// �Ҷȹ������󡢻Ҷȹ��־��� Gray-level co-occurrence matrix
cv::Mat getGLCM(cv::Mat input, int type = GLCM_HOR, int grayLevel = 256);

//��ͼ���һ��Ϊ0-255��������ʾ
// ���Զ�ͼ��������촦����ڽ��й�һ������ Ĭ�ϲ���������
cv::Mat norm_0_255(const cv::Mat& src, cv::Size s = cv::Size(0, 0));


// ����ĳ�еĻҶȱ仯����
void drawGrayCurveInRow(cv::Mat src, int rowIndex);


// ����ĳ�еĻҶȱ仯����
void drawGrayCurveInCol(cv::Mat src, int rowIndex);

// ͼ��ָ�  ���Ĭ�ϰ�������
bool ImgSegm(cv::Mat src, string outputPath, cv::Size size, string prefix = "");

// �����޸��ļ��������к�׺Ϊsuffixͼ��ĳߴ� �����޸ĳߴ� 
//bool resizeImgsInFolder(CString folderPath, cv::Size dstSize, CString dstFolder, CString suffix = "jpg", CString dstSuffix = "", bool gray = false);

// �����޸��ļ��������к�׺Ϊsuffixͼ��ĳߴ� �����޸ĳߴ� 
bool resizeImgsInFolder(string folderPath, cv::Size dstSize, string dstFolder, string suffix,
	string dstSuffix = "", bool gray = false);

// Floyd-Steinberg �����㷨
cv::Mat floydSteinbergDithering(cv::Mat input);

// ���ݶ�ͼ�� ��������ΪCV_32FC1
bool Gradient(cv::Mat src, cv::Mat& dst);

// ��ʾ�ռ��������ֵ ���� 
cv::Mat showPt3d(cv::Point3d pt);

// ���ɼ���ͼ�� LUT����������ΪCV_32SC1
cv::Mat getEncryptImage(cv::Mat input, cv::Mat &lut, int type = ENCRYPT_ALL_PIXELS);

// ����ͼ�� LUT����������ΪCV_32SC1
cv::Mat getDecodeImage(cv::Mat input, cv::Mat lut, int type = ENCRYPT_ALL_PIXELS);

// ��ά��ɢС���任����ͨ������ͼ��
// ����ͼ��Ҫ������ǵ�ͨ������ͼ�񣬶�ͼ���СҲ��Ҫ��
// ��1��任��w, h������2�ı�����2��任��w, h������4�ı�����3��任��w, h������8�ı���......��
cv::Mat DWT(cv::Mat input, int nLayer);

// ��ά��ɢС���任����ͨ������ͼ��
cv::Mat IDWT(cv::Mat src, int nLayer);

// ������Ĥͼ�� �Ҷ���Ĥ [threshLow,threshHigh]  �м�ֵȡ255������Ϊ0
// inverse����ȡ�� ���м�ֵ���㣬����Ϊ255
cv::Mat generateGrayMask(cv::Mat input, uchar threshLow, uchar threshHight, bool inverse = false);

// ͼ������ɷַ��� PCA�任
// number_principal_compent �����������ɷ��� Ĭ��Ϊ0 �������гɷַ���
cv::Mat PCATrans(cv::Mat src, int number_principal_compent = 0);

// ����GIF����
void makeGIF(cv::Size size, string gifName = "", int fps = 15);

// ����GIF���� 
// ��Ҫ��C:\\FFMEPG\\bin�ļ����е�exe�ļ�������
// fpsΪ֡��
void makeGIF(string folderPath, string prefix, string savePath, cv::Size size = cv::Size(0, 0), int fps = 15);

// �����������¿� �� �������¿�
cv::Mat fftshift(cv::Mat input);

// ��������
cv::Mat screenshot();

// Ϊͼ�����͸����ͨ�� ����Ϊ��ͨ������ͨ��ͼ��
// alphaΪ͸���ȣ�1Ϊ��͸����0Ϊ͸��
cv::Mat addAlphaChannel(cv::Mat input, double alpha);

// ȫ����ʾ
//namedWindow("s", CV_WINDOW_NORMAL);
//setWindowProperty("s", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);