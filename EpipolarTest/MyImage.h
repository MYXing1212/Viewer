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

// 灰度共生矩阵
#define GLCM_HOR 0
#define GLCM_VER 1
#define GLCM_TL 2
#define GLCM_TR 3

// 加密图像
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

// 灰度图像变为三通道图像，另外两通道复制第一通道
cv::Mat convert2BGR(cv::Mat gray);



class MyImage
{
public:
	MyImage();
	~MyImage();

	cv::Scalar my_color[3];

	bool IfLoadImg;                                      // 是否已载入图片
	bool IfLoadForGray;                                  // 是否以灰度方式载入图像

	
	void colorReduce(cv::Mat &image,int div);                // 减少图像中颜色的数目
	void sharpen(const cv::Mat &image,cv::Mat &result);          // 基于拉普拉斯算子对图像进行锐化
	void sharpen2D(const cv::Mat &image,cv::Mat &result);        // 图像锐化 效果同上

	cv::Mat LoadImage(void);                                 // 载入一张图片
	cv::Mat getInvert(const cv::Mat& image);                     // 得到图像的负片
	cv::Mat applyLookUp(const cv::Mat& image,const cv::Mat& lookup); // 利用查找表生成新图像
    
    
	
	
	cv::Mat getRadialGradient(const cv::Mat& img, cv::Point center, double scale);// 图像径向梯度变换
};


cv::Mat addNoises(const cv::Mat& image,int num,int type, double mu = 2, double sigma = 0.8);    // 添加噪声

// 生成高斯噪声
double generateGaussianNoise(double mu, double sigma);

cv::Mat GrayTrans(const cv::Mat image);                     // 转为灰度图像

cv::Mat getFlipImage(const cv::Mat& image,int type);         // 图像翻转

/** 改进后，输入图像为二值图像 0或255
* @brief 对输入图像进行细化
* @param[in] src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param[out] dst为对src细化后的输出图像,格式与src格式相同，调用前需要分配空间，元素中只有0与1,1代表有元素，0代表为空白
* @param[in] maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
*/
cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);

// 返回横向 f(x,y) = [f(x+1,y)-f(x-1,y)]/2 , x是所在列，y是所在行
cv::Mat gradientX(cv::Mat src, uchar thresh = 0);

// 返回纵向 f(x,y) = [f(x,y+1)-f(x,y-1)]/2 , x是所在列，y是所在行
cv::Mat gradientY(cv::Mat src, uchar thresh = 0);

// 返回梯度幅值 注意返回的是CV_32FC1格式 返回梯度幅值大于thresh且梯度方向在beginAngl与endAngl之间的【梯度幅值】
cv::Mat gradientAmpl(cv::Mat src, float thresh = 0.0, float beginAngl = 0.0, float endAngl = 360.0);

// 返回梯度方向 注意返回的是CV_8UC1格式掩模 输入采用角度制
cv::Mat gradientAngl(cv::Mat src, float beginAngl=0.0, float endAngl = 360.0);

// 载入图像
cv::Mat loadImage(std::string filepath, bool grayScale=false);

// 载入图像 载入同一文件夹下 命名规则相同的图片
bool loadImage(std::vector<cv::Mat>& images, std::string folderpath, std::string prefix, int num, std::string suffix="bmp", int initIndex=1);

// 载入图像
bool loadImages(vector<cv::Mat>& loadImages, vector<cv::Mat>& loadImagesRGB, vector<CString>& FPths);

// 【绘图】在一张图像上显示vector<Point2f>
void drawVecPoints(cv::Mat &mat, vector<cv::Point2f> points, cv::Scalar color = MC_YELLOW, int thickness = 2, bool useCross = false);


// 【绘图】在一张图像上显示vector<Point2f>
void drawVecPoints(cv::Mat &mat, vector<cv::Point> points, cv::Scalar color = MC_YELLOW, int thickness = 2, bool useCross = false);


// 【绘图】在一张图像上显示vector<Point2f>         rc 代表的是该点所在的靶标点行列
cv::Mat drawVecPoints(cv::Mat mat, vector<cv::Point2f> points, vector<cv::Point> rc, int col);

//【绘制】在一张图像上显示vector<Point2f>
cv::Mat drawVecPoints(vector<cv::Point2f> points, cv::Size size, cv::Scalar color = MC_BLUE, int thickness = 1);

// 【绘图】在一张图像上绘制vector<Point2f> 画布固定大小为Size(800*600) 
cv::Mat drawVecPoints(vector<cv::Point2f> points, cv::Scalar color, int thickness);

// 在一张图像上显示vector<Point2f>
cv::Mat drawVecPoints(vector<cv::Point2f> points, vector<cv::Point> rc, cv::Size size);

// 【绘图】在图像上显示vector<Point2f> 顺次将点用直线连接起来 二维点序列
void drawSeqPoints(cv::Mat& canvas, vector<cv::Point2f> points, cv::Scalar color = MC_WHITE, int lineWidth = 1);

//【绘制】 在图像上绘制十字
void drawCross(cv::Mat& src, cv::Point center, int len, cv::Scalar& color, int thickness = 1,
	int lineType = 8, int shift = 0);

// 绘制椭圆 x1， y1 代表左上角点坐标  x2 ,y2代表右下角点坐标
void drawEllipse(cv::Mat& src, double x1, double y1, double x2, double y2, cv::Scalar color = MC_YELLOW, int thickness = 2);

// 【绘制】 在图像两点A→B之间绘制箭头
void drawArrow(cv::Mat& src, cv::Point2f A, cv::Point2f B, cv::Scalar& color, int thickness = 1,
	int lineType = 8, int shift = 0);

// 【绘制】 在图像上绘制箭头，箭头的中心点坐标cen与箭头指向angle
// 方向规定水平向右为0°，竖直向上为90°，水平向左为180° angle为角度值
void drawArrow(cv::Mat& src, cv::Point cen, double angle, double len, cv::Scalar& color,
	int thickness = 1, int lineType = 8, int shift = 0);

// 【绘制】在图像上绘制三角形
void triangle(cv::Mat& src, cv::Point A, cv::Point B, cv::Point C, cv::Scalar& color, int thickness = 1);

// 【绘制】在图像上绘制正三角形 r为三角形的尺寸 r为三角形外接圆半径
// thickness = -1 为实心
void triangle(cv::Mat& src, cv::Point p, double r, cv::Scalar& color, int thickness = 1);

// 【绘制】正多边形 在图像上绘制正多边形 ,r为正多边形的外接圆半径 
// angleOff 为偏置角度 水平向右方向为0°，逆时针转则angleOff为正值
void regularPolygon(cv::Mat& src, cv::Point p, int n, double r, cv::Scalar& color, double angleOff = 0, int thickness = 1);

// 【绘制】五角星 在图像上绘制五角星，r为五角星的外接圆尺寸
void drawStar(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness = 1);

// 【绘制】方块 在图像上绘制方块，r为方块的边长
void drawSquare(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness = 1);

// 【绘制】× 绘制叉号 r为某线的长度
void drawSkewCross(cv::Mat& src, cv::Point p, double r, cv::Scalar color, int thickness = 1);

// 【绘制】两组像点的偏差 用十字叉丝来表示
cv::Mat drawDisparity(vector<cv::Point2f> pts1, vector<cv::Point2f> pts2, cv::Scalar color = MC_BLACK);

// 【绘制】两组像点的偏差 用十字叉丝来表示
cv::Mat drawDisparity(vector<vector<cv::Point2f>> pts1, vector<vector<cv::Point2f>>pts2);

// 【绘制】条形图
cv::Mat drawHistogram(vector<double> a, cv::Scalar color);

// 找亚像素角点
// qualityLevel 可允许接受的角点最差质量 默认 0.01
vector<cv::Point2f> getGoodFeaturePoints(const cv::Mat& src, int maxCornerNum,
	double qualityLevel = 0.01, double minDist = 10, cv::Mat mask = cv::Mat(),int blockSize = 3, double k = 0.04 );

// 【判断】点邻域是否有值在low 和 high之间 需确保src为灰度图像
bool hasNbhdPointInRange(cv::Mat& src, cv::Point center, int radius, int low, int high);

// 【判断】点是否符合掩模
bool isInMask(cv::Mat mask, cv::Point point);

// 【筛选】出合规矩的二维点
void siftGoodPoints(vector<cv::Point2f>& points, cv::Mat src);

// 统计掩模真值像素点个数
int countTrueNums(cv::Mat mask);

// 把图片的四条边的像素置零
void setBorderZero(cv::Mat& mat);

// 非零像素点转换为vector<Point>
vector<cv::Point> getTruePoints(cv::Mat mask);

// 返回图像对角线尺寸
float digLength(cv::Mat mat);

// 最佳二值化阈值选取
int otsu(cv::Mat dst);

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
int otsu(unsigned char *image, int rows, int cols, int x0, int y0, int dx, int dy, int vvv = 0);

/* 修改BMP图像尺寸*/
void Resize(CBitmap* src, CBitmap *dst, cv::Size size);

// 载入图像
bool loadImage(cv::Mat& image, CString& filepath, bool grayScale = false);

// 返回横向梯度图
cv::Mat gradX(cv::Mat src);

// 返回纵向梯度图
cv::Mat gradY(cv::Mat src);

// 返回纵向梯度图
cv::Mat gradXY(cv::Mat src);

// 图像旋转 angle为角度值 逆时针为正
cv::Mat getRotatedImg(const cv::Mat& img, double angle, double scale = 1.0);// 图像旋转

std::vector<float> getRowDataUchar(const cv::Mat &img, int rowId);
std::vector<float> getColDataUchar(const cv::Mat &img, int colId);
std::vector<float> getRowDataFloat(const cv::Mat &img, int rowId);
std::vector<float> getColDataFloat(const cv::Mat &img, int colId);

// 旋转变换,原始图像的面积不变 其他部分为黑色 类似最小包围四边形
// 顺时针为正
cv::Mat angleRotate(cv::Mat& src, int angle);

// 匹配
cv::Mat match2Img(cv::Mat A, cv::Mat B);



// 绘制H-S直方图
cv::Mat getH_SHistgram(cv::Mat src, cv::MatND &hist, int hueBinNum = 30);

// 绘制RGB三色直方图 输入图像为彩色图像
cv::Mat getRGBHistgram(cv::Mat src, cv::MatND &hist, int bins = 256);

// 绘制直方图 输入图像为彩色图像或灰度图像 若为彩色图像会转换为灰度图先
cv::Mat getHistgram(cv::Mat src, cv::MatND &hist, int bins = 256);

// 计算两幅图像hist直方图比较的结果
// 四种比较方法为 【CV_COMP_CORREL】【CV_COMP_CHISQR】【CV_COMP_INTERSECT】【CV_COMP_BHATTACHARYYA】 
double calCompareH_SHist(cv::Mat A, cv::Mat B, int method = CV_COMP_CORREL);

// 【反向投影】获取图像相对参考图像的反向投影图
cv::MatND getBackProjImage(cv::Mat src, cv::Mat ref, cv::Mat& hueRefHist, int bins = 30, bool equalHist = false);

// 模版匹配操作 method是匹配方法 共有六种
// 【CV_TM_SQDIFF】
// 【CV_TM_SQDIFF_NORMED】
// 【CV_TM_CCORR】
// 【CV_TM_CCORR_NORMED】
// 【CV_TM_CCOEFF】
// 【CV_TM_CCOEFF_NORMED】
cv::Mat getTemplateMatchImage(cv::Mat src, cv::Mat ref, cv::Point &p, int method = CV_TM_CCOEFF);

// 【保存图像】保存BMP图像  保存单通道的bmp图像 byte=1是好使的，其他的不好使
bool saveBmp(CString bmpName, unsigned char* imgBuf, int width, int height, int byte=1);

// 【保存 | 读取 图像】BMP图像序列  输入为 unsigned char*
int BmpSerialize(CString bmpName, unsigned short *imgBuf, bool bRead, int width, int height, int PicCount);		// int iType

// 【保存图像】BMP图像序列 输入为 float*         coordinateleft1.bin
// cv::Mat保存为二进制文件
bool BmpSerialize(string fileName, cv::Mat data, bool bRead);

// 【保存 | 读取 图像】BMP图像序列  输入为 float*
int BmpSerialize(CString bmpName, float *imgBuf, bool bRead, int width, int height, int PicCount);		// int iType

// 输出cv::Mat的基本信息
void printMatInfo(cv::Mat input);

// 平移操作，图像大小不变
cv::Mat imageTranslation1(cv::Mat& srcImage, int xOffset, int yOffset);

// 平移操作，图像大小改变
cv::Mat imageTranslation2(cv::Mat &srcImage, int xOffset, int yOffset);

// 基于等间隔提取图像缩放
cv::Mat imageReduction1(cv::Mat &srcImage, float kx, float ky);

// 基于区域子块 像素灰度平均值
cv::Vec3b areaAverage(const cv::Mat& srcImage, cv::Point_<int> leftPoint, cv::Point_<int> rightPoint);

// 基于区域子块提取图像缩放
// 区域子块提取图像缩放是通过对源图像进行区域子块划分，然后提取子块中像素值作为采样像素以构成新图像来实现的。
cv::Mat imageReduction2(const cv::Mat& srcImage, double kx, double ky);

// 获得仿射变换图像
cv::Mat getAffineTransformImage(cv::Mat srcImage, const cv::Point2f srcPts[], const cv::Point2f dstPts[]);

// 获得斜切图像
// 左斜为正  右斜为负 角度制
cv::Mat getSkewImage(cv::Mat srcImage, float angle);

// 视频质量评价
// 计算PSNR峰值信噪比，返回数值为30~50dB,值越大越好
double PSNR(const cv::Mat& I1, const cv::Mat& I2);

// 计算MSSIM结构相似性，返回值从0到1，值越大越好
cv::Scalar MSSIM(const cv::Mat& i1, const cv::Mat& i2);

// MatIterator_ 迭代器反色处理
cv::Mat inverseColor4(cv::Mat srcImage);

// isContinuous 反色处理
cv::Mat inverseColor5(cv::Mat srcImage);

// 代码2-28 LUT 查表反色处理
cv::Mat inverseColor6(cv::Mat srcImage);

// 单窗口显示多幅图像
void showManyImages(const std::vector<cv::Mat> &srcImages, cv::Size imgSize);

// 获取HSV图像
cv::Mat getHSVImage(const cv::Mat& image, cv::Mat& image_H, cv::Mat& image_S, cv::Mat& image_V);

// 自适应阈值化
cv::Mat getAdaptiveThresholdImage(const cv::Mat& image, double maxValue=255, int blockSize = 5, double C = 10,
	int adaptiveMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C, int thresholdType = cv::THRESH_BINARY_INV);

// 双阈值化
cv::Mat getDoubleThreshImage(const cv::Mat& image, double lowthresh, double highthresh, double maxValue = 255);

// 半阈值化
cv::Mat getHalfThreshImage(const cv::Mat& image, double thresh);

// 直方图均衡化
cv::Mat getEqualHistImage(const cv::Mat& image, bool useRGB = false);

// 直方图变换——查找
// 直方图变换查找方法实现的思路:
// (1) 将源图像转换为灰度图，计算图像的灰度直方图
// (2) 根据预设阈值参数由低到高查找iLow,再由高到低查找iHigh
// (3) 根据上步得到直方图iLow和iHigh并进行查找变变换
// (4) 通过查找表进行映射变换，完成直方图查找方法变换。
cv::Mat getHistogramTransLUT(const cv::Mat& srcImage, int segThreshold = 50);

// 直方图变换——累计
// 直方图变换累计方法实现的思路；
// （1） 将源图像转换为灰度图，计算图像的灰度直方图
// （2） 建立映射表，对直方图进行像素累积
// （3） 根据映射表进行元素映射得到最终的直方图变换
cv::Mat getHistogramTransAggregate(const cv::Mat& srcImage);

// 直方图匹配
// (1) 分别计算源图像与目标图像的累计概率分布
// (2) 分别对源图像与目标图像进行直方图均衡化操作
// (3) 利用组映射关系使源图像直方图按照规定进行变换
cv::Mat getHistgramMatchImage(const cv::Mat& srcImage, cv::Mat target);

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
cv::Mat getDistTransImage(cv::Mat& srcImage, int thresh = 100);

// 采用opencv自带的距离变换函数
cv::Mat getDistTransImage2(cv::Mat &srcImage, int thresh = 160);

// Gamma校正 伽马校正
// 一般情况下，当Gamma矫正的值大于1时，图像的高光部分被压缩而暗调部分被扩展
// 当Gamma矫正值小于1时，图像的高光部分被扩展而暗调部分被压缩。
cv::Mat getGammaTransformImage(cv::Mat& srcImage, float kFactor);

// 图像线性变换操作
cv::Mat getLinearTransformImage(cv::Mat& srcImage, float a, int b);

// 图像对数变换方法1
// 图像对数变换是将图像输入中范围较窄的低灰度值映射成输出中较宽范围的灰度值，
// 常用于扩展图像中被压缩的（灰度值较高区域的）低像素值。
cv::Mat getLogTransform1(cv::Mat srcImage, float c);

// 图像对数变换方法2
// 图像对数变换是将图像输入中范围较窄的低灰度值映射成输出中较宽范围的灰度值，
// 常用于扩展图像中被压缩的（灰度值较高区域的）低像素值。
cv::Mat getLogTransform2(cv::Mat srcImage, float c);

// 图像对数变换方法3
// 图像对数变换是将图像输入中范围较窄的低灰度值映射成输出中较宽范围的灰度值，
// 常用于扩展图像中被压缩的（灰度值较高区域的）低像素值。
cv::Mat getLogTransform3(cv::Mat srcImage, float c);

// 对比度拉伸操作
cv::Mat getContrastStretchImage(cv::Mat srcImage);

// 灰度级分层
// 将待提取的感兴趣区域的灰度值映射变大或变小，其他不感兴趣的灰度值保持原有值不变，最终输出图像仍为灰度图像
cv::Mat getGrayLayeredImage(cv::Mat srcImage, int controlMin, int controlMax);

// 获得灰度比特平面序列
std::vector<cv::Mat> getMBitPlans(cv::Mat srcImage);

// 最大熵阈值分割
float calculateCurrentEntropy(cv::Mat hist, int threshold);

// 寻找最大熵阈值并分割
cv::Mat maxEntropySegMentation(cv::Mat inputImage);

// 计算图像波峰点
// 投影曲线的波峰/波谷是通过判定其一阶导数为零点，二阶导数为正或负值来确定的，即对于一阶差分D，
// 我们关注的是图像差分的值的大小，因此这里需要将其进行符号化，然后再通过计算二阶差分的变化，
// 找到曲线斜率点U的满足条件（由正到负或由负到正）,点集U正是投影曲线的波峰波谷值.
// 返回的图像为找到的图像波峰图像 resultVec记录了所有的波峰所在列数
cv::Mat findPeak(cv::Mat srcImage, vector<int>& resultVec, int thresh);
 
// 获得垂直投影图像 计算各列白点个数 reduceMat 存储的是计算结果 是CV_32F形式的
cv::Mat getVerticalProjImage(cv::Mat srcImage, cv::Mat & reduceMat);


// 获得水平投影图像 计算各行白点个数 reduceMat 存储的是计算结果 是CV_32F形式的
cv::Mat getHorizontalProjImage(cv::Mat srcImage, cv::Mat & reduceMat);

// 图像操作
// 上层图像是下层低通滤波后通过下采样得到的，扩大后与原级的差值反应的是高斯金字塔两级间的信息差
void Pyramid(cv::Mat srcImage);

// 基于像素邻域的掩码操作
cv::Mat Myfilter2D(cv::Mat srcImage);

// opencv自带库掩码操作
cv::Mat filter2D_(cv::Mat srcImage);

// 图像傅里叶变换
cv::Mat DFT(cv::Mat srcImage);

// 图像离散余弦变换
cv::Mat DCT(cv::Mat srcImage);

// 图像卷积操作
cv::Mat convolution(cv::Mat srcImage, cv::Mat kernel);

// 均值滤波
cv::Mat getBlurImage(const cv::Mat& src, cv::Size ksize = cv::Size(3, 3));

// 中值滤波
cv::Mat getMedianBlurImage(const cv::Mat& src, cv::Size ksize = cv::Size(3, 3));

// 高斯滤波
cv::Mat getGaussianBlurImage(const cv::Mat& src, cv::Size ksize = cv::Size(3, 3), double sigmaX = 1.5, double sigmaY = 1.5);

// 双边滤波
cv::Mat getBilateralFilterImage(const cv::Mat& src, int d = 25, double sigmaColor = 50.0, double sigmaSpace = 12.5);

// 图像导向滤波
cv::Mat guidefilter(cv::Mat &srcImage,int r, double eps);

// 差分边缘检测实现
void diffOperation(const cv::Mat srcImage, cv::Mat& edgeXImage, cv::Mat& edgeYImage);

// 图像非极大值抑制Sobel边缘实现
cv::Mat getSobelVerEdge(cv::Mat srcImage);

// 图像直接卷积Sobel边缘实现 模长阈值为梯度幅值的阈值
cv::Mat getsobelEdge(const cv::Mat& srcImage, uchar threshold);

// 图像卷积下非极大值抑制Sobel实现 flag 可取下面的值
//#define EDGE_SOBEL_VER 0
//#define EDGE_SOBEL_HOR 1
//#define EDGE_SOBEL_ALL 2
cv::Mat getsobelOptaEdge(const cv::Mat& srcImage, int flag = EDGE_SOBEL_ALL);

// OpenCV自带库图像边缘计算 
// flag可取下面的值
//#define EDGE_SOBEL_VER 0
//#define EDGE_SOBEL_HOR 1
//#define EDGE_SOBEL_ALL 2
//#define EDGE_SCHARR_VER 3
//#define EDGE_SCHARR_HOR 4
//#define EDGE_SCHARR_ALL 5
cv::Mat getSobelEdgeImage(const cv::Mat srcImage, int flag);

// 获取Laplace边缘
cv::Mat getLaplaceEdge(cv::Mat srcImage);

// Robert边缘检测
// Robert算子是利用局部差分寻找边缘的一种算子，是最简单的边缘检测算子。Roberts算子利用对角线
// 方向相邻两像素之差近似梯度幅值来检测边缘，检测垂直边缘的效果要优于其他方向边缘，定位精度高，
// 但对噪声的抑制能力较弱。边缘检测算子检查每个像素的邻域并对灰度变化量进行量化，同时也包含
// 方向的确定。
cv::Mat getRobertsEdge(cv::Mat srcImage);

// Prewitt边缘检测
// Prewitt算子是一阶边缘检测算子，该算子对噪声有抑制作用。Prewitt算子对边缘的定位精度不如Roberts算子，
// Sobel算子对边缘检测的准确性更优于Prewitt算子。
cv::Mat getPrewittEdge(cv::Mat srcImage, bool verFlag = false);

// Canny库函数实现 推荐的高与低阈值比值在2:1到3:1之间
cv::Mat getCannyEdge(cv::Mat srcImage, int lowThresh, int highThresh);

// 改进边缘检测算子Marr-Hildreth
// 它把高斯平滑滤波器和拉普拉斯锐化滤波器结合起来，先平滑掉噪声，再进行边缘检测
// 例如 getMarrEdge(srcImage, 9, 1.6);
cv::Mat getMarrEdge(const cv::Mat src, int kerValue, double delta);

// MoravecCorners角点检测
cv::Mat MoravecCorners(cv::Mat srcImage, vector<cv::Point> & points, int kSize = 5, int threshold = 10000);


// 绘制Harris角点 thresh 表示角点质量强度
cv::Mat getHarrisCornersImage(const cv::Mat& srcImage, float thresh = 170, int blockSize = 2, int kSize = 3, double k = 0.04);

// 得到自定义核
// 参数一 int   kshape	: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
// 参数二 int   ksize	: 表示内核的尺寸
// 参数三 Point kpos		: 表示锚点的位置
cv::Mat getCustomKernel(int ksize, int kshape, cv::Point kpos);

// 腐蚀
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getErodeImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// 膨胀
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getDilateImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_ELLIPSE, cv::Point kpos = cv::Point(-1, -1));

// 开运算
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getOpeningOperationImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// 闭运算
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getClosingOperationImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// 形态学梯度
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getMorphGradientImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// 顶帽运算 Top Hat 又称“礼帽”运算
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getMorphTopHatImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// 黑帽运算 Black Hat 
// kshape: 表示内核的形状，有三种选择
//							<1> 矩形：		MORPH_RECT;
//							<2> 交叉形:		MORPH_CROSS;
//							<3> 圆形:		MORPH_ELLIPSE
cv::Mat getMorphBlackHatImage(const cv::Mat& src, int ksize, int kshape = cv::MORPH_RECT, cv::Point kpos = cv::Point(-1, -1));

// 分水岭图像分割
cv::Mat getWatershedSegmentImage(cv::Mat &srcImage, int& noOfSegments, cv::Mat& markers);

// 显示分水岭分割算法结果图像
cv::Mat showWaterSegResult(cv::Mat markers);

// 分割合并
void segMerge(cv::Mat& image, cv::Mat& segments, int & numSeg);

// 颜色通道分离
static void MergeSeg(cv::Mat& img, const cv::Scalar& colorDiff = cv::Scalar::all(1));

// 泛洪填充FloodFill图像分割
cv::Mat getFloodFillImage(const cv::Mat&srcImage, cv::Mat mask, cv::Point pt, int& area, int ffillMode = 1, int loDiff = 20, int upDiff = 20,
	int connectivity = 4, bool useMask = false, cv::Scalar color = MC_GREY, int newMaskVal = 255);

// MeanShift图像分割
cv::Mat getMeanShiftImage(const cv::Mat &srcImage, int spatialRad = 20, int colorRad = 20, int maxPyrLevel = 6);

// Grabcut图像分割
cv::Mat getGrabcutImage(const cv::Mat& srcImage, cv::Rect roi);

// 尺度变换实现
bool CreateScaleSpace(cv::Mat srcImage, std::vector<std::vector<cv::Mat>> &ScaleSpace, std::vector<std::vector<cv::Mat>> &DoG);

// 积分图实现 HOG特征描述实现
// 计算积分图
std::vector<cv::Mat> CalculateIntegralHOG(cv::Mat &srcMat);

// 快速区域积分直方图实现
// 计算单个cell HOG特征
void calHOGinCell(cv::Mat& HOGCellMat, cv::Rect roi, std::vector<cv::Mat>& integrals);

// 获取HOG直方图
cv::Mat getHog(cv::Point pt, std::vector<cv::Mat> &integrals);


// 计算HOG特征
std::vector<cv::Mat> calHOGFeature(cv::Mat srcImage);

// 计算LBP特征
cv::Mat getLBPImage(cv::Mat & srcImage);

// Haar特征提取 计算Haar特征
double HaarExtract(double const **image, int type_, cv::Rect roi);

// 计算单窗口的积分图
double calcIntegral(double const** image, int x, int y, int width, int height);

// 在图片上添加文体辅助函数
void GetStringSize(HDC hDC, const char* str, int* w, int* h);

// 在图片上添加文字 带旋转
void drawString(cv::Mat& dst, string text, cv::Point org, double angle, cv::Scalar color, int fontSize,
	bool italic = false, bool underline = false, bool black = false, const char* fontType = "黑体");

// 在图片上添加文字 支持汉字 可换行 可调字体 可斜体 可加粗
// 添加汉字 添加文字 
void drawString(cv::Mat& dst, string text, cv::Point org, cv::Scalar color, int fontSize,
	bool italic = false, bool underline = false, bool black = false, const char* fontType = "黑体");

// 两点间绘制虚线
void dottedLine(cv::Mat& src, cv::Point A, cv::Point B, cv::Scalar color, int thickness = 1);

// 在图像的某一行绘制虚线
void drawDottedLineRow(cv::Mat& src, int row, cv::Scalar color, int thickness);

// 在图像的某一列绘制虚线
void drawDottedLineCol(cv::Mat& src, int col, cv::Scalar color, int thickness);

// 双线性插值
cv::Vec3d bilinearlInterpolation(cv::Mat src, cv::Point2f pt);

// 计算曝光时间性能
cv::Mat calculateWellExpose(cv::Mat src);

// 计算饱和度	  返回cv::Mat记录各点的饱和度 double
cv::Mat calculateSaturate(cv::Mat src, bool useD = true);

// 计算对比度 返回cv::Mat记录各点的对比度 double
cv::Mat calculateContrast(cv::Mat src, bool useD = false);

// 拉普拉斯金字塔
vector<cv::Mat> laplacian_pyramid(cv::Mat src, int nlev=-1);

// 拉普拉斯金字塔重建
cv::Mat reconstruct_laplacian_pyramid(vector<cv::Mat> pyr);

// 高斯金字塔
vector<cv::Mat> gaussian_pyramid(cv::Mat src, int nlev=-1);

// 频域滤波 低通滤波  返回值value
cv::Mat getDFTBlur(cv::Mat img, cv::Mat& value);

// 【绘制】靶标图像
cv::Mat drawChessboardImage(cv::Size size, int gridWidth, int gridHeight = 0, int offsetX = 0, int offsetY = 0, bool LeftTopBlack = true);

// 【绘制】靶标图像
cv::Mat drawChessboardImage(cv::Size size, cv::Size gridCount, int offsetX = 0, int offsetY = 0, bool LeftTopBlack = true);

// 【绘制】靶标图像 给定格子数和格子的像素尺寸 以及左上角是否为黑
cv::Mat drawChessboardImage(cv::Size gridCount, int gridWidth, int gridHeight, bool LeftTopBlack);

// 合并两幅图 默认为横向合并 左右拼接 左右合并 上下拼接 上下合并
cv::Mat combine2Img(cv::Mat A, cv::Mat B, bool CmbHor = true);

// 【图像压缩】游程编码 对压缩二值图像非常管用
bool runLengthCoding(cv::Mat img, string outputPath);

// 游程编码解压缩
cv::Mat runLengthDecompress(string filepath);

// JPEG图像压缩 输入为彩色JPEG图像
cv::Mat JPEGCompress(cv::Mat src, int level = 95);

// 寻找cv::Mat的最大几个元素 TopEles  寻找前num个像素 vec[0]代表横坐标，vec[1]代表纵坐标,vec[3]代表值
vector<cv::Vec3d> findTopElements(cv::Mat input, int num, bool useFilter = false);

// 区域生长 4邻域
cv::Mat RegionGrow(cv::Mat& src, cv::Point seed, int type = 4);

// 区域生长 4邻域
int grow(cv::Mat src, cv::Mat& mask, cv::Point p, vector<cv::Point>& s, int type = 4);

// 将图片四等分
vector<cv::Mat> divideMatInto4(cv::Mat input, cv::Mat& draw, cv::Rect rect);

// 四叉树分解
void quadtreeSubdivision(cv::Mat input, cv::Mat& draw, cv::Mat cur, int divTimes = 0, int minSize = 5, double thresh = 10.0, cv::Rect rect = cv::Rect(0, 0, 0, 0));

// 灰度极差  灰度最大值减去灰度最小值
double getGrayRange(cv::Mat input);

// 灰度共生矩阵/灰度共现矩阵
// 水平方向 GLCM_HOR 0
// 竖直方向 GLCM_VER 1
// 左斜方向 GLCM_TL 2
// 右斜方向 GLCM_TR 3
// 灰度共生矩阵、灰度共现矩阵 Gray-level co-occurrence matrix
cv::Mat getGLCM(cv::Mat input, int type = GLCM_HOR, int grayLevel = 256);

//把图像归一化为0-255，便于显示
// 可以对图像进行拉伸处理后在进行归一化处理 默认不进行拉伸
cv::Mat norm_0_255(const cv::Mat& src, cv::Size s = cv::Size(0, 0));


// 绘制某行的灰度变化曲线
void drawGrayCurveInRow(cv::Mat src, int rowIndex);


// 绘制某列的灰度变化曲线
void drawGrayCurveInCol(cv::Mat src, int rowIndex);

// 图像分割  编号默认按行排列
bool ImgSegm(cv::Mat src, string outputPath, cv::Size size, string prefix = "");

// 批量修改文件夹内所有后缀为suffix图像的尺寸 批量修改尺寸 
//bool resizeImgsInFolder(CString folderPath, cv::Size dstSize, CString dstFolder, CString suffix = "jpg", CString dstSuffix = "", bool gray = false);

// 批量修改文件夹内所有后缀为suffix图像的尺寸 批量修改尺寸 
bool resizeImgsInFolder(string folderPath, cv::Size dstSize, string dstFolder, string suffix,
	string dstSuffix = "", bool gray = false);

// Floyd-Steinberg 抖动算法
cv::Mat floydSteinbergDithering(cv::Mat input);

// 求梯度图像 数据类型为CV_32FC1
bool Gradient(cv::Mat src, cv::Mat& dst);

// 显示空间点坐标数值 数字 
cv::Mat showPt3d(cv::Point3d pt);

// 生成加密图像 LUT的数据类型为CV_32SC1
cv::Mat getEncryptImage(cv::Mat input, cv::Mat &lut, int type = ENCRYPT_ALL_PIXELS);

// 解密图像 LUT的数据类型为CV_32SC1
cv::Mat getDecodeImage(cv::Mat input, cv::Mat lut, int type = ENCRYPT_ALL_PIXELS);

// 二维离散小波变换（单通道浮点图像）
// 输入图像要求必须是单通道浮点图像，对图像大小也有要求
// （1层变换：w, h必须是2的倍数；2层变换：w, h必须是4的倍数；3层变换：w, h必须是8的倍数......）
cv::Mat DWT(cv::Mat input, int nLayer);

// 二维离散小波变换（单通道浮点图像）
cv::Mat IDWT(cv::Mat src, int nLayer);

// 生成掩膜图像 灰度掩膜 [threshLow,threshHigh]  中间值取255，否则为0
// inverse可以取反 即中间值置零，否则为255
cv::Mat generateGrayMask(cv::Mat input, uchar threshLow, uchar threshHight, bool inverse = false);

// 图像的主成分分析 PCA变换
// number_principal_compent 保留最大的主成分数 默认为0 保留所有成分分量
cv::Mat PCATrans(cv::Mat src, int number_principal_compent = 0);

// 制作GIF动画
void makeGIF(cv::Size size, string gifName = "", int fps = 15);

// 制作GIF动画 
// 需要将C:\\FFMEPG\\bin文件夹中的exe文件都放在
// fps为帧率
void makeGIF(string folderPath, string prefix, string savePath, cv::Size size = cv::Size(0, 0), int fps = 15);

// 交换左上右下块 和 右上左下块
cv::Mat fftshift(cv::Mat input);

// 截屏函数
cv::Mat screenshot();

// 为图像添加透明度通道 输入为单通道或三通道图像
// alpha为透明度，1为不透明，0为透明
cv::Mat addAlphaChannel(cv::Mat input, double alpha);

// 全屏显示
//namedWindow("s", CV_WINDOW_NORMAL);
//setWindowProperty("s", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);