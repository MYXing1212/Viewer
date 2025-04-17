#pragma once
#include<stdio.h>
#include<opencv2\opencv.hpp>
#include<math.h>
#include<stdlib.h>
#include"MyVector.h"
#include"color.h"

using namespace std;

// Point 按X值进行比较
class comp_Point_X{
public:
	bool operator()(cv::Point A, cv::Point B)
		// Compare x and y based on their last base-10 digits:
	{
		return A.x < B.x;
	}
};

// Point 按Y值进行比较
class comp_Point_Y{
public:
	bool operator()(cv::Point A, cv::Point B)
		// Compare x and y based on their last base-10 digits:
	{
		return A.y < B.y;
	}
};

bool compare_x_Point(cv::Point A, cv::Point B);		// Point 按X值进行比较
bool compare_y_Point(cv::Point A, cv::Point B);		// Point 按Y值进行比较

// 【最高点编号】 图像最高点对应y坐标最小
int idxMinY(vector<cv::Point2f> points, bool* flag);

//【最左点编号】 图像最左点对应x坐标最小
int idxMinX(vector<cv::Point2f> points, bool* flag);

// 两点间连线与水平向右向量的夹角 angle = 向量AB 与 水平向右向量 的夹角
// 返回值在0到360之间
// 应注意到图像坐标系y是向下增加的，所以当A = Point（1,1） B = Point(1 , 2) 时，结果为270度，而不是 90
double anglePitch(cv::Point A, cv::Point B);

// 两点间斜率
double slope(double x1, double y1, double x2, double y2);

// 两点间斜率
double slope2(double x1, double y1, double x2, double y2);

// 两点间斜率
double slope(cv::Point2f p1, cv::Point2f p2);

// 两点间【距离】
double dist(cv::Point p1, cv::Point p2);

// 两点间【距离】
double dist(cv::Point2d p1, cv::Point2d p2);

// 两点间距离 【距离】
int distInt(cv::Point p1, cv::Point p2);

// 两点间【距离】
double distance(double x1, double y1, double x2, double y2);

// 两点间【距离】
float dist(cv::Point2f p1, cv::Point2f p2);

// 【距离】点到 [点集]中的点 距离最小值
float minDist(cv::Point p, cv::Mat mask);

// A有m个点， B有n个点，求A中各点到B的最小距离，得到minDist[m],求minDist的最大值 并找到最大值对应A中那个点
float maxR(cv::Mat A, cv::Mat B, cv::Point& cen, float scale = 1.0);

// 【点聚类】
//void 

// 【计算】行数和列数 棋盘格角点
int calRowAndCol(vector<cv::Point2f> points, vector<cv::Point>& rc);

// 【输出】二维点
void printPt2f(cv::Point2f point);

// 【输出】二维点
void printPt2d(cv::Point2d point);

// 【输出】二维点集
void printPts2f(vector<cv::Point2f> points);

// 【输出】二维点集
void printPts2d(vector<cv::Point2d> points);

// 计算欧式距离
float calcEuclideanDistance(int x1, int y1, int x2, int y2);

// 计算棋盘距离
int calcChessboardDistance(int x1, int y1, int x2, int y2);

// 计算街区距离
int calcBlockDistance(int x1, int y1, int x2, int y2);

// 生成一个单位圆的采样圆点阵 默认为单位圆
vector<cv::Point2d> generateSamplingCirclePts(int Num, cv::Point2d cen = cv::Point2d(0,0), double radius = 1.0);

// 【循环偏移】二维点集 offset<0 左移  offset>0右移
void recycleMove(vector<cv::Point>& pts, int offset = 1);

// 点集按y坐标进行排序
void sortByY(vector<cv::Point>& input);

// 点集按x坐标进行排序
void sortByX(vector<cv::Point>& input);

// 点集 x坐标最小值所在位置
int idxMinX(vector<cv::Point> input);

// 点集 x坐标最大值所在位置
int idxMaxX(vector<cv::Point> input);

// 点集 y坐标最小值所在位置
int idxMinY(vector<cv::Point> input);

// 点集 y坐标最大值所在位置
int idxMaxY(vector<cv::Point> input);

// 寻找二维平面上的最近点对
double minDifferent(vector<cv::Point> p, int l, int r, vector<cv::Point>& result);

// 在点集中查找离目标点最临近的点 返回最临近点的编号 和最临近距离 最近点
int findNearestPoint(cv::Point target, vector<cv::Point2f> pts, double* minDistance);

// 平面点平移
cv::Point translatePoint(cv::Point input, int offsetx, int offsety);

// 平面点旋转 theta为角度制
cv::Point rotatePoint(cv::Point input, double theta);

// 计算轮廓重心
cv::Point2f calContourBarycenter(vector<cv::Point> contour);

// 平面点集平移
vector<cv::Point> translatePoints(vector<cv::Point> input, int offsetx, int offsety);

// 平面点集平移 theta 为角度制
vector<cv::Point> rotatePoints(vector<cv::Point> input, double theta);

// 平面点变换 R为2*2旋转矩阵 T为1*2矩阵
vector<cv::Point> transformPoints(vector<cv::Point> input, cv::Mat R, cv::Mat T);

// 将二维点集分解为两个向量 第一个向量存横坐标,第二个向量存纵坐标
vector<vector<double>> split(vector<cv::Point> src);

// 将二维点集分解为两个向量 第一个向量存横坐标,第二个向量存纵坐标
vector<vector<double>> split(vector<cv::Point2d> src);

// 将两个向量合成二维点集 第一个向量存横坐标,第二个向量存纵坐标
vector<cv::Point2d> merge2Vec(vector<double> A, vector<double> B);

// 对轮廓计算角度距离图 角度距离图定义
// 对于一个图形，求得图形中心，计算图形外轮廓上任一点与方向向右的水平线的夹角，
// 计算该点与点云中心的距离，逆时针扫一周，以角度为自变量，以两点距离为因变量(如果同一个角度存在多个点，则取最远点计算距离)，得到一个角度距离图。
// 参数count  是把360度划分的个数，默认为360
vector<double> calContourAngleDistMap(vector<cv::Point> contour, int count = 360);

// 二维点集的粗拼接 主要返回旋转角度theta 和 平移向量   返回平移和旋转后的Pts2
// Pts2经过平移T和旋转theta可以得到Pts1
vector<cv::Point> coarseRegistration(vector<cv::Point> Pts1, vector<cv::Point> Pts2, cv::Vec2d& T, double &theta);

// 精拼接 
// Pts2经过平移T和旋转theta可以得到Pts1
vector<cv::Point> preciseRegistration(vector<cv::Point> Pts1, vector<cv::Point> Pts2, cv::Vec2d& T, double &theta);

// Point点集转换为Point2f点集
vector<cv::Point2f> convert2Point2f(vector<cv::Point> input);

// Point2d点集转换为Point点集
vector<cv::Point> convert2Point2d(vector<cv::Point2d> input);

// Mat N行2列转vector<Point>
vector<cv::Point> convert2VecPt(cv::Mat input);

// Point点集转换为cv::Point2d点集
vector<cv::Point2d> convert2Point2d(vector<cv::Point> input);

// Point点集转换为 Mat N行2列 double型
cv::Mat convert2Mat2d(vector<cv::Point> input);

// Point点集转换为 Mat N行2列 float型
cv::Mat convert2Mat2f(vector<cv::Point> input);

// Point点集转换为Point3d点集 第三维z默认为0
cv::Mat convert2Mat3d(vector<cv::Point> input, double z = 0);

// 绘制轮廓
void drawContour(cv::Mat& input, vector<cv::Point> contour, cv::Scalar color, int thickness = 1);

// 二维点集归一化 默认minX = 0; maxX = 1; minY = 0; maxY = 1
vector<cv::Point2d> normalizePts(vector<cv::Point2d> input, double minX = 0, double maxX = 1.0,	double minY = 0, double maxY = 1.0);

// 两点决定一个向量
// 输入	cv::Point2d A
//		cv::Point2d B
// 输出  Vec2d AB
cv::Vec2d vecA2B(cv::Point2d A, cv::Point2d B);

// 全局几何结构特征差异 参见论文《基于混合特征的非刚性点阵配准算法》汤昊林
// 输入:		vector<cv::Point2d> A
//			vector<cv::Point2d> B
// 输出:		Mat G  G的通道数为2 A.size() = n, B.size() = m, 则G为n行m列 
cv::Mat globalGeoFeatureDiff(vector<cv::Point2d> A, vector<cv::Point2d> B);

// 道格拉斯-普克算法 轮廓多边形近似 轮廓简化 轮廓近似
vector<cv::Point> Douglas(vector<cv::Point> src, double D);

// 合并两个点集
vector<cv::Point> merge2VecPoint(vector<cv::Point> A, vector<cv::Point> B);

// 点集的子集
vector<cv::Point> subVecPoint(vector<cv::Point> src, int start, int end);

// 选取轮廓中某点计算特征向量
cv::Vec2d calcFeatureVector(vector<cv::Point> src, int offset);

float flann_knn(cv::Mat& m_destinations, cv::Mat& m_object, vector<int>& ptpairs, vector<float>& dists = vector<float>());


void findBestReansformSVD(cv::Mat& _m, cv::Mat& _d);


// 判断点集中是否包含点
bool containsPt(vector<cv::Point> pts, cv::Point t);

// 坐标点转ID值
int Pt2ID(cv::Point pt, int width);

// ID值转坐标
cv::Point ID2Pt(int ID, int width);
