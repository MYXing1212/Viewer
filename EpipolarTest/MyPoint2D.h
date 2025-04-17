#pragma once
#include<stdio.h>
#include<opencv2\opencv.hpp>
#include<math.h>
#include<stdlib.h>
#include"MyVector.h"
#include"color.h"

using namespace std;

// Point ��Xֵ���бȽ�
class comp_Point_X{
public:
	bool operator()(cv::Point A, cv::Point B)
		// Compare x and y based on their last base-10 digits:
	{
		return A.x < B.x;
	}
};

// Point ��Yֵ���бȽ�
class comp_Point_Y{
public:
	bool operator()(cv::Point A, cv::Point B)
		// Compare x and y based on their last base-10 digits:
	{
		return A.y < B.y;
	}
};

bool compare_x_Point(cv::Point A, cv::Point B);		// Point ��Xֵ���бȽ�
bool compare_y_Point(cv::Point A, cv::Point B);		// Point ��Yֵ���бȽ�

// ����ߵ��š� ͼ����ߵ��Ӧy������С
int idxMinY(vector<cv::Point2f> points, bool* flag);

//��������š� ͼ��������Ӧx������С
int idxMinX(vector<cv::Point2f> points, bool* flag);

// �����������ˮƽ���������ļн� angle = ����AB �� ˮƽ�������� �ļн�
// ����ֵ��0��360֮��
// Ӧע�⵽ͼ������ϵy���������ӵģ����Ե�A = Point��1,1�� B = Point(1 , 2) ʱ�����Ϊ270�ȣ������� 90
double anglePitch(cv::Point A, cv::Point B);

// �����б��
double slope(double x1, double y1, double x2, double y2);

// �����б��
double slope2(double x1, double y1, double x2, double y2);

// �����б��
double slope(cv::Point2f p1, cv::Point2f p2);

// ����䡾���롿
double dist(cv::Point p1, cv::Point p2);

// ����䡾���롿
double dist(cv::Point2d p1, cv::Point2d p2);

// �������� �����롿
int distInt(cv::Point p1, cv::Point p2);

// ����䡾���롿
double distance(double x1, double y1, double x2, double y2);

// ����䡾���롿
float dist(cv::Point2f p1, cv::Point2f p2);

// �����롿�㵽 [�㼯]�еĵ� ������Сֵ
float minDist(cv::Point p, cv::Mat mask);

// A��m���㣬 B��n���㣬��A�и��㵽B����С���룬�õ�minDist[m],��minDist�����ֵ ���ҵ����ֵ��ӦA���Ǹ���
float maxR(cv::Mat A, cv::Mat B, cv::Point& cen, float scale = 1.0);

// ������ࡿ
//void 

// �����㡿���������� ���̸�ǵ�
int calRowAndCol(vector<cv::Point2f> points, vector<cv::Point>& rc);

// ���������ά��
void printPt2f(cv::Point2f point);

// ���������ά��
void printPt2d(cv::Point2d point);

// ���������ά�㼯
void printPts2f(vector<cv::Point2f> points);

// ���������ά�㼯
void printPts2d(vector<cv::Point2d> points);

// ����ŷʽ����
float calcEuclideanDistance(int x1, int y1, int x2, int y2);

// �������̾���
int calcChessboardDistance(int x1, int y1, int x2, int y2);

// �����������
int calcBlockDistance(int x1, int y1, int x2, int y2);

// ����һ����λԲ�Ĳ���Բ���� Ĭ��Ϊ��λԲ
vector<cv::Point2d> generateSamplingCirclePts(int Num, cv::Point2d cen = cv::Point2d(0,0), double radius = 1.0);

// ��ѭ��ƫ�ơ���ά�㼯 offset<0 ����  offset>0����
void recycleMove(vector<cv::Point>& pts, int offset = 1);

// �㼯��y�����������
void sortByY(vector<cv::Point>& input);

// �㼯��x�����������
void sortByX(vector<cv::Point>& input);

// �㼯 x������Сֵ����λ��
int idxMinX(vector<cv::Point> input);

// �㼯 x�������ֵ����λ��
int idxMaxX(vector<cv::Point> input);

// �㼯 y������Сֵ����λ��
int idxMinY(vector<cv::Point> input);

// �㼯 y�������ֵ����λ��
int idxMaxY(vector<cv::Point> input);

// Ѱ�Ҷ�άƽ���ϵ�������
double minDifferent(vector<cv::Point> p, int l, int r, vector<cv::Point>& result);

// �ڵ㼯�в�����Ŀ������ٽ��ĵ� �������ٽ���ı�� �����ٽ����� �����
int findNearestPoint(cv::Point target, vector<cv::Point2f> pts, double* minDistance);

// ƽ���ƽ��
cv::Point translatePoint(cv::Point input, int offsetx, int offsety);

// ƽ�����ת thetaΪ�Ƕ���
cv::Point rotatePoint(cv::Point input, double theta);

// ������������
cv::Point2f calContourBarycenter(vector<cv::Point> contour);

// ƽ��㼯ƽ��
vector<cv::Point> translatePoints(vector<cv::Point> input, int offsetx, int offsety);

// ƽ��㼯ƽ�� theta Ϊ�Ƕ���
vector<cv::Point> rotatePoints(vector<cv::Point> input, double theta);

// ƽ���任 RΪ2*2��ת���� TΪ1*2����
vector<cv::Point> transformPoints(vector<cv::Point> input, cv::Mat R, cv::Mat T);

// ����ά�㼯�ֽ�Ϊ�������� ��һ�������������,�ڶ���������������
vector<vector<double>> split(vector<cv::Point> src);

// ����ά�㼯�ֽ�Ϊ�������� ��һ�������������,�ڶ���������������
vector<vector<double>> split(vector<cv::Point2d> src);

// �����������ϳɶ�ά�㼯 ��һ�������������,�ڶ���������������
vector<cv::Point2d> merge2Vec(vector<double> A, vector<double> B);

// ����������ǶȾ���ͼ �ǶȾ���ͼ����
// ����һ��ͼ�Σ����ͼ�����ģ�����ͼ������������һ���뷽�����ҵ�ˮƽ�ߵļнǣ�
// ����õ���������ĵľ��룬��ʱ��ɨһ�ܣ��ԽǶ�Ϊ�Ա��������������Ϊ�����(���ͬһ���Ƕȴ��ڶ���㣬��ȡ��Զ��������)���õ�һ���ǶȾ���ͼ��
// ����count  �ǰ�360�Ȼ��ֵĸ�����Ĭ��Ϊ360
vector<double> calContourAngleDistMap(vector<cv::Point> contour, int count = 360);

// ��ά�㼯�Ĵ�ƴ�� ��Ҫ������ת�Ƕ�theta �� ƽ������   ����ƽ�ƺ���ת���Pts2
// Pts2����ƽ��T����תtheta���Եõ�Pts1
vector<cv::Point> coarseRegistration(vector<cv::Point> Pts1, vector<cv::Point> Pts2, cv::Vec2d& T, double &theta);

// ��ƴ�� 
// Pts2����ƽ��T����תtheta���Եõ�Pts1
vector<cv::Point> preciseRegistration(vector<cv::Point> Pts1, vector<cv::Point> Pts2, cv::Vec2d& T, double &theta);

// Point�㼯ת��ΪPoint2f�㼯
vector<cv::Point2f> convert2Point2f(vector<cv::Point> input);

// Point2d�㼯ת��ΪPoint�㼯
vector<cv::Point> convert2Point2d(vector<cv::Point2d> input);

// Mat N��2��תvector<Point>
vector<cv::Point> convert2VecPt(cv::Mat input);

// Point�㼯ת��Ϊcv::Point2d�㼯
vector<cv::Point2d> convert2Point2d(vector<cv::Point> input);

// Point�㼯ת��Ϊ Mat N��2�� double��
cv::Mat convert2Mat2d(vector<cv::Point> input);

// Point�㼯ת��Ϊ Mat N��2�� float��
cv::Mat convert2Mat2f(vector<cv::Point> input);

// Point�㼯ת��ΪPoint3d�㼯 ����άzĬ��Ϊ0
cv::Mat convert2Mat3d(vector<cv::Point> input, double z = 0);

// ��������
void drawContour(cv::Mat& input, vector<cv::Point> contour, cv::Scalar color, int thickness = 1);

// ��ά�㼯��һ�� Ĭ��minX = 0; maxX = 1; minY = 0; maxY = 1
vector<cv::Point2d> normalizePts(vector<cv::Point2d> input, double minX = 0, double maxX = 1.0,	double minY = 0, double maxY = 1.0);

// �������һ������
// ����	cv::Point2d A
//		cv::Point2d B
// ���  Vec2d AB
cv::Vec2d vecA2B(cv::Point2d A, cv::Point2d B);

// ȫ�ּ��νṹ�������� �μ����ġ����ڻ�������ķǸ��Ե�����׼�㷨�������
// ����:		vector<cv::Point2d> A
//			vector<cv::Point2d> B
// ���:		Mat G  G��ͨ����Ϊ2 A.size() = n, B.size() = m, ��GΪn��m�� 
cv::Mat globalGeoFeatureDiff(vector<cv::Point2d> A, vector<cv::Point2d> B);

// ������˹-�տ��㷨 ��������ν��� ������ ��������
vector<cv::Point> Douglas(vector<cv::Point> src, double D);

// �ϲ������㼯
vector<cv::Point> merge2VecPoint(vector<cv::Point> A, vector<cv::Point> B);

// �㼯���Ӽ�
vector<cv::Point> subVecPoint(vector<cv::Point> src, int start, int end);

// ѡȡ������ĳ�������������
cv::Vec2d calcFeatureVector(vector<cv::Point> src, int offset);

float flann_knn(cv::Mat& m_destinations, cv::Mat& m_object, vector<int>& ptpairs, vector<float>& dists = vector<float>());


void findBestReansformSVD(cv::Mat& _m, cv::Mat& _d);


// �жϵ㼯���Ƿ������
bool containsPt(vector<cv::Point> pts, cv::Point t);

// �����תIDֵ
int Pt2ID(cv::Point pt, int width);

// IDֵת����
cv::Point ID2Pt(int ID, int width);
