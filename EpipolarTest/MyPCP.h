#pragma once

#ifndef _AFXDLL
#define _AFXDLL
#endif // !_AFXDLL

#ifdef USE_MFC
#include<afxwin.h>
#include<afxdlgs.h>
#include<afx.h>
#endif

#include<stdio.h>
#include<math.h>
#include<vector>
#include"MyImage.h"
#include<fstream>

#include<glm/glm.hpp>

// ��
struct Pt3D
{
	double x;
	double y;
	double z;
	int flag;

	Pt3D() :x(0), y(0), z(0), flag(0)
	{}

	Pt3D(const double &x_, const double &y_, const double &z_, const int& flag_) :
		x(x_), y(y_), z(z_), flag(flag_)
	{}

	Pt3D &operator-(Pt3D& value)
	{
		x = x - value.x;
		y = y - value.y;
		z = z - value.z;
		return *this;
	}

	friend std::ostream &operator<<(std::ostream &out, Pt3D value);
};


struct NormVec
{
	double x;
	double y;
	double z;

	NormVec() :x(1.0), y(0), z(0)
	{}

	NormVec(const double &x_, const double &y_, const double &z_)
	{
		double tmp = std::sqrt(x_*x_ + y_*y_ + z_*z_);
		x = x_ / tmp;
		y = y_ / tmp;
		z = z_ / tmp;
	}
};

// �����
struct FittingSphere{
	double radius;
	Pt3D center;
};

// ���ƽ��
struct FittingPlane{
	double A;
	double B;
	double C;
	double stdError;
	double maxFDiv;
	double maxNDiv;
};

// �����˵��ŵľ���
struct CodeDist{
	int id1;
	int id2;
	double dist;
	int rank;

	CodeDist()
	{
		rank = -1;
	}

	CodeDist(const int &id1_, const int &id2_, const double &dist_) :
		id1(id1_), id2(id2_), dist(dist_)
	{}

	CodeDist& operator=(CodeDist& value)
	{
		id1 = value.id1;
		id2 = value.id2;
		dist = value.dist;
		return *this;
	}

	int jointPt(CodeDist& value){
		if (id1 == value.id1 || id1 == value.id2)
			return id1;
		if (id2 == value.id1 || id2 == value.id2)
			return id2;
		return -1;
	}

	friend std::ostream &operator<<(std::ostream &out, CodeDist value);
};

// Point ��Xֵ���бȽ�
class comp_CodeDist{
public:
	bool operator()(CodeDist A, CodeDist B)
	{
		return A.dist<B.dist;
	}
};

// ���½��������������
std::vector<Pt3D> createNoiSphePtC(double radius, Pt3D cen, double noiseScale);

// ���½�����������ƽ��
std::vector<Pt3D> createNoiPlane(double A, double B, double C, double noiseScale);

// ��������ơ� Data3D Ϊ3*N 
// ���ptInRowΪtrue��Data3D ΪN*3
bool loadPtC(CString filepath, cv::Mat& Data3D, bool ptInRow = false);

// ��������ơ� Data3D Ϊ3*N  ����ĵ��� �ȶ�������
bool loadPtCArchive(CString filepath, cv::Mat& Data3D);

// ��������ơ� 
bool loadPtC(CString filepath, std::vector<Pt3D>& src);

// ���ļ����������
std::vector<glm::vec3> loadPtC_GL(const std::string &filepath);

// ���ļ����������
std::vector<glm::dvec3> loadPtCd_GL(const std::string &filepath);

// ���ļ����������
std::vector<glm::vec2> loadPt2d_GL(const std::string &filepath);


// ��������ơ� 
std::vector<cv::Point3d> loadPtC(std::string filepath);

// �����桿���Ƶ��ļ�
bool savePtC(char* filepath, std::vector<Pt3D> src);

// �����桿���Ƶ��ļ�
bool savePtC(std::string filepath, std::vector<cv::Point3d> pts);

// �����ά�㼯���ļ�
bool save(std::string filepath, std::vector<cv::Point2d> pts);

// �����桿���Ƶ��ļ�
bool savePtC(std::string filepath, cv::Mat data);

// �����桿���Ƶ��ļ� �ȴ��ĸ��� ��notepad++ �п������� ���ǿ���������loadPtCArchive�ķ���������
bool savePtCArchive(CString filepath, cv::Mat Data3D);

// �����Ʋ�ֳ�3��double����
void splitXYZ(std::vector<Pt3D> input, double* x, double* y, double* z);
void splitXYZ(std::vector<cv::Point3d> input, std::vector<double>& x, std::vector<double>& y, std::vector<double>& z);

// ���������������� ����Ϊasc�ļ�
bool exportCloud(std::vector<cv::Point3d> cloud);

// ����ռ���ά�㼯 input Ϊ�ڵ� �Ϻ��׵�
std::vector<Pt3D> getPt3DsFromImg(cv::Mat input);

// ��Img�ϻ���Pt3Ds
bool drawPt3DsToImg(cv::Mat& A, std::vector<Pt3D> pts);

// Ѱ����������Զ�ĵ�
glm::vec3 farthestPoint(const std::vector<glm::vec3> &pts, const glm::vec3 &cen_ = glm::vec3(0));

// �ҵ����ֵ��
int findMaxPt3D(std::vector<Pt3D> pts, std::string s);

// �ҵ���Сֵ��
int findMinPt3D(std::vector<Pt3D> pts, std::string s);

// ����������
double Distance(Pt3D p1, Pt3D p2);

// ����������
double Distance(cv::Point3d p1, cv::Point3d p2);

// Ѱ��������Ե�ϵĵ�

// Ѱ�����ٽ���
int findNearestPt3D(std::vector<Pt3D> pts, int begin, double threshDist = 200.0);

/* ======================================================== */
/* ��Ƿ����μ�����L. Thurfjell, E. Bengtsson, B. Nordin,   */
/* A new three dimensional connected components labeling    */
/* algorithm with simultaneous object feature               */
/* extraction capability,  CVGIP 54, 1992, 357-364.         */
/* ======================================================== */
int* ConstrainedComponentLabelingM(char *Image, int iRows, int iCols, int* NoOfLabels, int iDilateCount = 1);

// Point_World��Point_Workpiece Ϊ3*4
void FindReferencePts(cv::Mat PointO_World, cv::Mat PointA_World, cv::Mat Vector_OB_World,
	cv::Mat &Point_World, cv::Mat &Point_Workpiece);

// �������е��ƽ���������
double getAverDist(std::vector<Pt3D> pts);

// ���ƶ����
void DrawPoly(cv::Mat& img, std::vector<Pt3D> pts);

// ����B�������ߵ�ʱ����Ҫ�ĺ���
void DaoShiTwo(Pt3D point1, Pt3D point2, Pt3D* pPoint);

//���ۼ��ҳ�Ϊ�����������߶˵㵼ʧ
void DaoShiThree(Pt3D point1, Pt3D point2, Pt3D point3, Pt3D* pPoint);

//�����������ֵ�㷴����ƶ���
std::vector<Pt3D> GetControlPointByInputPoint(std::vector<Pt3D> input);

// �������ξ���B��������
void DrawBSpline(cv::Mat& img, std::vector<Pt3D> pts, int begin = 0, int end = -1);

//�������ξ���B��������
void DrawBSpline(cv::Mat& img, Pt3D point1, Pt3D point2, Pt3D point3, Pt3D point4);

// ƽ�Ƶ���
std::vector<Pt3D> translatePts(std::vector<Pt3D> input, cv::Vec3d vec);

// ���ɿռ�㼯�ľ������ �Գ��󣬶Խ�Ԫ�ض�Ϊ0
cv::Mat DistMat(std::vector<cv::Point3d> input);

// ���ɿռ�㼯�ľ������ �Գ��󣬶Խ�Ԫ�ض�Ϊ0
std::vector<CodeDist> DistMap(std::vector<cv::Point3d> input, bool ifSort = false);

// ���Ƶ�RT�任
// pts 3*N CV_64FC1
// RT 4*4 ƽ�������ڵ�4��
// ����ֵ 3*N CV_64FC1
cv::Mat RT_Transform(cv::Mat pts, cv::Mat RT);

// ���Ƶ�RT�任
// pts 3*N CV_64FC1
// R 3*3 ��ת����
// T 3*1 ƽ������
// ����ֵ 3*N CV_64FC1
cv::Mat RT_Transform(cv::Mat pts, cv::Mat R, cv::Mat T);

std::vector<cv::Point3f> RT_Transform(std::vector<cv::Point3f> pts, cv::Mat rvec, cv::Mat tvec);

// ���Ƶ�RT�任
std::vector<glm::vec3> RT_Transform(const std::vector<glm::vec3> &pts, glm::vec3 rvec, glm::vec3 tvec);

std::vector<glm::vec3> RT_Transform(const std::vector<glm::vec3> &pts, cv::Mat rmat, cv::Mat tvec);

// ���Ƶ�RT�任
std::vector<glm::vec3> RT_Transform(const std::vector<glm::vec3> &pts, glm::mat4 trans);
glm::vec3 RT_Transform(const glm::vec3 &pts, glm::mat4 trans);
glm::vec3 RT_Transform(const glm::vec3 &pt, glm::mat3 rmat, glm::vec3 tvec);
glm::vec3 RT_Transform_inv(const glm::vec3 &pt, glm::mat3 rmat, glm::vec3 tvec);


// ���Ƶ�RT��任
// pts 3*N CV_64FC1 ���ɽ����RT�任�õ���
// R 3*3 ��ת����
// T 3*1 ƽ������
// ����ֵ 3*N CV_64FC1 �任ǰ�ĵ���
cv::Mat RT_Transform_Inv(cv::Mat pts, cv::Mat R, cv::Mat T);

// Mat תPoint3d 
cv::Point3d Mat2Point3d(cv::Mat input);

// Point3d A ��Point3d B������ 
cv::Vec3d pt2ptVec(cv::Point3d A, cv::Point3d B);

// �����Ƶľ����׼��
// Mat A B ����Ƭ��ά�������� ������3*N ����N*3 ����4*N ���� N*4
// rowData Ϊtrue ��A��B��ÿ��Ϊ������
double stdDist(cv::Mat A, cv::Mat B, bool rowData = true);

// ��Point3d תΪMat 3*1 ���� 4*1
// ifHC �Ƿ�תΪ������� ���Ϊtrue������󲹸�1
cv::Mat pt3d2mat(cv::Point3d pt, bool ifHC = false);

// ��Point2d תΪMat 2*1 ���� 3*1
// ifHC �Ƿ�תΪ������� ���Ϊtrue������󲹸�1
cv::Mat pt2d2mat(cv::Point2d pt, int nch = 1, bool ifHC = false);

// ��MatתΪPoint2d
cv::Point2d mat2pt2d(cv::Mat pt, int nch = 1);

// ��MatתΪPoint3d
cv::Point3d mat2pt3d(cv::Mat pt);

// std::vector<glm::vec3> תMat
cv::Mat pt3dVec2Mat(std::vector<glm::vec3> pts);

// ��������� ������������� Tips OpenGL
glm::vec3 getSpherePointf(float u, float v, float radius);
glm::dvec3 getSpherePointd(double u, double v, double radius);

// ����任 ����ת��ƽ��
void rotateAndMove(std::vector<glm::vec3> &pts, glm::mat3 RMat, glm::vec3 pos);
void rotateAndMove(std::vector<glm::dvec3> &pts, glm::dmat3 RMat, glm::dvec3 pos);


// ��vector<Point3d>ת��Ϊcv::Mat  N*3
template<typename T>
cv::Mat pt3dVec2Mat(std::vector<cv::Point3_<T>> pts){
	cv::Mat b = cv::Mat(pts);
	//cout << "b = " << b << endl;
	cv::Mat c = b.clone();
	cv::Mat result = c.reshape(1, (int)pts.size());

	result.convertTo(result, CV_64FC1);
	//= b.reshape(1, (int)pts.size());
	//cout << "c = " << c << endl;
	return result;
}

// �㼯���Ļ� 
// �㼯���㶼��ȥ�㼯��ƽ����
// Center ����3*1 double
cv::Mat Centerize(cv::Mat& pts, cv::Mat& Center);

// ������Ƶ�����
// ����㼯Ϊ3*N ���� N*3
// ����ֵΪ�������� 3*1 double
cv::Mat calcCenter(cv::Mat pts);

glm::vec3 calcCenter(const std::vector<glm::vec3> &pts);
glm::dvec3 calcCenter(const std::vector<glm::dvec3> &pts);

// ���ɷֶ��� result * src = dst
glm::mat4 registerPCA(const std::vector<glm::vec3> &src, const std::vector<glm::vec3> &dst);

// ����	Point_O_World	��������1*3��		ת����˵�Ϊ����ԭ��
//		Point_X_World	x����һ�� 1*3		ת����˵���X����
//		Point_OZ_World	ƽ�淨��1*3			ת����˷�����Ӧ��Z���غ�
//		Points			ԭʼ����3*N			���任�ĵ������
// ���	Point_Trans		ת��������3*N		�任��ĵ������
//		R_rotate		��ת����3*3
//		T_rotate		ƽ������3*1
void ReferenceframeTransform(cv::Mat Point_O_World, cv::Mat Point_X_World, cv::Mat Point_OZ_World,
	cv::Mat Points, cv::Mat &Points_Trans, cv::Mat &R_rotate, cv::Mat &T_rotate);


// ��������O,A,B���еĽǶȡ�AOB
// ��������ά����߶�ά��
// O, A, B����ʹ������Ҳ������������
template<typename T>
T angleOAB(cv::Mat O, cv::Mat A, cv::Mat B){
	T *po = (T*)O.data;
	T *pa = (T*)A.data;
	T *pb = (T*)B.data;
	int n = (int)A.total();
	cv::Mat OA = (cv::Mat_<T>(n, 1));
	cv::Mat OB = (cv::Mat_<T>(n, 1));
	for (int i = 0; i < n;i++){
		OA.ptr<T>(0)[i] = pa[i] - po[i];
		OB.ptr<T>(0)[i] = pb[i] - po[i];
	}
	return angleV<T>(OA, OB);
}

// ������ά���Ƶ�vector<Point>
template<typename T>
std::vector<cv::Point3_<T>> loadPtCFromFile(std::string filepath){
	std::vector<cv::Point3_<T>> result;
	fstream f(filepath, ios_base::in);
	if (!f.is_open())
	{
		cout << "loadPtC ���ļ�ʧ��!" << endl;
		exit(1);
	}
	while (!f.eof()){
		cv::Point3_<T> pt;
		f >> pt.x >> pt.y >> pt.z;
		result.push_back(pt);
	}
	return result;
}

bool savePtC(std::string filepath, std::vector<glm::vec3> pts);
bool savePtC(std::string filepath, std::vector<glm::dvec3> pts);

// �����桿���Ƶ��ļ�
template<typename T>
bool savePtC(std::string filepath, std::vector<cv::Point3_<T>> pts)
{
	std::ofstream r(filepath);
	for (int i = 0; i < (int)pts.size(); i++)
	{
		r << setprecision(12) << pts[i].x << " " << pts[i].y << " " << pts[i].z << endl;
	}
	return true;
}

// �����ά�㼯���ļ�
template<typename T>
bool savePtC(std::string filepath, std::vector<cv::Point_<T>> pts)
{
	std::ofstream r(filepath);
	for (int i = 0; i < (int)pts.size(); i++)
	{
		r << setprecision(12) << pts[i].x << " " << pts[i].y << endl;
	}
	return true;
}

// �����桿���Ƶ��ļ�
template<typename T>
bool savePtC(std::string filepath, cv::Mat data){
	cv::Mat d = data.clone();
	if (d.rows == 3 && d.cols > 3)
		d = d.t();
	std::ofstream r(filepath);
	for (int i = 0; i < d.rows; i++)
	{
		r << setprecision(12) << d.ptr<T>(i)[0] << " " << d.ptr<T>(i)[1] << " " << d.ptr<T>(i)[2] << endl;
	}
	r.close();
	return true;
}

// �ռ���ȼ��������
std::vector<glm::vec3> sampleInSpace3D(float left, float right, float top, float bottom, float nearPlane, float farPlane,
	float step = 3.0);

// ȥ��������Ҫ��ĵ� 
// ����pts������judgeOutliers�ĵ㶼���ߵ�
std::vector<glm::vec3> removeOutliers(const std::vector<glm::vec3> &pts,
	const std::function<bool(glm::vec3)> judgeOutliers);

// ����һ��㣬����������ֵ��x,y,z���������ϵķ�Χ ��Χ��
void boundingBox(glm::vec3 *pts, int cnt, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax);
void boundingBox(std::vector<glm::vec3> pts, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax);
cv::Vec6f boundingBox(std::vector<glm::vec3> pts);
cv::Vec4f boundingBox(std::vector<glm::vec2> pts);
cv::Vec6d boundingBox(std::vector<glm::dvec3> pts);

// �����Χ�еİ뾶
float calBoundingBoxRadius(cv::Vec6f box);			

// ����pt��������ϵ�µ����� 
// ���� ��ǰ������ ԭʼ����ϵ
//		������ϵ ԭ��
//		������ϵ x����������
//		������ϵ y����������
glm::vec3 transformCS(glm::vec3 pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);

// ����ϵ�任��RT���� ������ϵ�е�ԭ�� ���᷽����������ϵ�µı�ʾΪ cen xdir ydir zdir
// ��ô������ϵ�µĵ�ͨ��result��RT�任���Եõ�������ϵ�µ�����
glm::dmat4 transformCSRT(glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir, glm::dmat3 &R, glm::dvec3 &T);

// ����pt��������ϵ�µ����� 
// ���� ��ǰ������ ԭʼ����ϵ
//		������ϵ ԭ��
//		������ϵ x����������
//		������ϵ y����������
glm::dvec3 transformCS(glm::dvec3 pt, glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir);

// ����pt��ԭʼ����ϵ�µ�����
// ���� ��ǰ������ ������ϵ��
//		������ϵ ԭ��
//		������ϵ x����������
//		������ϵ y����������
glm::vec3 transformCSInverse(glm::vec3 pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);

// ����pt��������ϵ�µ����� 
// ���� ��ǰ������ 
//		������ϵ ԭ��
//		������ϵ x����������
//		������ϵ y����������
std::vector<glm::vec3> transformCS(std::vector<glm::vec3> pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);

// ����pt��������ϵ�µ����� 
// ���� ��ǰ������ 
//		������ϵ ԭ��
//		������ϵ x����������
//		������ϵ y����������
//		������ϵ z����������
std::vector<glm::dvec3> transformCS(std::vector<glm::dvec3> pt, glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir);

// ����pt��ԭʼ����ϵ�µ�����
// ���� ��ǰ������ ������ϵ��
//		������ϵ ԭ��
//		������ϵ x����������
//		������ϵ y����������
std::vector<glm::vec3> transformCSInverse(std::vector<glm::vec3> pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);


// ������ڵ�������
std::vector<glm::vec3> generateSphericalCrownData(glm::vec3 cen, glm::vec3 north, float radius, float angle, 
	float noise = 0);

// ���½�����������ƽ��
std::vector<glm::vec3> generateNoisePlane(const glm::vec3 &norm, const glm::vec3 &cen, const float &width,
	const float &height,
	const float &noiseScale,
	const float &sampleCnt = 20);

// angle Ϊ�Ƕ��� �ռ�Բ
std::vector<glm::vec3> generateCircle3D(const glm::vec3 &cen, const glm::vec3 &norm, const float &radius, const int &pointCnt = 100,
	const float &angle = 360.0f);
std::vector<glm::dvec3> generateCircle3D(const glm::dvec3 &cen, const glm::dvec3 &norm, const double &radius, const int &pointCnt = 100,
	const double &angle = 360.0f);

// ������ڸ߶�
// radius ����뾶
// diameterCrossSection ����ڽ���ֱ��
float calcSphericalCapHeight(float radius, float diameterCrossSection);

// ������� ���ĽǶ�
// radius ��뾶
// diameterCrossSection ����ڽ���ֱ��
float calcSphericalCapAngle(float radius, float diameterCrossSection);

// �ռ����ߺ������ཻ
bool intersect(glm::vec3 start, glm::vec3 dir, glm::vec3 sphereCen, float radius, float &t0, float &t1);
bool intersect(cv::Vec6f line, glm::vec3 sphereCen, float radius, float &t0, float &t1);


// lǰ��������ʾ���� ��3������ʾ�����ϵĵ�
bool intersectHalflineAndSphere(cv::Vec6d l, glm::dvec3 cen, double radius, std::vector<glm::dvec3> &result);

// �ϲ�����ӳ�����
cv::Mat mergeMultiViewPts(std::vector<std::vector<cv::Point3f>> input);

// ��ѡ�㼯 ����δ��ѡ�еĵ�����
std::vector<glm::vec3> inverseSelect(const std::vector<glm::vec3> &pts, const std::vector<int> &selectIds);

// ��ѡ�㼯 ���ر�ѡ�еĵ�����
std::vector<glm::vec3> select(const std::vector<glm::vec3> &pts, const std::vector<int> &selectIds);

// 
std::vector<glm::dvec3> convert2dvec3s(std::vector<glm::vec3> pts);

// ��ĳ������һ���������Ƶ� �޶�size
std::vector<glm::vec3> copyPtsAlongDir(glm::vec3 pt, glm::vec3 dir,
	float step = 1.0, float size = 100.0, bool invertDir = false, bool bothDir = false);

void addNoiseToPtCloud(std::vector<glm::vec3> &pts, double noiseLevel);

// Ѱ�����ٽ���
int findNearestPt2D(glm::vec2 input, std::vector<glm::vec2> pts, double threshDist, double *minDist = NULL);
