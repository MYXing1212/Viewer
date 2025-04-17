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

// 点
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

// 拟合球
struct FittingSphere{
	double radius;
	Pt3D center;
};

// 拟合平面
struct FittingPlane{
	double A;
	double B;
	double C;
	double stdError;
	double maxFDiv;
	double maxNDiv;
};

// 待两端点编号的距离
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

// Point 按X值进行比较
class comp_CodeDist{
public:
	bool operator()(CodeDist A, CodeDist B)
	{
		return A.dist<B.dist;
	}
};

// 【新建】创建球面点云
std::vector<Pt3D> createNoiSphePtC(double radius, Pt3D cen, double noiseScale);

// 【新建】创建噪声平面
std::vector<Pt3D> createNoiPlane(double A, double B, double C, double noiseScale);

// 【载入点云】 Data3D 为3*N 
// 如果ptInRow为true则Data3D 为N*3
bool loadPtC(CString filepath, cv::Mat& Data3D, bool ptInRow = false);

// 【载入点云】 Data3D 为3*N  读入的点云 先读点数量
bool loadPtCArchive(CString filepath, cv::Mat& Data3D);

// 【载入点云】 
bool loadPtC(CString filepath, std::vector<Pt3D>& src);

// 从文件中载入点云
std::vector<glm::vec3> loadPtC_GL(const std::string &filepath);

// 从文件中载入点云
std::vector<glm::dvec3> loadPtCd_GL(const std::string &filepath);

// 从文件中载入点云
std::vector<glm::vec2> loadPt2d_GL(const std::string &filepath);


// 【载入点云】 
std::vector<cv::Point3d> loadPtC(std::string filepath);

// 【保存】点云到文件
bool savePtC(char* filepath, std::vector<Pt3D> src);

// 【保存】点云到文件
bool savePtC(std::string filepath, std::vector<cv::Point3d> pts);

// 保存二维点集到文件
bool save(std::string filepath, std::vector<cv::Point2d> pts);

// 【保存】点云到文件
bool savePtC(std::string filepath, cv::Mat data);

// 【保存】点云到文件 先存点的个数 在notepad++ 中看不出来 但是可以用上面loadPtCArchive的方法读出来
bool savePtCArchive(CString filepath, cv::Mat Data3D);

// 将点云拆分成3个double数组
void splitXYZ(std::vector<Pt3D> input, double* x, double* y, double* z);
void splitXYZ(std::vector<cv::Point3d> input, std::vector<double>& x, std::vector<double>& y, std::vector<double>& z);

// 【导出】导出点云 保存为asc文件
bool exportCloud(std::vector<cv::Point3d> cloud);

// 读入空间三维点集 input 为黑底 上含白点
std::vector<Pt3D> getPt3DsFromImg(cv::Mat input);

// 在Img上绘制Pt3Ds
bool drawPt3DsToImg(cv::Mat& A, std::vector<Pt3D> pts);

// 寻找离质心最远的点
glm::vec3 farthestPoint(const std::vector<glm::vec3> &pts, const glm::vec3 &cen_ = glm::vec3(0));

// 找到最大值点
int findMaxPt3D(std::vector<Pt3D> pts, std::string s);

// 找到最小值点
int findMinPt3D(std::vector<Pt3D> pts, std::string s);

// 求两点间距离
double Distance(Pt3D p1, Pt3D p2);

// 求两点间距离
double Distance(cv::Point3d p1, cv::Point3d p2);

// 寻找两点间边缘上的点

// 寻找最临近点
int findNearestPt3D(std::vector<Pt3D> pts, int begin, double threshDist = 200.0);

/* ======================================================== */
/* 标记方法参见文献L. Thurfjell, E. Bengtsson, B. Nordin,   */
/* A new three dimensional connected components labeling    */
/* algorithm with simultaneous object feature               */
/* extraction capability,  CVGIP 54, 1992, 357-364.         */
/* ======================================================== */
int* ConstrainedComponentLabelingM(char *Image, int iRows, int iCols, int* NoOfLabels, int iDilateCount = 1);

// Point_World、Point_Workpiece 为3*4
void FindReferencePts(cv::Mat PointO_World, cv::Mat PointA_World, cv::Mat Vector_OB_World,
	cv::Mat &Point_World, cv::Mat &Point_Workpiece);

// 计算序列点的平均间隔距离
double getAverDist(std::vector<Pt3D> pts);

// 绘制多边形
void DrawPoly(cv::Mat& img, std::vector<Pt3D> pts);

// 绘制B样条曲线的时候需要的函数
void DaoShiTwo(Pt3D point1, Pt3D point2, Pt3D* pPoint);

//以累加弦长为参数求抛物线端点导失
void DaoShiThree(Pt3D point1, Pt3D point2, Pt3D point3, Pt3D* pPoint);

//根据输入的型值点反算控制顶点
std::vector<Pt3D> GetControlPointByInputPoint(std::vector<Pt3D> input);

// 绘制三次均匀B样条曲线
void DrawBSpline(cv::Mat& img, std::vector<Pt3D> pts, int begin = 0, int end = -1);

//绘制三次均匀B样条曲线
void DrawBSpline(cv::Mat& img, Pt3D point1, Pt3D point2, Pt3D point3, Pt3D point4);

// 平移点云
std::vector<Pt3D> translatePts(std::vector<Pt3D> input, cv::Vec3d vec);

// 生成空间点集的距离矩阵 对称阵，对角元素都为0
cv::Mat DistMat(std::vector<cv::Point3d> input);

// 生成空间点集的距离矩阵 对称阵，对角元素都为0
std::vector<CodeDist> DistMap(std::vector<cv::Point3d> input, bool ifSort = false);

// 点云的RT变换
// pts 3*N CV_64FC1
// RT 4*4 平移向量在第4列
// 返回值 3*N CV_64FC1
cv::Mat RT_Transform(cv::Mat pts, cv::Mat RT);

// 点云的RT变换
// pts 3*N CV_64FC1
// R 3*3 旋转矩阵
// T 3*1 平移向量
// 返回值 3*N CV_64FC1
cv::Mat RT_Transform(cv::Mat pts, cv::Mat R, cv::Mat T);

std::vector<cv::Point3f> RT_Transform(std::vector<cv::Point3f> pts, cv::Mat rvec, cv::Mat tvec);

// 点云的RT变换
std::vector<glm::vec3> RT_Transform(const std::vector<glm::vec3> &pts, glm::vec3 rvec, glm::vec3 tvec);

std::vector<glm::vec3> RT_Transform(const std::vector<glm::vec3> &pts, cv::Mat rmat, cv::Mat tvec);

// 点云的RT变换
std::vector<glm::vec3> RT_Transform(const std::vector<glm::vec3> &pts, glm::mat4 trans);
glm::vec3 RT_Transform(const glm::vec3 &pts, glm::mat4 trans);
glm::vec3 RT_Transform(const glm::vec3 &pt, glm::mat3 rmat, glm::vec3 tvec);
glm::vec3 RT_Transform_inv(const glm::vec3 &pt, glm::mat3 rmat, glm::vec3 tvec);


// 点云的RT逆变换
// pts 3*N CV_64FC1 是由结果经RT变换得到的
// R 3*3 旋转矩阵
// T 3*1 平移向量
// 返回值 3*N CV_64FC1 变换前的点云
cv::Mat RT_Transform_Inv(cv::Mat pts, cv::Mat R, cv::Mat T);

// Mat 转Point3d 
cv::Point3d Mat2Point3d(cv::Mat input);

// Point3d A 到Point3d B的向量 
cv::Vec3d pt2ptVec(cv::Point3d A, cv::Point3d B);

// 两点云的距离标准差
// Mat A B 是两片三维点云数据 可以是3*N 或者N*3 或者4*N 或者 N*4
// rowData 为true 则A，B的每行为点坐标
double stdDist(cv::Mat A, cv::Mat B, bool rowData = true);

// 将Point3d 转为Mat 3*1 或者 4*1
// ifHC 是否转为齐次坐标 如果为true，则最后补个1
cv::Mat pt3d2mat(cv::Point3d pt, bool ifHC = false);

// 将Point2d 转为Mat 2*1 或者 3*1
// ifHC 是否转为齐次坐标 如果为true，则最后补个1
cv::Mat pt2d2mat(cv::Point2d pt, int nch = 1, bool ifHC = false);

// 将Mat转为Point2d
cv::Point2d mat2pt2d(cv::Mat pt, int nch = 1);

// 将Mat转为Point3d
cv::Point3d mat2pt3d(cv::Mat pt);

// std::vector<glm::vec3> 转Mat
cv::Mat pt3dVec2Mat(std::vector<glm::vec3> pts);

// 计算球面点 利用球参数方程 Tips OpenGL
glm::vec3 getSpherePointf(float u, float v, float radius);
glm::dvec3 getSpherePointd(double u, double v, double radius);

// 坐标变换 先旋转后平移
void rotateAndMove(std::vector<glm::vec3> &pts, glm::mat3 RMat, glm::vec3 pos);
void rotateAndMove(std::vector<glm::dvec3> &pts, glm::dmat3 RMat, glm::dvec3 pos);


// 将vector<Point3d>转换为cv::Mat  N*3
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

// 点集中心化 
// 点集各点都减去点集的平均点
// Center 都是3*1 double
cv::Mat Centerize(cv::Mat& pts, cv::Mat& Center);

// 计算点云的重心
// 输入点集为3*N 或者 N*3
// 返回值为点云重心 3*1 double
cv::Mat calcCenter(cv::Mat pts);

glm::vec3 calcCenter(const std::vector<glm::vec3> &pts);
glm::dvec3 calcCenter(const std::vector<glm::dvec3> &pts);

// 主成分对齐 result * src = dst
glm::mat4 registerPCA(const std::vector<glm::vec3> &src, const std::vector<glm::vec3> &dst);

// 输入	Point_O_World	中心坐标1*3，		转换后此点为坐标原点
//		Point_X_World	x轴上一点 1*3		转换后此点在X轴上
//		Point_OZ_World	平面法向1*3			转换后此法向量应与Z轴重合
//		Points			原始坐标3*N			待变换的点的坐标
// 输出	Point_Trans		转换后坐标3*N		变换后的点的坐标
//		R_rotate		旋转矩阵3*3
//		T_rotate		平移向量3*1
void ReferenceframeTransform(cv::Mat Point_O_World, cv::Mat Point_X_World, cv::Mat Point_OZ_World,
	cv::Mat Points, cv::Mat &Points_Trans, cv::Mat &R_rotate, cv::Mat &T_rotate);


// 计算三点O,A,B所夹的角度∠AOB
// 可以是三维点或者二维点
// O, A, B可以使行向量也可以是列向量
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

// 载入三维点云到vector<Point>
template<typename T>
std::vector<cv::Point3_<T>> loadPtCFromFile(std::string filepath){
	std::vector<cv::Point3_<T>> result;
	fstream f(filepath, ios_base::in);
	if (!f.is_open())
	{
		cout << "loadPtC 打开文件失败!" << endl;
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

// 【保存】点云到文件
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

// 保存二维点集到文件
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

// 【保存】点云到文件
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

// 空间均匀间隔采样点
std::vector<glm::vec3> sampleInSpace3D(float left, float right, float top, float bottom, float nearPlane, float farPlane,
	float step = 3.0);

// 去除不符合要求的点 
// 凡是pts中满足judgeOutliers的点都被踢掉
std::vector<glm::vec3> removeOutliers(const std::vector<glm::vec3> &pts,
	const std::function<bool(glm::vec3)> judgeOutliers);

// 给定一组点，给出点坐标值在x,y,z三个方向上的范围 包围盒
void boundingBox(glm::vec3 *pts, int cnt, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax);
void boundingBox(std::vector<glm::vec3> pts, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax);
cv::Vec6f boundingBox(std::vector<glm::vec3> pts);
cv::Vec4f boundingBox(std::vector<glm::vec2> pts);
cv::Vec6d boundingBox(std::vector<glm::dvec3> pts);

// 计算包围盒的半径
float calBoundingBoxRadius(cv::Vec6f box);			

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 原始坐标系
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
glm::vec3 transformCS(glm::vec3 pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);

// 坐标系变换的RT矩阵 新坐标系中的原点 各轴方向在老坐标系下的表示为 cen xdir ydir zdir
// 那么老坐标系下的点通过result的RT变换可以得到新坐标系下的坐标
glm::dmat4 transformCSRT(glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir, glm::dmat3 &R, glm::dvec3 &T);

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 原始坐标系
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
glm::dvec3 transformCS(glm::dvec3 pt, glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir);

// 返回pt在原始坐标系下的坐标
// 输入 当前点坐标 新坐标系下
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
glm::vec3 transformCSInverse(glm::vec3 pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
std::vector<glm::vec3> transformCS(std::vector<glm::vec3> pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
//		新坐标系 z正方向向量
std::vector<glm::dvec3> transformCS(std::vector<glm::dvec3> pt, glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir);

// 返回pt在原始坐标系下的坐标
// 输入 当前点坐标 新坐标系下
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
std::vector<glm::vec3> transformCSInverse(std::vector<glm::vec3> pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir);


// 生成球冠点云数据
std::vector<glm::vec3> generateSphericalCrownData(glm::vec3 cen, glm::vec3 north, float radius, float angle, 
	float noise = 0);

// 【新建】创建噪声平面
std::vector<glm::vec3> generateNoisePlane(const glm::vec3 &norm, const glm::vec3 &cen, const float &width,
	const float &height,
	const float &noiseScale,
	const float &sampleCnt = 20);

// angle 为角度制 空间圆
std::vector<glm::vec3> generateCircle3D(const glm::vec3 &cen, const glm::vec3 &norm, const float &radius, const int &pointCnt = 100,
	const float &angle = 360.0f);
std::vector<glm::dvec3> generateCircle3D(const glm::dvec3 &cen, const glm::dvec3 &norm, const double &radius, const int &pointCnt = 100,
	const double &angle = 360.0f);

// 计算球冠高度
// radius 是球半径
// diameterCrossSection 是球冠截面直径
float calcSphericalCapHeight(float radius, float diameterCrossSection);

// 计算球冠 球心角度
// radius 球半径
// diameterCrossSection 是球冠截面直径
float calcSphericalCapAngle(float radius, float diameterCrossSection);

// 空间射线和球面相交
bool intersect(glm::vec3 start, glm::vec3 dir, glm::vec3 sphereCen, float radius, float &t0, float &t1);
bool intersect(cv::Vec6f line, glm::vec3 sphereCen, float radius, float &t0, float &t1);


// l前三个数表示方向 后3个数表示射线上的点
bool intersectHalflineAndSphere(cv::Vec6d l, glm::dvec3 cen, double radius, std::vector<glm::dvec3> &result);

// 合并多个视场点云
cv::Mat mergeMultiViewPts(std::vector<std::vector<cv::Point3f>> input);

// 反选点集 返回未被选中的点坐标
std::vector<glm::vec3> inverseSelect(const std::vector<glm::vec3> &pts, const std::vector<int> &selectIds);

// 反选点集 返回被选中的点坐标
std::vector<glm::vec3> select(const std::vector<glm::vec3> &pts, const std::vector<int> &selectIds);

// 
std::vector<glm::dvec3> convert2dvec3s(std::vector<glm::vec3> pts);

// 沿某方向以一定步长复制点 限定size
std::vector<glm::vec3> copyPtsAlongDir(glm::vec3 pt, glm::vec3 dir,
	float step = 1.0, float size = 100.0, bool invertDir = false, bool bothDir = false);

void addNoiseToPtCloud(std::vector<glm::vec3> &pts, double noiseLevel);

// 寻找最临近点
int findNearestPt2D(glm::vec2 input, std::vector<glm::vec2> pts, double threshDist, double *minDist = NULL);
