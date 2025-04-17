#if defined USE_PCH
#include"stdafx.h"
#endif

#include"MyPCP.h"
//#include"MyRotate.h"

#define Min(A, B) ((A)<(B)?(A):(B))
#define Max(A, B) ((A)>(B)?(A):(B))


std::ostream& operator<<(std::ostream &out, Pt3D value)
{
	out << "x = " << value.x << "; y = " << value.y << "; z = " << value.z << "; flag = " << value.flag << std::endl;
	return out;
}

std::ostream &operator<<(std::ostream &out, CodeDist value)
{
	out << std::setw(3) << value.rank << " Dist[" << value.id1 << "]-[" << value.id2 << "] = " << std::setprecision(12) << value.dist;
	return out;
}

// 计算球面点 利用球参数方程 Tips OpenGL
glm::vec3 getSpherePointf(float u, float v, float radius)
{
	float x = sin(v)*cos(u);
	float y = sin(v)*sin(u);
	float z = cos(v);
	return glm::vec3(x, y, z)*radius;
}

glm::dvec3 getSpherePointd(double u, double v, double radius)
{
	double x = sin(v)*cos(u);
	double y = sin(v)*sin(u);
	double z = cos(v);
	return glm::dvec3(x, y, z)*radius;
}

// 从文件中载入点云
std::vector<glm::dvec3> loadPtCd_GL(const std::string &filepath)
{
	std::ifstream fPtC(filepath, std::ios::in);
	if (!fPtC)
	{
		return std::vector<glm::dvec3>();
	}
	std::vector<glm::dvec3> result;
	glm::dvec3 tmpPt;
	int tmp = 0;
	float value;
	while (fPtC >> value)
	{
		tmpPt[tmp] = value;
		tmp++;
		if (tmp == 3)
		{
			result.push_back(tmpPt);
			tmp = 0;
		}
	}
	fPtC.close();
	return result;
}

// 从文件中载入点云
std::vector<glm::vec2> loadPt2d_GL(const std::string &filepath)
{
	std::ifstream fPtC(filepath, std::ios::in);
	if (!fPtC)
	{
		return std::vector<glm::vec2>();
	}
	std::vector<glm::vec2> result;
	glm::vec2 tmpPt;
	int tmp = 0;
	float value;
	while (fPtC >> value)
	{
		tmpPt[tmp] = value;
		tmp++;
		if (tmp == 2)
		{
			result.push_back(tmpPt);
			tmp = 0;
		}
	}
	fPtC.close();
	return result;
}



// 【保存】点云到文件
bool savePtC(char* filepath, std::vector<Pt3D> src){
	std::ofstream r(filepath);
	for (int i = 0; i < (int)src.size(); i++){
		r << src[i].x << " " << src[i].y << " " << src[i].z << std::endl;
	}
	return true;
}

void addNoiseToPtCloud(std::vector<glm::vec3> &pts, double noiseLevel)
{
	for (int i = 0; i < pts.size(); i++)
	{
		pts[i].x += GaussRand(0, noiseLevel);
		pts[i].y += GaussRand(0, noiseLevel);
		pts[i].z += GaussRand(0, noiseLevel);
	}
}

// 将点云拆分成3个double数组
void splitXYZ(std::vector<Pt3D> input, double* x, double* y, double* z){
	for (int i = 0; i < (int)input.size(); i++){
		x[i] = input[i].x;
		y[i] = input[i].y;
		z[i] = input[i].z;
	}
}

// 将点云拆分成3个vector数组
void splitXYZ(std::vector<cv::Point3d> input, std::vector<double>& x, std::vector<double>& y, 
	std::vector<double>& z){
	x.clear();
	y.clear();
	z.clear();
	x.resize(input.size());
	y.resize(input.size());
	z.resize(input.size());
	std::vector<cv::Point3d>::iterator it_p = input.begin();
	int i = 0;
	while (it_p != input.end()){
		x[i] = it_p->x;
		y[i] = it_p->y;
		z[i] = it_p->z;
		i++;
		it_p++;
	}
}


// 读入空间三维点集 input 为黑底 上含白点
std::vector<Pt3D> getPt3DsFromImg(cv::Mat input){
	CV_Assert(input.channels() == 1);
	std::vector<Pt3D> result;
	uchar * pt = input.ptr<uchar>(0);
	for (int i = 0; i < input.rows; i++){
		for (int j = 0; j < input.cols; j++){
			if (*pt++)
				result.push_back(Pt3D((double)j, (double)i, 0, 0));
		}
	}
	return result;
}

// 在Img上绘制Pt3Ds
bool drawPt3DsToImg(cv::Mat& img, std::vector<Pt3D> pts){
	for (int i = 0; i < (int)pts.size(); i++){
		circle(img, cv::Point((int)pts[i].x, (int)pts[i].y), 3, cv::Scalar(0, 255, 255), 1, 8, 0);
		/*string id = int2string(i);
		putText(mat, id, Point(points[i].x + 10, points[i].y - 5), FONT_HERSHEY_COMPLEX,
		0.9, Scalar(155, 255, 255), 2);*/
	}
	return true;
}

// 寻找离质心最远的点
glm::vec3 farthestPoint(const std::vector<glm::vec3> &pts, const glm::vec3 &cen_/* = glm::vec3(0)*/)
{
	float dist = FLT_MIN;
	glm::vec3 farthest = pts[0];
	glm::vec3 cen = glm::vec3(0);
	if (cen_ != glm::vec3(0))
		cen = cen_;
	for (int i = 0; i < pts.size(); i++)
	{
		if (dist < glm::distance(pts[i], cen))
		{
			farthest = pts[i];
			dist = glm::distance(pts[i], cen);
		}
	}
	return farthest;
}

// 找到最大值点
int findMaxPt3D(std::vector<Pt3D> pts, std::string s){
	CV_Assert(s == "x" || s == "X" || s == "y" || s == "Y" || s == "z" || s == "Z");
	CV_Assert(pts.size() >= 1);
	int n = pts.size();
	int result = 0;
	if (s == "x" || s == "X"){
		double x0 = pts[0].x;
		for (int i = 0; i < n; i++){
			if (x0 < pts[i].x){
				x0 = pts[i].x;
				result = i;
			}
		}
	}
	else if(s == "y" || s == "Y"){
		double y0 = pts[0].y;
		for (int i = 0; i < n; i++){
			if (y0 < pts[i].y){
				y0 = pts[i].y;
				result = i;
			}
		}
	}
	else if (s == "z" || s == "Z"){
		double z0 = pts[0].z;
		for (int i = 0; i < n; i++){
			if (z0 < pts[i].z){
				z0 = pts[i].z;
				result = i;
			}
		}
	}
	return result;
}

// 找到最小值点
int findMinPt3D(std::vector<Pt3D> pts, std::string s){
	CV_Assert(s == "x" || s == "X" || s == "y" || s == "Y" || s == "z" || s == "Z");
	CV_Assert(pts.size() >= 1);
	int n = pts.size();
	int result = 0;
	if (s == "x" || s == "X"){
		double x0 = pts[0].x;
		for (int i = 0; i < n; i++){
			if (x0 > pts[i].x){
				x0 = pts[i].x;
				result = i;
			}
		}
	}
	else if (s == "y" || s == "Y"){
		double y0 = pts[0].y;
		for (int i = 0; i < n; i++){
			if (y0 > pts[i].y){
				y0 = pts[i].y;
				result = i;
			}
		}
	}
	else if (s == "z" || s == "Z"){
		double z0 = pts[0].z;
		for (int i = 0; i < n; i++){
			if (z0 > pts[i].z){
				z0 = pts[i].z;
				result = i;
			}
		}
	}
	return result;
}

// 求两点间距离
double Distance(Pt3D p1, Pt3D p2){
	return (sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z)));
}

// 求两点间距离
double Distance(cv::Point3d p1, cv::Point3d p2){
	return (sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z)));
}

// 寻找最临近点
int findNearestPt3D(std::vector<Pt3D> pts, int begin, double threshDist){
	if (begin >= pts.size() || begin < 0){
		std::cout << "findNearestPt3D 参数begin有误！！！" << std::endl;
		return -1;
	}
	int N = pts.size();
	double dist = threshDist;
	int result = begin;

	for (int i = 0; i < pts.size(); i++){
		if ((i+begin)%N != begin && pts[(begin+i)%N].flag == 0){
			double tmpDist = Distance(pts[(begin + i) % N], pts[begin]);
			//cout << tmpDist << endl;
			if (tmpDist < threshDist){
				if (dist > tmpDist){
					dist = tmpDist;
					result = (begin + i) % N;
					//cout << "result = " << result << endl;
				}
			}
		}
	}

	if (result == begin){
		std::cout << "未找到阈值范围内的最临近点！！！" << std::endl;
		return -1;
	}
	else {
		return result;
	}
}


/* ======================================================== */
/* 标记方法参见文献L. Thurfjell, E. Bengtsson, B. Nordin,   */
/* A new three dimensional connected components labeling    */
/* algorithm with simultaneous object feature               */
/* extraction capability,  CVGIP 54, 1992, 357-364.         */
/* ======================================================== */
int* ConstrainedComponentLabelingM(char *Image, int iRows, int iCols, int* NoOfLabels, int iDilateCount)// , int *Area
{
	int *A = new int[iRows*iCols];
	int Label = 0, Upper, Left, MinL, MaxL;
	int *Equiv = new int[(iRows - 1)*(iCols - 1)];//最大label数量
	int i, j, k, ii, jj;

	std::ofstream f3("E:\\Image.txt", std::ios::out);

	for (j = 0; j<iRows; j++)
	{
		for (i = 0; i<iCols; i++)
		{
			f3 << int(Image[i + j*iCols]) << "  ";

		}
		f3 << std::endl;
	}
	f3.close();


	//将Image膨胀生成A，膨胀次数由iDilateCount决定

	for (i = 0; i<iCols; i++)
	{
		for (j = 0; j<iRows; j++)
		{
			if (Image[i + j*iCols] == 0)
			{
				A[i + j*iCols] = 0;
			}
			else
			{

				for (ii = i - iDilateCount; ii <= i + iDilateCount; ii++)
				{
					for (jj = j - iDilateCount; jj <= j + iDilateCount; jj++)
					{
						if (((ii + jj*iCols) >= 0) && ((ii + jj*iCols)<iRows*iCols))
						{
							A[ii + jj*iCols] = 1;
						}
					}

				}

			}

		}

	}



	//将第一行置零
	for (i = 0; i<iCols; i++)
	{
		A[i] = 0;
	}
	//将第一列置零
	for (j = 0; j<iRows; j++)
	{
		A[j*iRows] = 0;
	}
	/* First pass */

	for (i = 1; i<iCols; i++)
	for (j = 1; j<iRows; j++)
	{
		if (A[i + j*iCols] == 0)//像素A[i][j]是否为零
			continue;
		Upper = A[i + (j - 1)*iCols];//Upper是A[i][j]上边的像素
		Left = A[i - 1 + j*iCols];//Left是A[i][j]左边的像素

		if ((Upper == 0) && (Left == 0))
		{
			Label++;
			//A[i][j]和Equiv[Label]的值都是Label
			A[i + j*iCols] = Equiv[Label] = Label;
			continue;
		}
		//如果上边＝0，则A[i][j]和左边的一样
		if (Upper == 0)
		{
			A[i + j*iCols] = Left;
			continue;
		}
		//如果左边＝0，则A[i][j]和上边的一样
		if (Left == 0)
		{
			A[i + j*iCols] = Upper;
			continue;
		}
		//如果上边＝左边，则A[i][j]和上边的一样
		if (Upper == Left)
		{
			A[i + j*iCols] = Upper;
			continue;
		}

		/* 纪录 label equivalence */
		MinL = Min(Upper, Left);
		MaxL = Max(Upper, Left);
		A[i + j*iCols] = MinL;
		while (Equiv[MaxL] != MaxL)
			MaxL = Equiv[MaxL];
		while (Equiv[MinL] != MinL)
			MinL = Equiv[MinL];
		if (MaxL >= MinL)
			Equiv[MaxL] = MinL;
		else
			Equiv[MinL] = MaxL;
	}

	/* 建立 equivalence  */
	//NoOfLabels是顺序排列的Equiv numble，如果两个区域相连，则记小的Equiv 数字
	(*NoOfLabels) = 0;//记录共有多少区域
	for (k = 1; k <= Label; k++)
	{
		if (Equiv[k] == k)
		{
			(*NoOfLabels)++;
			Equiv[k] = (*NoOfLabels);
		}
		else
			Equiv[k] = Equiv[Equiv[k]];
	}

	/* Second pass */

	//    for (k=1; k<=*NoOfLabels; k++) 
	// 		Area[k] = 0;

	//重新遍历

	for (i = 1; i<iCols; i++)
	for (j = 1; j<iRows; j++)
	{
		if (A[i + j*iCols] == 0)
			continue;
		//按照NoOfLabels的序号把A重新填写了一遍，相同区域的有同样的序号
		A[i + j*iCols] = Equiv[A[i + j*iCols]];
		//Area中存放的是每个区域有的元素个数
		//             Area[A[i+j*iCols]]++;
	}
	//将Image与膨胀后A中对应中元素置为序号
	for (k = 0; k<iCols*iRows; k++)
	{
		if (Image[k] != 0)
			Image[k] = A[k];
	}

	std::ofstream f4("E:\\ImageNew.txt", std::ios::out);

	for (j = 0; j<iRows; j++)
	{
		for (i = 0; i<iCols; i++)
		{
			f4 << int(A[i + j*iCols]) << "  ";

		}
		f4 << std::endl;
	}
	f4.close();
	return A;
}

// Point_World、Point_Workpiece 为3*4
void FindReferencePts(cv::Mat PointO_World, cv::Mat PointA_World, cv::Mat Vector_OB_World,
	cv::Mat &Point_World, cv::Mat &Point_Workpiece){
	double *temp_Point_O_World = PointO_World.ptr<double>(0);
	double *temp_Point_A_World = PointA_World.ptr<double>(0);
	double *temp_Point_OB_World = Vector_OB_World.ptr<double>(0);
	double *temp_Point_World = Point_World.ptr<double>(0);
	double *temp_Point_Workpiece = Point_Workpiece.ptr<double>(0);

	// X轴方向向量，O点为坐标原点，OA为x轴，为求解方便
	// 求OA
	cv::Mat Point_OA_World = PointA_World - PointO_World;

	// 去掉x轴与z轴不垂直的影响，以z轴方向为准，x轴与z轴垂直	
	cv::Mat OY_Mat = Vector_OB_World.cross(Point_OA_World);
	Point_OA_World = OY_Mat.cross(Vector_OB_World);
	double *temp_Point_OA_World = Point_OA_World.ptr<double>(0);

	// 距离，为单位化向量
	double tempDistanceA = norm(Point_OA_World);
	double tempDistanceB = norm(Point_OA_World);
	// Y轴方向向量 辅助，为求解方便
	cv::Mat Point_OY_World = Vector_OB_World.cross(Point_OA_World);
	double tempDistanceC = norm(Point_OY_World);
	double *temp_Point_OY_World = Point_OY_World.ptr<double>(0);

	int Count = 4;
	Point_Workpiece.col(0).setTo(cv::Scalar::all(0));

	// A
	Point_Workpiece.col(1) = (cv::Mat_<double>(3, 1) << tempDistanceA, 0, 0);

	Point_Workpiece.col(2) = (cv::Mat_<double>(3, 1) << 0, tempDistanceC, 0);

	Point_Workpiece.col(3) = (cv::Mat_<double>(3, 1) << 0, 0, tempDistanceB);

	// O
	*(temp_Point_World) = *(temp_Point_O_World + 0);
	*(temp_Point_World + Count) = *(temp_Point_O_World + 1);
	*(temp_Point_World + 2 * Count) = *(temp_Point_O_World + 2);

	// A
	*(temp_Point_World + 1) = *(temp_Point_O_World + 0) + (*(temp_Point_OA_World + 0));
	*(temp_Point_World + Count + 1) = *(temp_Point_O_World + 1) + (*(temp_Point_OA_World + 1));
	*(temp_Point_World + 2 * Count + 1) = *(temp_Point_O_World + 2) + (*(temp_Point_OA_World + 2));

	*(temp_Point_World + 2) = *(temp_Point_O_World + 0) + (*(temp_Point_OY_World + 0));
	*(temp_Point_World + Count + 2) = *(temp_Point_O_World + 1) + (*(temp_Point_OY_World + 1));
	*(temp_Point_World + 2 * Count + 2) = *(temp_Point_O_World + 2) + (*(temp_Point_OY_World + 2));

	// B
	*(temp_Point_World + 3) = *(temp_Point_O_World + 0) + (*(temp_Point_OB_World + 0));
	*(temp_Point_World + Count + 3) = *(temp_Point_O_World + 1) + (*(temp_Point_OB_World + 1));
	*(temp_Point_World + 2 * Count + 3) = *(temp_Point_O_World + 2) + (*(temp_Point_OB_World + 2));
}

// 绘制多边形
void DrawPoly(cv::Mat& img, std::vector<Pt3D> pts){
	cv::Point **p = (cv::Point**)malloc(sizeof(cv::Point*));
	p[0] = (cv::Point*)malloc(sizeof(cv::Point)*pts.size());
	for (int i = 0; i < pts.size(); i++){
		p[0][i].x = pts[i].x;
		p[0][i].y = pts[i].y;
	}

	const cv::Point* ppt[1] = { p[0] };
	int npt[] = { pts.size() };

	fillPoly(img, ppt, npt,1,  MC_YELLOW, 8);
}

// 计算序列点的平均间隔距离
double getAverDist(std::vector<Pt3D> pts){
	double sum = Distance(pts[0], pts[pts.size() - 1]);
	for (int i = 0; i < pts.size() - 1; i++){
		sum += Distance(pts[i], pts[i + 1]);
	}
	return (sum / (pts.size()));
}

// 绘制三次均匀B样条曲线
void DrawBSpline(cv::Mat& img, std::vector<Pt3D> pts, int begin, int end){
	if (end == -1)
		end = pts.size()-1;
	if (pts.size() >= 4){
		// 绘制控制点
		/*for (int i = 0; i < pts.size(); i++){
			drawEllipse(img, pts[i].x - 4, pts[i].y - 4, pts[i].x + 4, pts[i].y + 4);
		}*/
		// 绘制样条曲线
		for (int i = 0; i<pts.size() - 3; i++)
		{
			DrawBSpline(img, pts[i], pts[i + 1],
				pts[i + 2], pts[i + 3]);
		}
	}
}

// 绘制B样条曲线的时候需要的函数
void DaoShiTwo(Pt3D point1, Pt3D point2, Pt3D* pPoint)
{
	double x1 = (double)point1.x;
	double y1 = (double)point1.y;
	double x2 = (double)point2.x;
	double y2 = (double)point2.x;

	double s = sqrt(((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)) / (((x2 - x1) + (y2 - y1))*
		((x2 - x1) + (y2 - y1)) + (((y2 - y1) - (x2 - x1))*((y2 - y1) - (x2 - x1)))));
	double x = s*((y2 - y1) - (x2 - x1));
	double y = s*((y2 - y1) + (x2 - x1));
	pPoint->x = x;
	pPoint->y = y;
	pPoint->z = 0.0;

	return;
}

//以累加弦长为参数求抛物线端点导失
void DaoShiThree(Pt3D point1, Pt3D point2, Pt3D point3, Pt3D* pPoint)
{
	double x1 = (double)point1.x, y1 = (double)point1.y,
		x2 = (double)point2.x, y2 = (double)point2.y,
		x3 = (double)point3.x, y3 = (double)point3.y;

	double s1, s2, u, x, y;
	s1 = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
	s2 = sqrt((x2 - x3)*(x2 - x3) + (y2 - y3)*(y2 - y3));
	u = s1 / (s1 + s2);
	x = x1*(-u - 1) / u + x2*(-1) / (u*(u - 1)) + x3*(-u) / (1 - u);
	y = y1*(-u - 1) / u + y2*(-1) / (u*(u - 1)) + y3*(-u) / (1 - u);
	pPoint->x = x;
	pPoint->y = y;

	return;
}

//根据输入的型值点反算控制顶点
std::vector<Pt3D> GetControlPointByInputPoint(std::vector<Pt3D> input)
{
	int iNumber = input.size();
	int i;
	double iXiShu = 1;
	Pt3D DaoShiPoint, ControlPoint;

	std::vector<Pt3D> controlPoint;
	std::vector<double> m_vXiShu;

	
	if (iNumber == 1)
	{
		return controlPoint;
	}
	if (iNumber == 2)
	{
		DaoShiTwo(input[0], input[1], &DaoShiPoint);
		ControlPoint.x = 3 * input[0].x - DaoShiPoint.x;
		ControlPoint.y = 3 * input[0].y - DaoShiPoint.y;
		controlPoint.push_back(ControlPoint);
	}

	if (iNumber >= 3)
	{
		DaoShiThree(input[0], input[1], input[2], &DaoShiPoint);
		ControlPoint.x = 3 * input[0].x - DaoShiPoint.x;
		ControlPoint.y = 3 * input[0].y - DaoShiPoint.y;
		controlPoint.push_back(ControlPoint);
	}
	for (i = 0; i<iNumber; i++)
	{
		ControlPoint.x = 6 * input[i].x;
		ControlPoint.y = 6 * input[i].y;
		controlPoint.push_back(ControlPoint);
	}
	if (iNumber == 2)
	{
		DaoShiTwo(input[0], input[1], &DaoShiPoint);
		ControlPoint.x = 3 * input[1].x + DaoShiPoint.x;
		ControlPoint.y = 3 * input[1].y - DaoShiPoint.y;
		controlPoint.push_back(ControlPoint);
	}

	if (iNumber >= 3)
	{
		DaoShiThree(input[iNumber - 3], input[iNumber - 2],
			input[iNumber - 1], &DaoShiPoint);
		ControlPoint.x = 3 * input[iNumber - 1].x + DaoShiPoint.x;
		ControlPoint.y = 3 * input[iNumber - 1].y + DaoShiPoint.y;
		controlPoint.push_back(ControlPoint);
	}
	for (i = 0; i<controlPoint.size(); i++)
	{
		m_vXiShu.push_back(iXiShu);
	}
	m_vXiShu[0] = 2;
	for (i = 1; i<m_vXiShu.size() - 1; i++)
	{
		m_vXiShu[i] = 1 / (4 - m_vXiShu[i - 1]);
		controlPoint[i].x = (controlPoint[i].x - controlPoint[i - 1].x)*m_vXiShu[i];
		controlPoint[i].y = (controlPoint[i].y - controlPoint[i - 1].y)*m_vXiShu[i];
	}
	m_vXiShu[i] = 1 - 2 * m_vXiShu[i - 1];
	controlPoint[i].x = (controlPoint[i].x - 2 * controlPoint[i - 1].x) / m_vXiShu[i];
	controlPoint[i].y = (controlPoint[i].y - 2 * controlPoint[i - 1].y) / m_vXiShu[i];
	for (i = controlPoint.size() - 2; i >= 0; i--)
	{
		controlPoint[i].x = controlPoint[i].x - controlPoint[i + 1].x*m_vXiShu[i];
		controlPoint[i].y = controlPoint[i].y - controlPoint[i + 1].y*m_vXiShu[i];
	}

	return controlPoint;
}


//绘制拟合B样条曲线
void DrawBSpline(cv::Mat& img, Pt3D point1, Pt3D point2, Pt3D point3, Pt3D point4)
{
	double x1, y1, x2, y2, x3, y3, x4, y4;//传递4个点坐标值
	x1 = point1.x;
	y1 = point1.y;
	x2 = point2.x;
	y2 = point2.y;
	x3 = point3.x;
	y3 = point3.y;
	x4 = point4.x;
	y4 = point4.y;
	
	//line(img, Point(x1, y1), Point(x2, y2), MC_NAVY_BLUE, 2, 8, 0);
	//line(img, Point(x2, y2), Point(x3, y3), MC_NAVY_BLUE, 2, 8, 0);
	//line(img, Point(x3, y3), Point(x4, y4), MC_NAVY_BLUE, 2, 8, 0);

	double x = 0, y = 0, ax, ay, s1, s2, s3, u;
	double old_x, old_y;
//	pDC->SetPixel(x1, y1, RGB(255, 0, 0));
	ax = (x1 + 4 * x2 + x3) / 6.0;
	ay = (y1 + 4 * y2 + y3) / 6.0;
	for (u = 0; u <= 1; u = u + 0.001)
	{
		s1 = u;
		s2 = s1*s1;
		s3 = s1*s1*s1;
		x = ax + ((x3 - x1) / 2.0)*s1 + ((x1 - 2 * x2 + x3) / 2.0)*s2 + ((x4 - 3 * x3 + 3 * x2 - x1) / 6.0)*s3;
		y = ay + ((y3 - y1) / 2.0)*s1 + ((y1 - 2 * y2 + y3) / 2.0)*s2 + ((y4 - 3 * y3 + 3 * y2 - y1) / 6.0)*s3;
		if (u == 0)
		{
			old_x = (int)ax;
			old_y = (int)ay;
			//pDC->MoveTo((int)ax, (int)ay);
		}

		cv::line(img, cv::Point((int)old_x, (int)old_y), cv::Point((int)x, (int)y), MC_NAVY_BLUE, 1, 8, 0);
		old_x = x;
		old_y = y;
		//pDC->LineTo((int)x, (int)y);
		//if (u == 0 || u == 1)
		//{
		//	//pDC->Ellipse(x - 3, y - 3, x + 3, y + 3);
		//	drawEllipse(img, x - 3, y - 3, x + 3, y + 3);
		//}
	}
}

// 平移点云
std::vector<Pt3D> translatePts(std::vector<Pt3D> input, cv::Vec3d vec){
	std::vector<Pt3D>::iterator it = input.begin();
	std::vector<Pt3D> result(input.size());
	std::vector<Pt3D>::iterator it_r = result.begin();

	for (; it != input.end(); it++){
		it_r->x = it->x + vec.val[0];
		it_r->y = it->y + vec.val[1];
		it_r->z = it->z + vec.val[2];
		it_r++;
	}
	return result;
}

// 生成空间点集的距离矩阵 对称阵，对角元素都为0
cv::Mat DistMat(std::vector<cv::Point3d> input){
	int n = (int)input.size();
	cv::Mat result = cv::Mat::zeros(n, n, CV_64FC1);
	for (int i = 0; i < n; i++){
		for (int j = i+1; j < n; j++){
			result.at<double>(i, j) = Distance(input[i], input[j]);
			result.at<double>(j, i) = Distance(input[i], input[j]);
		}
	}
	return result;
}

// 生成空间点集的距离矩阵 以及【距离】-【点对】 键值对
// 以及【距离】-【点对】 键值对
std::vector<CodeDist> DistMap(std::vector<cv::Point3d> input, bool ifSort/* = false*/){
	std::vector<CodeDist> result;
	int n = (int)input.size();
	for (int i = 0; i < n; i++){
		for (int j = i + 1; j < n; j++){
			CodeDist temp;
			temp.id1 = i;
			temp.id2 = j;
			temp.dist = Distance(input[i], input[j]);
			result.push_back(temp);
		}
	}
	if (ifSort)
		sort(result.begin(), result.end(), comp_CodeDist());
	for (int i = 0; i < (int)result.size(); i++){
		result[i].rank = i;
	}
	return result;
}

// 点云的RT变换
// pts 3*N CV_64FC1
// R 3*3 旋转矩阵
// T 3*1 平移向量
// 返回值 3*N CV_64FC1
cv::Mat RT_Transform(cv::Mat pts, cv::Mat R, cv::Mat T){
	int N = pts.cols;			// 点的个数
	// 扩展T
	cv::Mat T_roate_Repeat = repeat(T, 1, N);
	cv::Mat result = R * pts + T_roate_Repeat;
	return result;
}


glm::vec3 RT_Transform(const glm::vec3 &pt, glm::mat3 rmat, glm::vec3 tvec)
{
	return (rmat*pt + tvec);
}

glm::vec3 RT_Transform_inv(const glm::vec3 &pt, glm::mat3 rmat, glm::vec3 tvec)
{
	return (glm::inverse(rmat)*(pt - tvec));
}

// 点云的RT逆变换
// pts 3*N CV_64FC1 是由结果经RT变换得到的
// R 3*3 旋转矩阵
// T 3*1 平移向量
// 返回值 3*N CV_64FC1 变换前的点云
cv::Mat RT_Transform_Inv(cv::Mat pts, cv::Mat R, cv::Mat T){
	int N = pts.cols;			// 点的个数
	// 扩展T
	cv::Mat T_roate_Repeat = repeat(T, 1, N);
	cv::Mat result = R.inv()*(pts - T_roate_Repeat);
	return result;
}

// Mat 转Point3d 
cv::Point3d Mat2Point3d(cv::Mat input){
	cv::Point3d result;
	result.x = input.ptr<double>(0)[0];
	result.y = input.ptr<double>(0)[1];
	result.z = input.ptr<double>(0)[2];
	return result;
}

// Point3d A 到Point3d B的向量 
cv::Vec3d pt2ptVec(cv::Point3d A, cv::Point3d B)
{
	return cv::Vec3d(B.x - A.x, B.y - A.y, B.z - A.z);
}

// 将Point3d 转为Mat 3*1 或者 4*1
// ifHC 是否转为齐次坐标 如果为true，则最后补个1
cv::Mat pt3d2mat(cv::Point3d pt, bool ifHC/* = false*/)
{
	if (ifHC)	
		return (cv::Mat_<double>(4, 1) << pt.x, pt.y, pt.z, 1.0);
	else
		return (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
}

// 将Point2d 转为Mat 2*1 或者 3*1
// ifHC 是否转为齐次坐标 如果为true，则最后补个1
cv::Mat pt2d2mat(cv::Point2d pt, int nch /*= 1*/, bool ifHC/* = false*/){
	if (nch == 1)
	{
		if (ifHC)
			return (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1.0);
		else
			return (cv::Mat_<double>(2, 1) << pt.x, pt.y);
	}
	else if (nch == 2)
	{
		if (ifHC)
			return (cv::Mat_<cv::Vec3d>(1, 1) << cv::Vec3d(pt.x, pt.y, 1.0));
		else
			return (cv::Mat_<cv::Vec2d>(1, 1) << cv::Vec2d(pt.x, pt.y));
	}	
}

// 将Mat转为Point2d
cv::Point2d mat2pt2d(cv::Mat pt, int nch/* = 1*/){
	if (nch == 1)
		return cv::Point2d(pt.ptr<double>(0)[0], pt.ptr<double>(0)[1]);
	else if (nch == 2)
	{
		cv::Vec2d v = pt.ptr<cv::Vec2d>(0)[0];
		return cv::Point2d(v[0], v[1]);
	}

}
// 将Mat转为Point3d
cv::Point3d mat2pt3d(cv::Mat pt){
	return cv::Point3d(pt.ptr<double>(0)[0], pt.ptr<double>(0)[1], pt.ptr<double>(0)[2]);
}

//// 将vector<Point3d>转换为cv::Mat  N*3
//cv::Mat pt3dVec2Mat(vector<cv::Point3d> pts){
//	cv::Mat b = Mat(pts);
//	//cout << "b = " << b << endl;
//	cv::Mat c = b.clone();
//	//= b.reshape(1, (int)pts.size());
//	//cout << "c = " << c << endl;
//	return c.reshape(1, (int)pts.size());
//}

glm::vec3 calcCenter(const std::vector<glm::vec3> &pts)
{
	glm::vec3 tmp(0);
	for (int i = 0; i < (int)pts.size(); i++)
		tmp += pts[i];
	return (tmp / (float)pts.size());
}

glm::dvec3 calcCenter(const std::vector<glm::dvec3> &pts)
{
	glm::dvec3 tmp(0);
	for (int i = 0; i < (int)pts.size(); i++)
		tmp += pts[i];
	return (tmp / (double)pts.size());
}

// 调整PCA的几个特征向量
void adjustPCAVecs(const std::vector<glm::vec3> &pts, cv::Mat &vecs)
{
	glm::vec3 cen = calcCenter(pts);
	glm::vec3 farthest = farthestPoint(pts);	// 找到离质心最远的点

	cv::Mat v1 = vecs.row(0).clone();
	glm::vec3 v1_gl(v1.ptr<double>(0)[0], v1.ptr<double>(0)[1], v1.ptr<double>(0)[2]);
	if (glm::dot(farthest - cen, v1_gl) < 0)	// 求质心到最远点向量 与 v1 的夹角， 如果夹角大于90度，则v1取反
		v1 = -v1;
	v1.copyTo(vecs(cv::Rect(0, 0, 3, 1)));

	cv::Mat v2 = vecs.row(1).clone();
	glm::vec3 v2_gl(v2.ptr<double>(0)[0], v2.ptr<double>(0)[1], v2.ptr<double>(0)[2]);
	if (glm::dot(farthest - cen, v2_gl) < 0)	// 求质心到最远点向量 与 v2 的夹角， 如果夹角大于90度，则v2取反
		v2 = -v2;
	v2.copyTo(vecs(cv::Rect(0, 1, 3, 1)));

	cv::Mat v3 = v1.cross(v2);
	v3.copyTo(vecs(cv::Rect(0, 2, 3, 1)));
}

// 去除不符合要求的点 
// 凡是pts中满足judgeOutliers的点都被踢掉
std::vector<glm::vec3> removeOutliers(const std::vector<glm::vec3> &pts,
	const std::function<bool(glm::vec3)> judgeOutliers)
{
	std::vector<glm::vec3> result;
	for (auto pt : pts)
		if (!judgeOutliers(pt))
			result.push_back(pt);
	return result;
}

// 给定一组点，给出点坐标值在x,y,z三个方向上的范围
void boundingBox(glm::vec3 *pts, int cnt, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax)
{
	*xmin = pts[0].x;
	*xmax = pts[0].x;
	*ymin = pts[0].y;
	*ymax = pts[0].y;
	*zmin = pts[0].z;
	*zmax = pts[0].z;
	for (int i = 0; i < cnt; i++)
	{
		if (*xmin > pts[i].x) *xmin = pts[i].x;
		if (*xmax < pts[i].x) *xmax = pts[i].x;
		if (*ymin > pts[i].y) *ymin = pts[i].y;
		if (*ymax < pts[i].y) *ymax = pts[i].y;
		if (*zmin > pts[i].z) *zmin = pts[i].z;
		if (*zmax < pts[i].z) *zmax = pts[i].z;
	}
}

void boundingBox(std::vector<glm::vec3> pts, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax)
{
	*xmin = pts[0].x;
	*xmax = pts[0].x;
	*ymin = pts[0].y;
	*ymax = pts[0].y;
	*zmin = pts[0].z;
	*zmax = pts[0].z;
	for (int i = 0; i < pts.size(); i++)
	{
		if (*xmin > pts[i].x) *xmin = pts[i].x;
		if (*xmax < pts[i].x) *xmax = pts[i].x;
		if (*ymin > pts[i].y) *ymin = pts[i].y;
		if (*ymax < pts[i].y) *ymax = pts[i].y;
		if (*zmin > pts[i].z) *zmin = pts[i].z;
		if (*zmax < pts[i].z) *zmax = pts[i].z;
	}
}

cv::Vec6f boundingBox(std::vector<glm::vec3> pts)
{
	cv::Vec6f result(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX);
	for (int i = 0; i < pts.size(); i++)
	{
		if (result[0] > pts[i].x) result[0] = pts[i].x;
		if (result[1] < pts[i].x) result[1] = pts[i].x;
		if (result[2] > pts[i].y) result[2] = pts[i].y;
		if (result[3] < pts[i].y) result[3] = pts[i].y;
		if (result[4] > pts[i].z) result[4] = pts[i].z;
		if (result[5] < pts[i].z) result[5] = pts[i].z;
	}
	return result;
}

cv::Vec6d boundingBox(std::vector<glm::dvec3> pts)
{
	cv::Vec6d result(DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX, DBL_MAX, -DBL_MAX);
	for (int i = 0; i < pts.size(); i++)
	{
		if (result[0] > pts[i].x) result[0] = pts[i].x;
		if (result[1] < pts[i].x) result[1] = pts[i].x;
		if (result[2] > pts[i].y) result[2] = pts[i].y;
		if (result[3] < pts[i].y) result[3] = pts[i].y;
		if (result[4] > pts[i].z) result[4] = pts[i].z;
		if (result[5] < pts[i].z) result[5] = pts[i].z;
	}
	return result;
}

cv::Vec4f boundingBox(std::vector<glm::vec2> pts)
{
	cv::Vec4f result(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX);
	for (int i = 0; i < pts.size(); i++)
	{
		if (result[0] > pts[i].x) result[0] = pts[i].x;
		if (result[1] < pts[i].x) result[1] = pts[i].x;
		if (result[2] > pts[i].y) result[2] = pts[i].y;
		if (result[3] < pts[i].y) result[3] = pts[i].y;
	}
	return result;
}

// std::vector<glm::vec3> 转Mat
cv::Mat pt3dVec2Mat(std::vector<glm::vec3> pts)
{
	cv::Mat r(pts.size(), 3, CV_32FC1);
	for (int i = 0; i < pts.size(); i++)
	{
		r.ptr<float>(i)[0] = pts[i].x;
		r.ptr<float>(i)[1] = pts[i].y;
		r.ptr<float>(i)[2] = pts[i].z;
	}
	return r;
}

// 返回不同坐标系下坐标的转换关系
// 输入 当前点坐标 
//		坐标系原点
//		x正方向向量
//		y正方向向量
glm::vec3 transformCS(glm::vec3 pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir)
{
	xdir = glm::normalize(xdir);
	ydir = glm::normalize(ydir);
	glm::vec3 zdir = glm::normalize(glm::cross(xdir, ydir));
	pt = pt - cen;
	return glm::vec3(
		glm::dot(pt, xdir),
		glm::dot(pt, ydir), 
		glm::dot(pt, zdir)
		);
}

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 原始坐标系
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
glm::dvec3 transformCS(glm::dvec3 pt, glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir)
{
	xdir = glm::normalize(xdir);
	ydir = glm::normalize(ydir);
	zdir = glm::normalize(zdir);

	pt = pt - cen;
	return glm::dvec3(
		glm::dot(pt, xdir),
		glm::dot(pt, ydir),
		glm::dot(pt, zdir)
		);
}

// 返回pt在原始坐标系下的坐标
// 输入 当前点坐标 新坐标系下
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
glm::vec3 transformCSInverse(glm::vec3 pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir)
{
	xdir = glm::normalize(xdir);
	ydir = glm::normalize(ydir);
	glm::vec3 zdir = glm::normalize(glm::cross(xdir, ydir));
	glm::mat3 trans;
	trans[0][0] = xdir.x;
	trans[1][0] = xdir.y;
	trans[2][0] = xdir.z;

	trans[0][1] = ydir.x;
	trans[1][1] = ydir.y;
	trans[2][1] = ydir.z;


	trans[0][2] = zdir.x;
	trans[1][2] = zdir.y;
	trans[2][2] = zdir.z;

	return (glm::inverse(trans)*pt + cen);
}

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
std::vector<glm::vec3> transformCS(std::vector<glm::vec3> pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir)
{
	std::vector<glm::vec3> result;
	for (auto p : pt)
		result.push_back(transformCS(p, cen, xdir, ydir));
	return result;	
}

// 返回pt在新坐标系下的坐标 
// 输入 当前点坐标 
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
//		新坐标系 z正方向向量
std::vector<glm::dvec3> transformCS(std::vector<glm::dvec3> pt, glm::dvec3 cen, glm::dvec3 xdir, glm::dvec3 ydir, glm::dvec3 zdir)
{
	std::vector<glm::dvec3> result;
	for (auto p : pt)
		result.push_back(transformCS(p, cen, xdir, ydir, zdir));
	return result;
}

// 返回pt在原始坐标系下的坐标
// 输入 当前点坐标 新坐标系下
//		新坐标系 原点
//		新坐标系 x正方向向量
//		新坐标系 y正方向向量
std::vector<glm::vec3> transformCSInverse(std::vector<glm::vec3> pt, glm::vec3 cen, glm::vec3 xdir, glm::vec3 ydir)
{
	std::vector<glm::vec3> result;
	for (auto p : pt)
		result.push_back(transformCSInverse(p, cen, xdir, ydir));
	return result;
}

// 坐标变换 先旋转后平移
void rotateAndMove(std::vector<glm::vec3> &pts, glm::mat3 RMat, glm::vec3 pos)
{
	for (auto &p:pts) p = RMat * p + pos;
}

void rotateAndMove(std::vector<glm::dvec3> &pts, glm::dmat3 RMat, glm::dvec3 pos)
{
	for (auto &p : pts) p = RMat * p + pos;
}

// 计算球冠高度
// radius 是球半径
// diameterCrossSection 是球冠截面直径
float calcSphericalCapHeight(float radius, float diameterCrossSection)
{
	return (radius - sqrt(radius * radius - diameterCrossSection * diameterCrossSection / 4.0f));
}

// 计算球冠 球心角度
// radius 球半径
// diameterCrossSection 是球冠截面直径
float calcSphericalCapAngle(float radius, float diameterCrossSection)
{
	return (2.0f * asin(diameterCrossSection / 2.0f / radius) * 180.0f / CV_PI);
}

// 空间射线和球面相交
bool intersect(glm::vec3 start, glm::vec3 dir, glm::vec3 sphereCen, float radius, float &t0, float &t1)
{
	glm::vec3 oc = sphereCen - start;
	float projoc = glm::dot(dir, oc);

	if (projoc < 0)
		return false;

	float oc2 = glm::dot(oc, oc);
	float distance2 = oc2 - projoc * projoc;
	//printf("distance2 = %f radius = %f\n", distance2, radius);

	if (distance2 > radius * radius)
		return false;

	float discriminant = radius * radius - distance2;
	if (discriminant < std::numeric_limits<float>::epsilon())
		t0 = t1 = projoc;
	else
	{
		discriminant = sqrt(discriminant);
		t0 = projoc - discriminant;
		t1 = projoc + discriminant;
		if (t0 < 0)
			t0 = t1;
	}
	return true;
}

bool intersect(cv::Vec6f line, glm::vec3 sphereCen, float radius, float &t0, float &t1)
{
	glm::vec3 start(line[0], line[1], line[2]);
	glm::vec3 dir(line[3], line[4], line[5]);
	return intersect(start, dir, sphereCen, radius, t0, t1);
}

// l前三个数表示方向 后3个数表示射线上的点
bool intersectHalflineAndSphere(cv::Vec6d l, glm::dvec3 cen, double radius, std::vector<glm::dvec3> &result)
{
	glm::dvec3 start(l[3], l[4], l[5]);
	glm::dvec3 dir = glm::normalize(glm::dvec3(l[0], l[1], l[2]));
	glm::dvec3 oc = cen - start;
	result.clear();
	double projoc = glm::dot(dir, oc);

	if (projoc < 0)
		return false;

	double oc2 = glm::dot(oc, oc);
	double distance2 = oc2 - projoc * projoc;

	if (distance2 > radius * radius)
		return false;

	double discriminant = radius * radius - distance2;
	if (discriminant < std::numeric_limits<double>::epsilon())
		result.push_back(start + projoc*dir);
	else
	{
		discriminant = sqrt(discriminant);
		double t0 = projoc - discriminant;
		double t1 = projoc + discriminant;
		if (t0 < 0)
		{
			result.push_back(start + t0*dir);
		}
		else
		{
			result.push_back(start + t0*dir);
			result.push_back(start + t1*dir);
		}
	}
	return true;
}

// 合并多个视场点云
cv::Mat mergeMultiViewPts(std::vector<std::vector<cv::Point3f>> input)
{
	cv::Mat result;
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[i].size(); j++)
		{
			cv::Mat tmp = (cv::Mat_<double>(1, 3) << input[i][j].x, input[i][j].y, input[i][j].z);
			result.push_back(tmp);
		}
	}
	return result;
}

bool savePtC(std::string filepath, std::vector<glm::vec3> pts)
{
	std::ofstream r(filepath);
	for (int i = 0; i < (int)pts.size(); i++)
	{
		r << std::setprecision(12) << pts[i].x << " " << pts[i].y << " " << pts[i].z << std::endl;
	}
	return true;
}

bool savePtC(std::string filepath, std::vector<glm::dvec3> pts)
{
	std::ofstream r(filepath);
	for (int i = 0; i < (int)pts.size(); i++)
	{
		r << std::setprecision(12) << pts[i].x << " " << pts[i].y << " " << pts[i].z << std::endl;
	}
	return true;
}

// 反选点集 返回未被选中的点坐标
std::vector<glm::vec3> inverseSelect(const std::vector<glm::vec3> &pts, const std::vector<int> &selectIds)
{
	std::vector<int> ids;
	ids.assign(selectIds.begin(), selectIds.end());
	sort(ids.begin(), ids.end());

	std::vector<glm::vec3> result;

	int k = 0;
	for (int i = 0; i < pts.size(); i++)
	{
		if (i == ids[k])
		{
			k++;
			continue;
		}
		result.push_back(pts[i]);
	}
	return result;
}

// 反选点集 返回被选中的点坐标
std::vector<glm::vec3> select(const std::vector<glm::vec3> &pts, const std::vector<int> &selectIds)
{
	std::vector<glm::vec3> result;
	for (int i = 0; i < selectIds.size(); i++)
	{
		result.push_back(pts[selectIds[i]]);
	}
	return result;
}

std::vector<glm::dvec3> convert2dvec3s(std::vector<glm::vec3> pts)
{
	std::vector<glm::dvec3> result;
	for (int i = 0; i < pts.size();i++)
		result.push_back(glm::dvec3((double)pts[i].x, (double)pts[i].y, (double)pts[i].z));
	return result;
}

// 沿某方向以一定步长复制点 限定size
std::vector<glm::vec3> copyPtsAlongDir(glm::vec3 pt, glm::vec3 dir, 
	float step /*= 1.0*/, float size /*= 100.0*/, bool invertDir /*= false*/, bool bothDir /*= false*/)
{
	std::vector<glm::vec3> result;
	dir = glm::normalize(dir);

	if ((invertDir == true && bothDir == false) || bothDir)
	{
		for (int i = size / step + 1;i>0; i--)
		{
			if (step*i <= size)
				result.push_back(pt - step*i*dir);
		}
	}
	result.push_back(pt);
	if (invertDir == false || bothDir)
	{
		for (int i = 1;; i++)
		{
			result.push_back(pt + step*i*dir);

			if (step *i > size)
				break;
		}
	}
	return result;
}

float calBoundingBoxRadius( cv::Vec6f box )
{
	float xspan = box[1] - box[0];
	float yspan = box[3] - box[2];
	float zspan = box[5] - box[4];
	return (0.5f * sqrt(xspan * xspan + yspan * yspan + zspan * zspan));
}

// 寻找最临近点
int findNearestPt2D(glm::vec2 input, std::vector<glm::vec2> pts, double threshDist, double *minDist/* = NULL*/)
{
	std::vector<double> dists(pts.size());
#pragma omp parallel for
	for (int i = 0; i < pts.size(); i++)
		dists[i] = glm::distance(input, pts[i]);

	int index = minIndexV<double>(dists);
	if (dists[index] > threshDist)
	{
		return -1;
	}
	else
	{
		if (minDist!=NULL)
			*minDist = dists[index];
		return index;
	}
}

#ifdef USE_MFC
// 【载入点云】 
std::vector<cv::Point3d> loadPtC(std::string filepath){
	std::vector<cv::Point3d> pts;
	FILE *fp;
	fopen_s(&fp, string2pChar(filepath), "r");

	cv::Point3d tmp;
	for (int i = 0; !feof(fp); i++)
	{
		fscanf_s(fp, "%lf %lf %lf\n", &tmp.x, &tmp.y, &tmp.z, sizeof(double)); // 循环读
		pts.push_back(tmp);
		//cout << tmp << endl;
	}
	fclose(fp);
	return pts;
}

// 【载入点云】 Data3D 为3*N 
bool loadPtCArchive(CString filepath, cv::Mat& Data3D){
	CFile saveF;
	if (FALSE == saveF.Open(filepath, CFile::modeRead))
	{
		AfxMessageBox(_T("读取点云文件失败！！！"));
		return FALSE;
	}
	CArchive ar(&saveF, CArchive::load);
	unsigned long count;
	ar >> count;
	if (Data3D.data){
		Data3D.release();
	}
	Data3D = cv::Mat::zeros(3, count, CV_64FC1);
	double *Data3DBuf = Data3D.ptr<double>(0);
	for (int i = 0; i < count; i++){
		ar >> *(Data3DBuf + i) >> *(Data3DBuf + i + count) >> *(Data3DBuf + i + count * 2);
	}
	ar.Close();
	saveF.Close();
	return true;
}

// 【保存】点云到文件 Archive
bool savePtCArchive(CString filepath, cv::Mat Data3D){
	CFile saveF;
	if (FALSE == saveF.Open(filepath, CFile::modeCreate | CFile::modeWrite))
	{
		AfxMessageBox(_T("保存点云文件失败！！！"));
		return FALSE;
	}
	CArchive ar(&saveF, CArchive::store);
	int count = Data3D.cols;
	ar << count;
	double *Data3DBuf = Data3D.ptr<double>(0);
	for (int i = 0; i < count; i++){
		ar << *(Data3DBuf + i) << *(Data3DBuf + i + count) << *(Data3DBuf + i + count * 2);
	}
	ar.Close();
	saveF.Close();
	return true;
}


// 【导出】导出点云 保存为asc文件
bool exportCloud(std::vector<cv::Point3d> cloud){

	AfxSetResourceHandle(GetModuleHandle(NULL));
	//printf("hh\n");

	CFileDialog dlg(FALSE,			// open
		_T("asc"),						// no default extension
		NULL,						// no initial file name
		OFN_OVERWRITEPROMPT | OFN_HIDEREADONLY,
		_T("ASC点云文件|*.asc"));


	if (dlg.DoModal() != IDOK){
		return false;
	}

	system("cls");

	CString filename = dlg.GetPathName();
	CString ext = dlg.GetFileExt();

	if (ext == _T("asc") || ext == ""){
		if (ext == "")
			filename += ".asc";
		//if (viewList.){
		FILE *fp = fopen(CString2pChar(filename), "w");
		if (fp){
			for (size_t i = 0; i <cloud.size(); i++){
				if (fabs(cloud[i].z) > 0.0f)
					fprintf(fp, "%f %f %f\n", cloud[i].x, cloud[i].y, cloud[i].z);
			}
			fclose(fp);
		}
		printf("点云保存完毕！\n");
		//}
	}
	return true;
}

// 【载入点云】 输出点云为3*N的Mat结构
bool loadPtC(CString filepath, cv::Mat& Data3D, bool ptInRow/* = false*/){
	FILE *fp = NULL;
	fopen_s(&fp, CString2pChar(filepath), "r");

	if (fp == NULL)
	{
		std::cout << "载入点云数据失败!" << std::endl;
		return false;
	}

	cv::Point3d tmp;
	std::vector<double> pv;
	for (int i = 0; !feof(fp); i++)
	{
		fscanf_s(fp, "%lf %lf %lf\n", &tmp.x, &tmp.y, &tmp.z, sizeof(double)); // 循环读
		pv.push_back(tmp.x);
		pv.push_back(tmp.y);
		pv.push_back(tmp.z);
	}
	//pv.data();
	cv::Mat dataMat(pv.size() / 3, 3, CV_64FC1, pv.data());
	if (ptInRow)
	{
		Data3D = dataMat.clone();
	}
	else
	{
		Data3D = dataMat.t();
	}
	fclose(fp);
	return true;
}

// 【载入点云】 
bool loadPtC(CString filepath, std::vector<Pt3D>& src){
	FILE *fp;
	fopen_s(&fp, CString2pChar(filepath), "r");

	Pt3D tmp;
	for (int i = 0; !feof(fp); i++)
	{
		fscanf_s(fp, "%lf %lf %lf\n", &tmp.x, &tmp.y, &tmp.z, sizeof(double)); // 循环读
		src.push_back(tmp);
	}
	fclose(fp);
	return true;
}



#endif
