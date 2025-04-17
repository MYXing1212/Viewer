#include"stdafx.h"
#include"MyPoint2D.h"

using namespace cv;

bool compare_x_Point(Point A, Point B) { return A.x < B.x; }
bool compare_y_Point(Point A, Point B) { return A.y < B.y; }

 //【最高点编号】 图像最高点对应y坐标最小
int idxMinY(vector<Point2f> points, bool* flag){
	if (!points.size()){
		//printf("【一组二维点求最高点编号】ERROR:二维点集为空!\n");
		return -1;
	}
	float min = 10000;
	int index = 0;
	vector<Point2f>::iterator itp = points.begin();
	for (int i = 0; itp < points.end(); itp++, i++){
		if (itp->y < min && !flag[i]){
			index = i;
			min = itp->y;
		}
	}
	return index;
}

//【最左点编号】 图像最左点对应x坐标最小
int idxMinX(vector<Point2f> points, bool* flag){
	if (!points.size()){
		return -1;
	}
	float min = 10000;
	int index = 0;
	vector<Point2f>::iterator itp = points.begin();
	for (int i = 0; itp < points.end(); itp++, i++){
		if (itp->x < min && !flag[i]){
			index = i;
			min = itp->x;
		}
	}
	return index;
}

// 两点间连线与水平向右向量的夹角 angle = 向量AB 与 水平向右向量 的夹角 返回值为角度制
// 返回值在0到360之间
// 应注意到图像坐标系y是向下增加的，所以当A = Point（1,1） B = Point(1 , 2) 时，结果为270度，而不是 90
double anglePitch(Point A, Point B){
	double tmp = (atan2((double)(A.y - B.y), (double)(B.x - A.x)) / CV_PI * 180.0);
	return (tmp<0)?tmp+360.0:tmp;
}

// 两点间斜率
double slope(double x1, double y1, double x2, double y2){
	return (y1 - y2) / (x1 - x2);
}

// 两点间斜率
double slope2(double x1, double y1, double x2, double y2){
	return (x1 - x2) / (y1 - y2);
}

// 两点间斜率
double slope(Point2f p1, Point2f p2){
	if (p2.x == p1.y)
	{
//		AfxMessageBox(_T("ERROR! 斜率为无穷大!"));
		return 0;
	}
	return ((p2.y - p1.y) / (p2.x - p1.x));
}

// 两点间【距离】
double dist(Point p1, Point p2){
	double x = (double)(p1.x - p2.x);
	double y = (double)(p1.y - p2.y);
	return sqrt(x*x + y*y);
}

// 两点间距离 【距离】
int distInt(Point p1, Point p2){
	int xi = abs(p1.x - p2.x);
	int yi = abs(p1.y - p2.y);
	if (xi > yi)
		return (xi - yi) * 10 + yi * 14;
	else
		return (yi - xi) * 10 + xi * 14;
}

// 两点间【距离】
float dist(Point2f p1, Point2f p2){
	float x =p1.x - p2.x;
	float y = p1.y - p2.y;
	return sqrt(x*x + y*y);
}

// 两点间【距离】
double dist(Point2d p1, Point2d p2){
	double x = p1.x - p2.x;
	float y = p1.y - p2.y;
	return sqrt(x*x + y*y);
}

// 两点间【距离】
double distance(double x1, double y1, double x2, double y2){
	double dx = x2 - x1;
	double dy = y2 - y1;
	return (sqrt(dx*dx + dy*dy));
}

// 【距离】点到 [点集]中的点 距离最小值
float minDist(Point p, Mat mask){
	float minD = 100000;
	uchar* g = mask.ptr<uchar>(0);
	for (int i = 0; i < mask.rows; i++){
		for (int j = 0; j < mask.cols; j++){
			if (*g){
				if (abs(p.x - j)>minD || abs(p.y - i)>minD){
					g++;
					continue;
				}
				if (minD > dist(p, Point(j, i)))
					minD = dist(p, Point(j, i));	
			}
			g++;
		}
	}
	//printf("minD = %lf\n", minD);
	return minD;
}

// A有m个点， B有n个点，求A中各点到B的最小距离，得到minDist[m],求minDist的最大值 并找到最大值对应A中那个点
float maxR(Mat A, Mat B, Point& cen, float scale){
	float maxd = 0;

	Mat scaleA, scaleB;
	resize(A, scaleA, Size(A.cols*scale, A.rows*scale));
	resize(B, scaleB, Size(B.cols*scale, B.rows*scale));
	threshold(scaleA, scaleA, 1, 255, THRESH_BINARY);
	threshold(scaleB, scaleB, 1, 255, THRESH_BINARY);
	
	//Mat tt;
	//bitwise_or(scaleA, scaleB, tt);
	//imshow("thin", tt);
	//imshow("edge", scaleB);
	
	uchar* pt = scaleA.ptr<uchar>(0);
	for (int i = 0; i < scaleA.rows; i++){
		for (int j = 0; j < scaleA.cols; j++){
			if (*pt++){
				float tmp = minDist(Point(j, i), scaleB);
				if (tmp > maxd){
					maxd = tmp;
					cen = Point(j, i);
				}
			}
		}		
	}
	cen = Point(cen.x / scale, cen.y / scale);
	return maxd/scale;
}

// 【计算】行数和列数 棋盘格角点
int calRowAndCol(vector<Point2f> points, vector<Point>& rc){
	// k为各点到最高点的斜率
	float* k = new float[points.size()];
	int* N = initAscendingVectorInt(points.size(), 0, 1);
	bool *flag = initVectorBool(points.size());		// 点是否已标号
	rc.resize(points.size());
	int rowNum = 0, colNum = 0;
	int codedNum = 0;	// 已编号点数 

	for (int i = 0; i < points.size(); i++){
		rc[i] = Point(0, 0);
	}
	

	// 给出行编号
	while (codedNum < points.size()){
		//while (rowNum<=8){
		// 【计算最高点】
		int topIndex = idxMinY(points, flag);
		//printf("top = %d\n", topIndex);
		Point2f top = points[topIndex];
		
		for (int i = 0; i < points.size(); i++){
			//if (points[i].x > top.x && !flag[i])
			if (points[i].x != top.x  && !flag[i]){			
				k[i] = ((points[i].y - top.y) / (points[i].x - top.x));
				if (k[i] < 0){
					if (k[i]>-0.1)
						k[i] = -k[i];
					else
						k[i] = 100000;
				}
				
			}
			else
				k[i] = 100000;
		}
		N = initAscendingVectorInt(points.size(), 0, 1);
		for (int i = 0; i < points.size() - 1; i++){
			for (int j = 0; j < points.size() - i - 1; j++){
				if (k[j] > k[j + 1]){
					swap2float(k[j], k[j + 1]);
					swap2int(N[j], N[j + 1]);
				}
			}
		}
		//if (rowNum == 1)
		//	printVector(k, points.size());

		int ps = numGrp1st(k, points.size());
		//printf("ps = %d\n", ps);
		rc[topIndex].x = rowNum+1;
		flag[topIndex] = true;
		for (int i = 0; i < ps; i++){
			rc[N[i]].x = rowNum+1;
			flag[N[i]] = true;
		}
		codedNum += ps + 1;
		rowNum++;
		//printf("rowNum = %d\n", rowNum);
	}

	codedNum = 0;
	flag = initVectorBool(points.size());

	// 给出列编号
	while (codedNum < points.size()){
	//	while (colNum<1){
		// 【计算最左点】
		int leftIndex = idxMinX(points, flag);
		printf("top = %d\n", leftIndex);
		Point2f left = points[leftIndex];

		for (int i = 0; i < points.size(); i++){
			//if (points[i].x > top.x && !flag[i])
			if (points[i].y != left.y  && !flag[i]){
				k[i] = ((points[i].x - left.x) / (points[i].y - left.y));
				if (k[i] > 0){
					if (k[i]<0.15)
						k[i] = -k[i];
					else
						k[i] = -100000;
				}

			}
			else
				k[i] = -100000;
		}
		//printVector(k, points.size());
		N = initAscendingVectorInt(points.size(), 0, 1);
		for (int i = 0; i < points.size() - 1; i++){
			for (int j = 0; j < points.size() - i - 1; j++){
				if (k[j] < k[j + 1]){
					swap2float(k[j], k[j + 1]);
					swap2int(N[j], N[j + 1]);
				}
			}
		}
		oppositeNumVec(k, points.size());
		
		int ps = numGrp1st(k, points.size());
		//printf("ps = %d\n", ps);
		rc[leftIndex].y = colNum + 1;
		flag[leftIndex] = true;
		for (int i = 0; i < ps; i++){
			rc[N[i]].y = colNum + 1;
			flag[N[i]] = true;
		}
		codedNum += ps + 1;
		colNum++;
		//printf("colNum = %d\n", colNum);
	}

	return colNum;
}



// 【输出】二维点
void printPt2f(Point2f point){
	printf("Point2f(%.6f, %.6f)\n", point.x, point.y);
}

// 【输出】二维点
void printPt2d(Point2d point){
	printf("Point2d(%.6lf, %.6lf)\n", point.x, point.y);
}

// 【输出】二维点集
void printPts2f(vector<Point2f> points){
	printf("vector<Point2f> = [\n");
	for (int i = 0; i < points.size(); i++){
		printf("\t");
		printPt2f(points[i]);
	}
}

// 【输出】二维点集
void printPts2d(vector<Point2d> points){
	printf("vector<Point2d> = [\n");
	for (int i = 0; i < points.size(); i++){
		printf("\t");
		printPt2d(points[i]);
	}
}

// 计算欧式距离
float calcEuclideanDistance(int x1, int y1, int x2, int y2){
	return sqrt(float((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}

// 计算棋盘距离
int calcChessboardDistance(int x1, int y1, int x2, int y2){
	return max(abs(x1 - x2), abs(y1 - y2));
}

// 计算街区距离
int calcBlockDistance(int x1, int y1, int x2, int y2){
	return abs(x1 - x2) + abs(y1 - y2);
}

// 生成一个单位圆的采样圆点阵 默认为单位圆
vector<Point2d> generateSamplingCirclePts(int Num, Point2d cen, double radius){
	vector<Point2d> result(Num);

	for (int i = 0; i < Num; i++){
		double x = cos((radius * i / (double)Num) * 2 * CV_PI) + cen.x;
		double y = sin((radius * i / (double)Num) * 2 * CV_PI) + cen.y;
		result.push_back(Point2d(x,y));
	}
	return result;
}

// 【循环偏移】二维点集 offset<0 左移  offset>0右移
void recycleMove(vector<Point>& pts, int offset){
	if (offset >0){
		vector<Point> tmp;
		for (int i = (int)pts.size() - offset; i < pts.size(); i++)
			tmp.push_back(pts[i]);
		for (int i = (int)pts.size()-1; i >= (int)offset; i--){
			pts[i] = pts[i - offset];
		}
		for (int i = 0; i < offset; i++){
			pts[i] = tmp[i];
			//tmp.pop_back();
		}
	}
	else if(offset < 0 ){
		vector<Point> tmp;
		for (int i = 0; i < (-offset); i++)
			tmp.push_back(pts[i]);

		for (int i = 0; i <(int)pts.size() + offset; i++){
			pts[i] = pts[i - offset];
		}

		for (int i = (int)pts.size()+offset,j= 0; i < (int)pts.size(); i++){
			pts[i] = tmp[j++];
		}

		cout << endl;
	}
}

// 点集按y坐标进行排序
void sortByY(vector<Point>& input){
	sort(input.begin(), input.end(), comp_Point_Y());
}

// 点集按x坐标进行排序
void sortByX(vector<Point>& input){
	sort(input.begin(), input.end(), comp_Point_X());
}

// 点集 x坐标最小值所在位置
int idxMinX(vector<Point> input){
	if (input.size() == 0)
		return -1;
	return (int)(min_element(input.begin(), input.end(), compare_x_Point) - input.begin());
}

// 点集 x坐标最大值所在位置
int idxMaxX(vector<Point> input){
	if (input.size() == 0)
		return -1;
	return (int)(max_element(input.begin(), input.end(), compare_x_Point) - input.begin());
}

// 点集 y坐标最小值所在位置
int idxMinY(vector<Point> input){
	if (input.size() == 0)
		return -1;
	return (int)(min_element(input.begin(), input.end(), compare_y_Point) - input.begin());
}

// 点集 y坐标最大值所在位置
int idxMaxY(vector<Point> input){
	if (input.size() == 0)
		return -1;
	return (int)(max_element(input.begin(), input.end(), compare_y_Point) - input.begin());
}

// 拆入排序
template<class T, class Pr>
void insert_sort(vector<T> &vec, int l, int r, Pr pred){
	int i, j;
	for (i = l + 1; i <= r; i++){
		T tmp = vec[i];
		for (j = i - 1; j >= l && pred(tmp, vec[j]); j--)
			vec[j + 1] = vec[j];
		vec[j + 1] = tmp;
	}
}

// 找到key所在的位置
template<class T>
int get_position(vector<T> &vec, int l, int r, T key){
	for (int i = l; i <= r; i++){
		if (key == vec[i])
			return i;
	}
	return -1;
}

// 按第一个元素对vec进行划分
template<class T, class Pr>
int partition(vector<T>&vec, int l, int r, Pr pred){
	int i, j;
	for (i = l + 1, j = l; i <= r; i++){
		if (pred(vec[i], vec[l])){
			++j;
			swap(vec[i], vec[j]);
		}
	}
	swap(vec[j], vec[l]);
	return j;
}

// 顺序统计得到第k个元素的值
template<class T, class Pr>
T select(vector<T> &vec, int l, int r, int k, Pr pred){
	int n = r - l + 1;
	if (n == 1){
		if (k != 0)
			printf("Out of Boundary!\n");
		return vec[l];
	}
	// 找中位数的中位数作为分割点
	int cnt = n / 5;
	int tcnt = (n + 4) / 5;
	int rem = n % 5;
	vector<T> group(tcnt);
	int i, j;
	for (i = 0, j = l; i < cnt; i++, j += 5){
		insert_sort(vec, j, j + 4, pred);
		group[i] = vec[j + 2];
	}
	if (rem){
		insert_sort(vec, j, j + rem - 1, pred);
		group[i] = vec[j + (rem - 1) / 2];
	}
	T key = select(group, 0, tcnt - 1, (tcnt - 1) / 2, pred);
	// 找到分割点的位置
	int key_pos = get_position(vec, l, r, key);
	swap(vec[key_pos], vec[l]);
	// 用分割点对数组进行花粉，小的在左边，大的在右边
	int pos = partition(vec, l, r, pred);
	int x = pos - l;
	if (x == k) return key;
	else if (x < k)
		return select(vec, pos + 1, r, k - x - 1, pred);
	else
		return select(vec, l, pos - 1, k, pred);
}

// 寻找二维平面上的最近点对
double minDifferent(vector<Point> p, int l, int r, vector<Point>& result){
	// 按中位数进行划分后的子区域的元素个数都会减小到2或3，不会再到1
	if ((r - l + 1) == 2){
		result[0] = p[l];
		result[1] = p[r];
		if (compare_x_Point(p[r], p[l])) swap(p[l], p[r]);
		return dist(p[l], p[r]);
	}
	if ((r - l + 1) == 3){
		insert_sort(p, l, r, compare_x_Point);
		double tmp1 = dist(p[l], p[l + 1]);
		double tmp2 = dist(p[l + 1], p[l + 2]);
		double ret = min(tmp1, tmp2);
		if (tmp1 == ret){
			result[0] = p[l];
			result[1] = p[l + 1];
		}
		else{
			result[0] = p[l + 1];
			result[1] = p[l + 2];
		}
		return ret;
	}
	// 大于3个点的情况
	int mid = (r + 1) >> 1;
	Point median = select(p, l, r, mid - l, compare_x_Point);
	vector<Point> res1(2), res2(2);
	double min_l = minDifferent(p, l, mid, res1);
	double min_r = minDifferent(p, mid + 1, r, res2);
	double minum = min(min_l, min_r);
	if (minum == min_l){
		result[0] = res1[0];
		result[1] = res1[1];
	}
	else{
		result[0] = res2[0];
		result[1] = res2[1];
	}
	// 对[p[mind+1]-minum, p[mid]+minum]的带状区域按y排序
	vector<Point> yvec;
	for (int i = mid + 1; i <= r; i++){
		if (p[i].x - p[mid].x < minum)
			yvec.push_back(Point(p[i]));
	}
	for (int i = mid; i >= l; i--){
		if (p[mid + 1].x - p[i].x < minum)
			yvec.push_back(Point(p[i]));
	}
	sort(yvec.begin(), yvec.end(), compare_y_Point);
	for (int i = 0; i < (int)yvec.size(); i++){
		// 至多只有与其后最多7个点的距离会小于minum
		for (int j = i + 1; j < (int)yvec.size() && yvec[j].y - yvec[i].y < minum && j <= i + 7; j++){
			double delta = dist(yvec[i], yvec[j]);
			if (delta < minum){
				minum = delta;
				result[0] = yvec[i];
				result[1] = yvec[j];
			}
		}
	}
	return minum;
}

// 在点集中查找离目标点最临近的点 返回最临近点的编号 和最临近距离
int findNearestPoint(Point target, vector<Point2f> pts, double* minDistance){
	if (pts.size() == 0)
		return -1;	
	
	cv::flann::KDTreeIndexParams indexParams;
	cv::flann::Index kdtree(cv::Mat(pts).reshape(1), indexParams);

	Point search;

	vector<float> query;
	query.push_back(target.x);
	query.push_back(target.y);

	int k = 4;	// number of nearest neighbors 
	std::vector<int> indices(k);
	std::vector<float> dists(k);

	kdtree.knnSearch(query, indices, dists, k, cv::flann::SearchParams(64));

	*minDistance = dists[0];
	return indices[0];
}

// 平面点平移
Point translatePoint(Point input, int offsetx, int offsety){
	return Point(input.x + offsetx, input.y + offsety);
}

// 平面点旋转 theta为角度制
Point rotatePoint(Point input, double theta){
	double angle = theta * CV_PI / 180.0;
	double cosTheta = cos(angle);
	double sinTheta = sin(angle);
	return Point((int)(cosTheta * input.x + sinTheta * input.y), (int)(-sinTheta*input.x + cosTheta * input.y));
}

// 平面点旋转 R为2*2旋转矩阵
vector<Point> transformPoints(vector<Point> input, Mat R, Mat T){
	vector<Point> output;

	float r00 = R.at<float>(0, 0); float r01 = R.at<float>(0, 1);
	float r10 = R.at<float>(1, 0); float r11 = R.at<float>(1, 1);
	float t0 = T.ptr<float>(0)[0]; float t1 = T.ptr<float>(0)[1];

	vector<Point>::iterator it_p = input.begin();
	for (; it_p != input.end(); it_p++){
		int x = (int)(r00*it_p->x + r01*it_p->y + t0);
		int y = (int)(r10*it_p->x + r11*it_p->y + t1);
		//query[1] = r10*T[idx * 2 + 0] + r11*T[idx * 2 + 1] + t1;
		output.push_back(Point(x,y));
	}

	//output = translatePoints(output, (int)cen.x, (int)cen.y);
	return output;
}

// 平面点集平移
vector<Point> translatePoints(vector<Point> input, int offsetx, int offsety){
	vector<Point> output;

	vector<Point>::iterator it_p = input.begin();
	for (; it_p != input.end(); it_p++){
		output.push_back(translatePoint(*it_p, offsetx, offsety));
	}
	return output;
}

// 计算轮廓重心
Point2f calContourBarycenter(vector<Point> contour){
	Moments mu = moments(contour, false);
	return Point2f((mu.m10 / mu.m00), (mu.m01 / mu.m00));
}

// 平面点集平移 theta为角度制
vector<Point> rotatePoints(vector<Point> input, double theta){
	Point2f cen = calContourBarycenter(input);
	vector<Point> tmp = translatePoints(input, -(int)cen.x, -(int)cen.y);
	
	vector<Point> output;

	vector<Point>::iterator it_p = tmp.begin();
	for (; it_p != tmp.end(); it_p++){
		output.push_back(rotatePoint(*it_p, theta));
	}

	output = translatePoints(output, (int)cen.x, (int)cen.y);
	return output;
}

// 将二维点集分解为两个向量 第一个向量存横坐标,第二个向量存纵坐标
vector<vector<double>> split(vector<Point> src){
	vector<vector<double>> r(2);
	vector<Point>::iterator it = src.begin();
	for (; it != src.end(); it++){
		r[0].push_back(it->x);
		r[1].push_back(it->y);
	}
	return r;
}

// 将二维点集分解为两个向量 第一个向量存横坐标,第二个向量存纵坐标
vector<vector<double>> split(vector<Point2d> src){
	vector<vector<double>> r(2);
	vector<Point2d>::iterator it = src.begin();
	for (; it != src.end(); it++){
		r[0].push_back(it->x);
		r[1].push_back(it->y);
	}
	return r;
}

// 将两个向量合成二维点集 第一个向量存横坐标,第二个向量存纵坐标
vector<Point2d> merge2Vec(vector<double> A, vector<double> B){
	vector<Point2d> r;
	CV_Assert(A.size() == B.size());
	vector<double>::iterator it_x = A.begin();
	vector<double>::iterator it_y = B.begin();
	for (; it_x != A.end(); it_x++, it_y++){
		r.push_back(Point2d(*it_x, *it_y));
	}
	return r;
}

// 对轮廓计算角度距离图 角度距离图定义
// 对于一个图形，求得图形中心，计算图形外轮廓上任一点与方向向右的水平线的夹角，
// 计算该点与点云中心的距离，逆时针扫一周，以角度为自变量，以两点距离为因变量(如果同一个角度存在多个点，则取最远点计算距离)，得到一个角度距离图。
// 参数deltaTheta应满足  360/deltaTheta 为一个整数
vector<double> calContourAngleDistMap(vector<Point> contour, int count){
	Point2f b = calContourBarycenter(contour);
	Point cen = Point((int)b.x, (int)b.y);

	double deltaTheta = 360.0 / count;

	vector<double> result = initVectord(count);
	for (int i = 0; i < (int)contour.size(); i++){
		int tmpAngle = (int)anglePitch(cen, contour[i]);
		double tmpDist = dist(cen, contour[i]);
		int tmpIndex = (int)(tmpAngle / deltaTheta);
		if (tmpDist > result[tmpIndex]){
			result[tmpIndex] = tmpDist;
		}
	}
	return result;
}

// 二维点集的粗拼接 主要返回旋转角度theta 和 平移向量   返回平移和旋转后的Pts2
// Pts2经过平移T和旋转theta可以得到Pts1
vector<Point> coarseRegistration(vector<Point> Pts1, vector<Point> Pts2, Vec2d& T, double &theta){
	vector<double> t1 = calContourAngleDistMap(Pts1);

	double minD = DBL_MAX;
	int index = -1;

	for (int i = 0; i < 3600; i++){
		vector<Point> tps = rotatePoints(Pts2, i/10.0);
		vector<double> t2 = calContourAngleDistMap(tps);
		double tmpDist = dist2Vec(t1, t2);
		if (minD > tmpDist){
			index = i;
			minD = tmpDist;
		}
	}

	Point2f cen1 = calContourBarycenter(Pts1);
	Point2f cen2 = calContourBarycenter(Pts2);

	theta = index/10.0;
	T[0] = cen1.x - cen2.x;
	T[1] = cen1.y - cen2.y;

	return translatePoints(rotatePoints(Pts2, theta), T[0], T[1]);
}

// 精拼接 
// Pts2经过平移T和旋转theta可以得到Pts1
vector<Point> preciseRegistration(vector<Point> Pts1, vector<Point> Pts2, Vec2d& T, double &theta){
	vector<Point> nearestPts;
	
	// 先把Pts2转Point2f
	vector<Point2f> f2 = convert2Point2f(Pts2);

	// 对f1中的每个点，在f2中找到距离最近点 
	cv::flann::KDTreeIndexParams indexParams;
	cv::flann::Index kdtree(cv::Mat(f2).reshape(1), indexParams);
	//kdtree.build()

	vector<Point>::iterator it_pts1 = Pts1.begin();
	vector<float> query;
	for (; it_pts1 != Pts1.end(); it_pts1++){
		query.clear();
		query.push_back((float)(it_pts1->x));
		query.push_back((float)(it_pts1->y));
		
		int k = 4;	// number of nearest neighbors 
		std::vector<int> indices(k);
		std::vector<float> dists(k);

		kdtree.knnSearch(query, indices, dists, k, cv::flann::SearchParams(64));
		nearestPts.push_back(Pts2[indices[0]]);
	}

	Mat img = Mat::zeros(500, 500, CV_8UC3);
	drawContour(img, nearestPts, MC_RED, 2);
	drawContour(img, Pts1, MC_BLUE, 2);
	imwrite("w.bmp", img);
	//imshow("i", img);

	vector<vector<double>> Data1 = split(Pts1);
	vector<vector<double>> Data2 = split(nearestPts);

	//printVector(Data2[0]);

	int n = Pts1.size();

	double X = meanV(Data1[0]);
	double Y = meanV(Data1[1]);
	double X_ = meanV(Data2[0]);
	double Y_ = meanV(Data2[1]);

	double sum1 = 0, sum2 = 0;
	for (int i = 0; i < n; i++){
		sum1 += Data1[0][i] * Data2[1][i] - Data2[0][i] * Data1[1][i];
		sum2 += Data2[0][i] * Data1[0][i] + Data2[1][i] * Data1[1][i];
	}

	theta = atan2(n*(X_*Y - X*Y_) + sum1, n*(X*X_ + Y*Y_) - sum2);
	//T[0] = (sumV(Data2[0]) - cos(theta) * sumV(Data1[0]) - sin(theta)*sumV(Data1[1])) / n;
	//T[1] = (sumV(Data2[1]) + sin(theta) * sumV(Data1[0]) - cos(theta)*sumV(Data1[1])) / n;
	T[0] = 0;
	T[1] = 0;
	
	theta = theta / CV_PI*180.0;

	cout << theta << endl;
	cout << "T = " << T << endl;
	return rotatePoints(translatePoints(Pts2, T[0], T[1]), theta);
}

// Point点集转换为Point2f点集
vector<Point2f> convert2Point2f(vector<Point> input){
	vector<Point>::iterator it = input.begin();
	vector<Point2f> result;
	for (; it != input.end(); it++){
		result.push_back(Point2f((float)(it->x), (float)(it->y)));
	}
	return result;
}

// Point2d点集转换为Point点集
vector<cv::Point> convert2Point2d(vector<cv::Point2d> input){
	vector<Point2d>::iterator it = input.begin();
	vector<Point> result;
	for (; it != input.end(); it++){
		result.push_back(Point((int)(it->x), (int)(it->y)));
	}
	return result;
}

// 20170616
// Point点集转换为Point2d点集
vector<Point2d> convert2Point2d(vector<Point> input){
	vector<Point>::iterator it = input.begin();
	vector<Point2d> result;
	for (; it != input.end(); it++){
		result.push_back(Point2d((double)(it->x), (double)(it->y)));
	}
	return result;
}

// Point点集转换为 Mat N行2列 double型
Mat convert2Mat2d(vector<Point> input){
	vector<Point>::iterator it = input.begin();
	Mat result = Mat::zeros(input.size(), 2, CV_64FC1);
	double *data = result.ptr<double>(0);
	for (; it != input.end(); it++){
		*data++ = (double)(it->x);
		*data++ = (double)(it->y);
	}
	return result;
}

// Mat N行2列转vector<Point>
vector<Point> convert2VecPt(Mat input){
	vector<Point> result;
	
	for (int i = 0; i < input.rows; i++){
		float *data = input.ptr<float>(i);
		result.push_back(Point((int)(data[0]), (int)(data[1])));
		data++;
	}
	return result;
}

// Point点集转换为 Mat N行2列 float型
Mat convert2Mat2f(vector<Point> input){
	vector<Point>::iterator it = input.begin();
	Mat result = Mat::zeros(input.size(), 2, CV_32FC1);
	float *data = result.ptr<float>(0);
	for (; it != input.end(); it++){
		*data++ = (float)(it->x);
		*data++ = (float)(it->y);
	}
	return result;
}

// Point点集转换为Point3d点集 二维点集转换为三维点Mat 第三维z默认为0 补0
Mat convert2Mat3d(vector<Point> input, double z/* = 0*/){
	vector<Point>::iterator it = input.begin();
	Mat result = Mat::zeros(input.size(), 3, CV_64FC1);
	double *data = result.ptr<double>(0);
	for (; it != input.end(); it++){
		*data++ = (double)(it->x);
		*data++ = (double)(it->y);
		*data++ = z;
	}
	return result;
}

// 绘制轮廓
void drawContour(Mat& input, vector<Point> contour, Scalar color, int thickness /* = 1*/)
{
	vector<vector<Point>> t;
	t.push_back(contour);
	drawContours(input, t, 0, color, thickness, 8);
}

// 二维点集归一化 默认minX = 0; maxX = 1; minY = 0; maxY = 1
vector<Point2d> normalizePts(vector<Point2d> input, double minX /* = 0*/, double maxX /*= 1.0*/,
	double minY /* = 0*/, double maxY /* = 1.0*/){
	vector<vector<double>> val = split(input);
	vector<double> newx = normalizeV(val[0], minX, maxX);
	vector<double> newy = normalizeV(val[1], minY, maxY);
	return merge2Vec(newx, newy);
}

// 两点决定一个向量
// 输入	Point2d A
//		Point2d B
// 输出  Vec2d AB
Vec2d vecA2B(Point2d A, Point2d B){
	return Vec2d(B.x - A.x, B.y - A.y);
}

// 全局几何结构特征差异 参见论文《基于混合特征的非刚性点阵配准算法》汤昊林
// 输入:		vector<Point2d> A
//			vector<Point2d> B
// 输出:		Mat G  G的通道数为2 A.size() = n, B.size() = m, 则G为n行m列 
Mat globalGeoFeatureDiff(vector<Point2d> A, vector<Point2d> B){
	int n = (int)A.size();
	int m = (int)B.size();
	vector<Point2d> normalized_A = normalizePts(A);
	vector<Point2d> normalized_B = normalizePts(B);
	Mat result = Mat::zeros(n, m, CV_64FC1);
	for (int r = 0; r < n; r++){
		for (int c = 0; c < m; c++){
			Vec2d tmp = Vec2d(0, 0);

			// 计算Va
			for (int k = 0; k < n; k++){
				if (k != r)
					tmp += vecA2B(normalized_A[r], normalized_A[k]);
			}
			// 计算Vb
			for (int k = 0; k < m; k++){
				if (k != c)
					tmp -= vecA2B(normalized_B[c], normalized_B[k]);
			}
			
			result.at<double>(r, c) = norm(tmp);
		}
	}
	return result;
}


// 道格拉斯-普克算法 轮廓多边形近似 轮廓简化 轮廓近似
vector<Point> Douglas(vector<Point> src, double D){
	double maxD = 0;
	int max_index = 1;

	if (src.size() == 2){
		return src;
	}

	for (int i = 1; i < src.size() - 1; i++){
		Vec3d OA = Vec3d(src[0].x - src[i].x, src[0].y - src[i].y, 0);
		Vec3d OB = Vec3d(src[src.size()-1].x - src[i].x, src[src.size()-1].y - src[i].y, 0);
		double h = norm(OA.cross(OB))/dist(src[0], src[src.size()-1]);
		if (h > maxD){
			maxD = h;
			max_index = i;
		}
	}

	if (maxD > D){
		return merge2VecPoint(Douglas(subVecPoint(src, 0, max_index), D), Douglas(subVecPoint(src, max_index, src.size()-1), D));
	}
	else if (maxD <= D){
		vector<Point> result;
		result.push_back(src[0]);
		result.push_back(src[max_index]);
		result.push_back(src[src.size()-1]);		
		return result;
	}

	vector<Point> result_null;
	return result_null;
}

// 合并两个点集
vector<Point> merge2VecPoint(vector<Point> A, vector<Point> B){
	vector<Point>::iterator it = B.begin()+1;
	while (it != B.end()){
		A.push_back(*it++);
	}
	return A;
}

// 点集的子集
vector<Point> subVecPoint(vector<Point> src, int start, int end){
	vector<Point> result(end - start + 1);
	memcpy(result.data(), src.data() + start, sizeof(Point)*(end - start + 1));
	return result;
}

// 选取轮廓中某点计算特征向量
Vec2d calcFeatureVector(vector<Point> src, int offset){
	Point A = ((offset == 0) ? src[src.size() - 1] : src[offset - 1]);
	Point O = src[offset];
	Point B = src[(offset + 1) % (src.size())];

	Vec2d result;
	result[0] = max(dist(A, O), dist(B, O)) / min(dist(A, O), dist(B, O));
	result[1] = acos(((A.x - O.x)*(B.x - O.x) + (A.y - O.y)*(B.y - O.y)) / (dist(A, O)*dist(B, O)));
	return result;
}

float flann_knn(Mat& m_destinations, Mat& m_object, vector<int>& ptpairs, vector<float>& dists/* = vector<float>()*/) {
	// find nearest neighbors using FLANN
	cv::Mat m_indices(m_object.rows, 1, CV_32S);
	cv::Mat m_dists(m_object.rows, 1, CV_32F);

	Mat dest_32f; m_destinations.convertTo(dest_32f, CV_32FC2);
	Mat obj_32f; m_object.convertTo(obj_32f, CV_32FC2);

	assert(dest_32f.type() == CV_32F);

	cv::flann::Index flann_index(dest_32f, cv::flann::KDTreeIndexParams(2));  // using 2 randomized kdtrees
	flann_index.knnSearch(obj_32f, m_indices, m_dists, 1, cv::flann::SearchParams(64));

	int* indices_ptr = m_indices.ptr<int>(0);
	//float* dists_ptr = m_dists.ptr<float>(0);
	for (int i = 0; i < m_indices.rows; ++i) {
		ptpairs.push_back(indices_ptr[i]);
	}

	dists.resize(m_dists.rows);
	m_dists.copyTo(Mat(dists));

	return cv::sum(m_dists)[0];
}

void findBestReansformSVD(Mat& _m, Mat& _d) {
	Mat m; _m.convertTo(m, CV_32F);
	Mat d; _d.convertTo(d, CV_32F);

	Scalar d_bar = mean(d);
	Scalar m_bar = mean(m);
	Mat mc = m - m_bar;
	Mat dc = d - d_bar;

	mc = mc.reshape(1); dc = dc.reshape(1);

	Mat H(2, 2, CV_32FC1);
	for (int i = 0; i < mc.rows; i++) {
		Mat mci = mc(Range(i, i + 1), Range(0, 2));
		Mat dci = dc(Range(i, i + 1), Range(0, 2));
		H = H + mci.t() * dci;
	}

	cv::SVD svd(H);

	Mat R = svd.vt.t() * svd.u.t();
	double det_R = cv::determinant(R);
	if (abs(det_R + 1.0) < 0.0001) {
		float _tmp[4] = { 1, 0, 0, cv::determinant(svd.vt*svd.u) };
		R = svd.u * Mat(2, 2, CV_32FC1, _tmp) * svd.vt;
	}
#ifdef BTM_DEBUG
	//for some strange reason the debug version of OpenCV is flipping the matrix
	R = -R;
#endif
	float* _R = R.ptr<float>(0);
	Scalar T(d_bar[0] - (m_bar[0] * _R[0] + m_bar[1] * _R[1]), d_bar[1] - (m_bar[0] * _R[2] + m_bar[1] * _R[3]));

	m = m.reshape(1);
	m = m * R;
	m = m.reshape(2);
	m = m + T;// + m_bar;
	m.convertTo(_m, CV_32S);
}

// 判断点集中是否包含点
bool containsPt(vector<cv::Point> pts, cv::Point t){
	vector<Point>::iterator it = pts.begin();
	while (it != pts.end()){
		if (*it == t){
			return true;
		}
		it++;
	}
	return false;
}

// 坐标点转ID值
int Pt2ID(cv::Point pt, int width){
	return (pt.y*width + pt.x);
}

// ID值转坐标
cv::Point ID2Pt(int ID, int width){
	return Point(ID % width, ID / width);
}