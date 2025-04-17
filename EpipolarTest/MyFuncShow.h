#pragma once
#include<opencv2/opencv.hpp>
#include"MyImage.h"
#include"color.h"

using namespace std;

#define HAAR    3 // 绘制Haar小波



//	typedef struct RANGE;
struct RANGE{
	double maxV;
	double minV;
	double spanV;

	RANGE(){
		maxV = 0;
		minV = 0;
		spanV = 0;
	}

	RANGE(const double& maxV_, const double& minV_, const double& spanV_){
		maxV = maxV_;
		minV = minV_;
		spanV = spanV_;
	}
	friend ostream &operator<<(ostream &out, RANGE value);
};

struct LIMIT{
	double TopV;
	double BottomV;
	double spanL;
	friend ostream &operator<<(ostream &out, LIMIT value);
};

class MyFuncShow
{
	typedef int DRAW_MODE;

public:
	MyFuncShow();
	MyFuncShow(vector<double> val, cv::Scalar color, int N_ = -1);
	MyFuncShow(vector<double> inputX, vector<double>inputY, int N_ = -1);
	MyFuncShow(double* inputX, double *inputY, int length);
	~MyFuncShow();

	// 绘图方式
	static const int STAIRS = 0;
	static const int LINE = 1;
	static const int STEM = 2;
	static const int SCATTER = 3;

	// 元素绘制方式
	static const int ELE_DEFAULT = -1;
	static const int ELE_NULL = 0;
	static const int ELE_SKEW_CROSS = 1;
	static const int ELE_STAR = 2;
	static const int ELE_SQUARE = 3;
	static const int ELE_CIRCLE = 4;
	static const int ELE_CROSS = 5;
	static const int ELE_TRIANGLE = 6;
	static const int ELE_TRIANGLE_SOLID = 7;
	static const int ELE_SQUARE_SOLID = 8;
	static const int ELE_STAR_SOLID = 9;
	static const int ELE_CIRCLE_SOLID = 10;

	// 信号的通道数
	int nChannel;

	// 信号点数
	vector<int> N;

	// 信号相关
	vector<vector<double>> dataX;
	vector<vector<double>> dataY;

	// 信号步长
	vector<double> stepX;
	vector<double> stepY;
		
	// 信号范围
	vector<RANGE> rangeX;	// 分别记录 最小值 最大值 和跨距=最大值-最小值
	vector<RANGE> rangeY;

	vector<LIMIT> limitX;	// 分别记录 区间 最小值，最大值和跨距 = 最大值-最小值
	vector<LIMIT> limitY;

	RANGE totalRangeX;		// 总的范围Range 原始数据的真实范围
	RANGE totalRangeY;		// 总的范围Range 原始数据的真实范围
	LIMIT totalLimitX;		// 总的Limit 考虑分辨率后的范围
	LIMIT totalLimitY;		// 总的Limit 考虑分辨率后的范围

	vector<double> ratio_x;	// x轴分辨率
	vector<double> ratio_y;	// y轴分辨率

	// 绘图相关
	vector<vector<cv::Point>> drawPoint;
	
	vector<DRAW_MODE> draw_mode;
	vector<DRAW_MODE> ele_mode;

	int height;
	int width;

	// x轴所在行 和 y轴所在列
	vector<pair<int, int>> axis_pos;
	
	// 是否绘制网格
	bool ifShowGrid;
	void showGrid(bool visible);

	// 绘制元素
	void drawEle();
	
	inline int mapValueY(double input){
		int tmp = (int)((input - limitY[0].BottomV) / limitY[0].spanL * (rect_Work.height - 1));
		return (rect_Work.height - 1) - ((tmp<rect_Work.height) ? tmp : --tmp);
	}

	// 设置函数绘制区域背景色
	void setWorkAreaColor(cv::Scalar color);

	// 设置函数绘制颜色
	void setFuncColor(cv::Scalar color, int index = -1);

	// 设置X轴分辨率
	void setRatioX(double rX, int index = -1);

	// 设置Y轴分辨率
	void setRatioY(double rY, int index = -1);

	// 设置X轴范围 Y轴范围
	void setAxis(double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax, int index = -1);

	// 设置线宽
	void setLineWidth(int lw, int index = -1);

	// 设置y轴方向的范围
	void setLimitY(double minValue, double maxValue){
		totalLimitY.BottomV = minValue;
		totalLimitY.TopV = maxValue;
		totalLimitY.spanL = totalLimitY.TopV - totalLimitY.BottomV;
	}

	// 设置元素尺寸
	void setEle(int mode, int size, int index = -1, cv::Scalar color = cv::Scalar(-1, -1, -1));
	
	void Init();
	cv::Mat UpdateShow(/*DRAW_MODE mode = LINE*/);
	
	void getRanges();

	void getLimits();

	void resizeChannels(int size);

	void UpdateDrawPoints();

	// 输入像素点坐标，返回数据的id
	int getDataFromPixelCoord(const float &xpos, const float &ypos, const float &distThresh = 20);

	// 添加一路信号
	bool AddSignal(vector<double> t, vector<double> y, cv::Scalar color = cv::Scalar(-1, -1, -1));

	// 添加一路信号
	bool AddSignal(vector<double> y, cv::Scalar color);

	// 绘制栅格线
	void drawGrids();

	// 设置绘图模式
	void setDrawMode(DRAW_MODE mode, int index = -1);

	// 设置x轴名称
	bool xlabel(string title);

	// 设置y轴名称
	bool ylabel(string title);
	
	// 设置图像标题
	bool setTitle(string title);
	cv::Mat workArea;
private:
	vector<int> linewidth;		//  绘制函数图像的线宽
	vector<int> eleSize;		//  绘制函数图像 元素的尺寸

	bool onlyYdata;				// 只给出了y的数据

	cv::Mat canvas;

	cv::Size size_canvas;
	cv::Rect rect_Work;

	cv::Scalar colorDefault;
	cv::Scalar colorWorkBK;
	cv::Scalar colorBK;
	cv::Scalar colorGrid;
	cv::Scalar colorFont;
	vector<cv::Scalar> colorFunc;
	vector<cv::Scalar> colorEle;

	string title;			// 标题
	string labelXAxis;		// xlabel
	string labelYAxis;		// ylabel
};

class MyWindow
{
public:
	MyWindow(std::string name, int w = 0, int h = 0, int x = -1, int y = -1)
	{
		this->name = name;
		cv::namedWindow(name, 0);
		if (w > 0 || h > 0)
			cvResizeWindow(name.c_str(), w, h);
		if (x >= 0 && y >= 0)
			cv::moveWindow(name, x, y);
	}

	~MyWindow(){}

	void setFuncShow(MyFuncShow *mf)
	{
		this->mf = mf;
	}

	std::string getWindowName(){ return name; }
	cv::Size getWindowSize(){ return cv::Size(srcWidth, srcHeight); }
	int getWindowHeight(){ return srcHeight; }
	int getWindowWidth(){ return srcWidth; }
	static void enableShowRGB() { b_showRGB = true; }
	static void disenableShowRGB() { b_showRGB = false; }

	void show(cv::Mat img)
	{
		src = convert2BGR(img);
		//cv::imshow(name, src);
		cv::setMouseCallback(name, on_MouseHandle, (void*)&src);
		srcWidth = img.cols;
		srcHeight = img.rows;
		crossLen = img.rows / 400 * 5;
		crossThickness = img.rows / 500;

		roi = cv::Vec4f(0, 0, srcWidth, srcHeight);

		showimg = getROI2SameSize(img);
		cv::imshow(name, showimg);
		showimg_copy = showimg.clone();

		while (1){
			int c = cv::waitKey(0);
			//	printf("c = %d\n", c);
			if (c == 26){					// Ctrl + Z
				for (int i = 0; i < (int)pt_flag.size(); i++){
					if (pt_flag[i] == 0)
						pt_flag[i] = 1;
				}
			}
			if (c == 0){				// Delete
				doDel = true;
			}
			if (c == 7602176){				// F5
				setAllTo<int>(pt_flag, 1);
			}
			if (c == 13 || c == -1){					// 如果按下回车键，那么确认当前结果 销毁窗口
				destroyWindow();
				return;
			}
			showimg = getROI2SameSize(img);
			if (selectId != -1)
			{
				drawCross(showimg, cv::Point(mf->drawPoint[0][selectId].x + 100, mf->drawPoint[0][selectId].y + 100),
					20, MC_RED, 2);
				drawString(showimg, int2string(selectId) + "  " + double2string(mf->dataY[0][selectId], 3),
					cv::Point(mf->drawPoint[0][selectId].x + 100, mf->drawPoint[0][selectId].y + 65), MC_BLACK, 20);
			}
			cv::imshow(name, showimg);
			showimg_copy = showimg.clone();
		}
	}

	static void on_MouseHandle(int event, int x, int y, int flags, void* param) {
		cv::Mat& img = *(cv::Mat*)param;
		switch (event)
		{
			// 鼠标移动消息
		case cv::EVENT_MOUSEMOVE:
		{
			pt_MouseMove.x = x / (double)(srcWidth - 1) * roi[2] + roi[0];
			pt_MouseMove.y = y / (double)(srcHeight - 1) * roi[3] + roi[1];
			if (pt_MouseMove.y < 0) pt_MouseMove.y = 0;
			//cout << "鼠标移动：" << pt_MouseMove << endl;
			if (b_showRGB){
				std::printf("RGB [%3d %3d %3d]\n", img.at<cv::Vec3b>(pt_MouseMove.y, pt_MouseMove.x)[2],
					img.at<cv::Vec3b>(pt_MouseMove.y, pt_MouseMove.x)[1],
					img.at<cv::Vec3b>(pt_MouseMove.y, pt_MouseMove.x)[0]);

				// 退格输出
				//for (int c = 0; c < 17; c++)printf("\b");
			}

			if (isMoving){
				pt_MoveDone.x = x / (double)(srcWidth - 1) * roi[2] + roi[0];
				pt_MoveDone.y = y / (double)(srcHeight - 1) * roi[3] + roi[1];
				roi[0] += pt_MoveStart.x - pt_MoveDone.x;
				roi[1] += pt_MoveStart.y - pt_MoveDone.y;
				showimg = getROI2SameSize(img);
				if (selectId != -1)
				{
					drawCross(showimg, cv::Point(mf->drawPoint[0][selectId].x + 100, mf->drawPoint[0][selectId].y + 100),
						20, MC_RED, 2);
					drawString(showimg, "[" + int2string(selectId) + "]  " + double2string(mf->dataY[0][selectId], 3),
						cv::Point(mf->drawPoint[0][selectId].x + 100, mf->drawPoint[0][selectId].y + 65), MC_BLACK, 20);
				}
				cv::imshow(name, showimg);
				showimg_copy = showimg.clone();
			}

			//if (isSelecting)
			//{
			//	cv::Rect sr = cv::Rect(selectStart, cv::Point(x, y));
			//	//cv::Mat copy = showimg.clone();
			//	rectangle(showimg_copy, sr, MC_BLUE, crossThickness, 8, 0);
			//	cv::imshow(name, showimg_copy);
			//	showimg_copy = showimg.clone();
			//	selectRect = cv::Rect(pt_LButtonDown, pt_MouseMove);
			//}
			break;
		}
		case cv::EVENT_RBUTTONDOWN:
		{
			isMoving = true;
			pt_MoveStart.x = x / (double)(srcWidth - 1) * roi[2] + roi[0];
			pt_MoveStart.y = y / (double)(srcHeight - 1) * roi[3] + roi[1];
			break;
		}
		case cv::EVENT_RBUTTONUP:
		{
			isMoving = false;
			break;
		}
		case cv::EVENT_LBUTTONDOWN:
		{
			isSelecting = true;
			selectStart.x = x;
			selectStart.y = y;
			pt_LButtonDown.x = x / (double)(srcWidth - 1) * roi[2] + roi[0];
			pt_LButtonDown.y = y / (double)(srcHeight - 1) * roi[3] + roi[1];
			int id = mf->getDataFromPixelCoord(pt_LButtonDown.x, pt_LButtonDown.y);
			selectId = id;
			if (id != -1)
			{
				double val = mf->dataY[0][id];
				printf("x = %d \ty = %lf\nval - min: %lf\nmax - val: %lf\n-----------------------------\n", 
					id, mf->dataY[0][id], val - mf->rangeY[0].minV, mf->rangeY[0].maxV - val);
				showimg = img.clone();

				drawCross(showimg, cv::Point(mf->drawPoint[0][id].x + 100, mf->drawPoint[0][id].y + 100), 
					20, MC_RED, 2);
				drawString(showimg, "[" + int2string(selectId) + "]  " + double2string(val, 3),
					cv::Point(mf->drawPoint[0][id].x + 100, mf->drawPoint[0][id].y + 65), MC_BLACK, 20);

				cv::imshow(name, showimg);
				showimg_copy = showimg.clone();
			}//cout << "鼠标左键按下：" << pt_LButtonDown << endl;
			break;
		}
		case cv::EVENT_MBUTTONDOWN:
		{
			std::string savepath = selectSavePath("图片", "bmp");
			imwrite(savepath, showimg);
			destroyWindow();
			return;
		}
		case cv::EVENT_LBUTTONUP:
		{
			isSelecting = false;
			pt_LButtonUp.x = x / (double)(srcWidth - 1) * roi[2] + roi[0];
			pt_LButtonUp.y = y / (double)(srcHeight - 1) * roi[3] + roi[1];

			/*selectRect = cv::Rect(pt_LButtonDown, pt_LButtonUp);
			for (int i = 0; i < (int)inputPts.size(); i++)
				if (selectRect.contains(inputPts[i]) && pt_flag[i] != -1)
					pt_flag[i] = 0;
			showimg = getROI2SameSize(img);
			cv::imshow(name, showimg);
			showimg_copy = showimg.clone();*/
			//cout << "鼠标左键抬起：" << pt_LButtonUp << endl;
			break;
		}
		case CV_EVENT_MOUSEWHEEL:
			break;
		}
	}

	static void destroyWindow(){
		cv::destroyWindow(name);
	}

	static int selectId;

	static MyFuncShow *mf;

	static cv::Point pt_LButtonDown;			// 左键按下时的点坐标
	static cv::Point pt_LButtonUp;				// 左键抬起时的点坐标
	static cv::Point pt_MouseMove;				// 鼠标移动时的点坐标

	static cv::Point2d pt_MoveStart;
	static cv::Point2d pt_MoveDone;
	static cv::Point selectStart;				// 开始选取的点 是showimg的x,y坐标
	static cv::Rect selectRect;					// 选取矩形框
	static int selectRectThickness;				// 选取矩形框边的宽度
	static bool isMoving;
	static bool isSelecting;					// 是否正在选取区域

	static cv::Vec4d roi;						// 当前窗口显示的roi的参数
	static double scale;							// 缩放比例

	static int srcWidth;
	static int srcHeight;

	static std::vector<int> pt_flag;					// 1-数据点有效 0-数据点被选中待删除 -1-数据点无效
	static std::vector<cv::Point2d> inputPts;
	static int crossLen;						// 点绘制十字长度
	static int crossThickness;					// 点绘制十字粗细

	static bool doDel;							// 执行删除操作

	static cv::Mat showimg;						// 当前绘制的图像
	static cv::Mat showimg_copy;				// 当前绘制图像的副本，在显示图像后进行复制
private:
	// 获取图像的ROI 
	// ROI的参数在Vec4f中
	// 将ROI放大至与图像的尺寸大小相同
	static cv::Mat getROI2SameSize(cv::Mat src);

	// 获取roi的参数 默认是wheelUp即图像放大
	static void updateRoiParms(int x, int y, bool wheelUp = true);


	cv::Mat src;						// 源图像
	static std::string name;
	static bool b_showRGB;				// 显示RGB图像坐标

	int x;								// 左上角的x坐标
	int y;								// 左上角的y坐标	
};


