#pragma once
#include<opencv2/opencv.hpp>
#include"MyImage.h"
#include"color.h"

using namespace std;

#define HAAR    3 // ����HaarС��



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

	// ��ͼ��ʽ
	static const int STAIRS = 0;
	static const int LINE = 1;
	static const int STEM = 2;
	static const int SCATTER = 3;

	// Ԫ�ػ��Ʒ�ʽ
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

	// �źŵ�ͨ����
	int nChannel;

	// �źŵ���
	vector<int> N;

	// �ź����
	vector<vector<double>> dataX;
	vector<vector<double>> dataY;

	// �źŲ���
	vector<double> stepX;
	vector<double> stepY;
		
	// �źŷ�Χ
	vector<RANGE> rangeX;	// �ֱ��¼ ��Сֵ ���ֵ �Ϳ��=���ֵ-��Сֵ
	vector<RANGE> rangeY;

	vector<LIMIT> limitX;	// �ֱ��¼ ���� ��Сֵ�����ֵ�Ϳ�� = ���ֵ-��Сֵ
	vector<LIMIT> limitY;

	RANGE totalRangeX;		// �ܵķ�ΧRange ԭʼ���ݵ���ʵ��Χ
	RANGE totalRangeY;		// �ܵķ�ΧRange ԭʼ���ݵ���ʵ��Χ
	LIMIT totalLimitX;		// �ܵ�Limit ���Ƿֱ��ʺ�ķ�Χ
	LIMIT totalLimitY;		// �ܵ�Limit ���Ƿֱ��ʺ�ķ�Χ

	vector<double> ratio_x;	// x��ֱ���
	vector<double> ratio_y;	// y��ֱ���

	// ��ͼ���
	vector<vector<cv::Point>> drawPoint;
	
	vector<DRAW_MODE> draw_mode;
	vector<DRAW_MODE> ele_mode;

	int height;
	int width;

	// x�������� �� y��������
	vector<pair<int, int>> axis_pos;
	
	// �Ƿ��������
	bool ifShowGrid;
	void showGrid(bool visible);

	// ����Ԫ��
	void drawEle();
	
	inline int mapValueY(double input){
		int tmp = (int)((input - limitY[0].BottomV) / limitY[0].spanL * (rect_Work.height - 1));
		return (rect_Work.height - 1) - ((tmp<rect_Work.height) ? tmp : --tmp);
	}

	// ���ú����������򱳾�ɫ
	void setWorkAreaColor(cv::Scalar color);

	// ���ú���������ɫ
	void setFuncColor(cv::Scalar color, int index = -1);

	// ����X��ֱ���
	void setRatioX(double rX, int index = -1);

	// ����Y��ֱ���
	void setRatioY(double rY, int index = -1);

	// ����X�᷶Χ Y�᷶Χ
	void setAxis(double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax, int index = -1);

	// �����߿�
	void setLineWidth(int lw, int index = -1);

	// ����y�᷽��ķ�Χ
	void setLimitY(double minValue, double maxValue){
		totalLimitY.BottomV = minValue;
		totalLimitY.TopV = maxValue;
		totalLimitY.spanL = totalLimitY.TopV - totalLimitY.BottomV;
	}

	// ����Ԫ�سߴ�
	void setEle(int mode, int size, int index = -1, cv::Scalar color = cv::Scalar(-1, -1, -1));
	
	void Init();
	cv::Mat UpdateShow(/*DRAW_MODE mode = LINE*/);
	
	void getRanges();

	void getLimits();

	void resizeChannels(int size);

	void UpdateDrawPoints();

	// �������ص����꣬�������ݵ�id
	int getDataFromPixelCoord(const float &xpos, const float &ypos, const float &distThresh = 20);

	// ���һ·�ź�
	bool AddSignal(vector<double> t, vector<double> y, cv::Scalar color = cv::Scalar(-1, -1, -1));

	// ���һ·�ź�
	bool AddSignal(vector<double> y, cv::Scalar color);

	// ����դ����
	void drawGrids();

	// ���û�ͼģʽ
	void setDrawMode(DRAW_MODE mode, int index = -1);

	// ����x������
	bool xlabel(string title);

	// ����y������
	bool ylabel(string title);
	
	// ����ͼ�����
	bool setTitle(string title);
	cv::Mat workArea;
private:
	vector<int> linewidth;		//  ���ƺ���ͼ����߿�
	vector<int> eleSize;		//  ���ƺ���ͼ�� Ԫ�صĳߴ�

	bool onlyYdata;				// ֻ������y������

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

	string title;			// ����
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
			if (c == 13 || c == -1){					// ������»س�������ôȷ�ϵ�ǰ��� ���ٴ���
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
			// ����ƶ���Ϣ
		case cv::EVENT_MOUSEMOVE:
		{
			pt_MouseMove.x = x / (double)(srcWidth - 1) * roi[2] + roi[0];
			pt_MouseMove.y = y / (double)(srcHeight - 1) * roi[3] + roi[1];
			if (pt_MouseMove.y < 0) pt_MouseMove.y = 0;
			//cout << "����ƶ���" << pt_MouseMove << endl;
			if (b_showRGB){
				std::printf("RGB [%3d %3d %3d]\n", img.at<cv::Vec3b>(pt_MouseMove.y, pt_MouseMove.x)[2],
					img.at<cv::Vec3b>(pt_MouseMove.y, pt_MouseMove.x)[1],
					img.at<cv::Vec3b>(pt_MouseMove.y, pt_MouseMove.x)[0]);

				// �˸����
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
			}//cout << "���������£�" << pt_LButtonDown << endl;
			break;
		}
		case cv::EVENT_MBUTTONDOWN:
		{
			std::string savepath = selectSavePath("ͼƬ", "bmp");
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
			//cout << "������̧��" << pt_LButtonUp << endl;
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

	static cv::Point pt_LButtonDown;			// �������ʱ�ĵ�����
	static cv::Point pt_LButtonUp;				// ���̧��ʱ�ĵ�����
	static cv::Point pt_MouseMove;				// ����ƶ�ʱ�ĵ�����

	static cv::Point2d pt_MoveStart;
	static cv::Point2d pt_MoveDone;
	static cv::Point selectStart;				// ��ʼѡȡ�ĵ� ��showimg��x,y����
	static cv::Rect selectRect;					// ѡȡ���ο�
	static int selectRectThickness;				// ѡȡ���ο�ߵĿ��
	static bool isMoving;
	static bool isSelecting;					// �Ƿ�����ѡȡ����

	static cv::Vec4d roi;						// ��ǰ������ʾ��roi�Ĳ���
	static double scale;							// ���ű���

	static int srcWidth;
	static int srcHeight;

	static std::vector<int> pt_flag;					// 1-���ݵ���Ч 0-���ݵ㱻ѡ�д�ɾ�� -1-���ݵ���Ч
	static std::vector<cv::Point2d> inputPts;
	static int crossLen;						// �����ʮ�ֳ���
	static int crossThickness;					// �����ʮ�ִ�ϸ

	static bool doDel;							// ִ��ɾ������

	static cv::Mat showimg;						// ��ǰ���Ƶ�ͼ��
	static cv::Mat showimg_copy;				// ��ǰ����ͼ��ĸ���������ʾͼ�����и���
private:
	// ��ȡͼ���ROI 
	// ROI�Ĳ�����Vec4f��
	// ��ROI�Ŵ�����ͼ��ĳߴ��С��ͬ
	static cv::Mat getROI2SameSize(cv::Mat src);

	// ��ȡroi�Ĳ��� Ĭ����wheelUp��ͼ��Ŵ�
	static void updateRoiParms(int x, int y, bool wheelUp = true);


	cv::Mat src;						// Դͼ��
	static std::string name;
	static bool b_showRGB;				// ��ʾRGBͼ������

	int x;								// ���Ͻǵ�x����
	int y;								// ���Ͻǵ�y����	
};


