#include "MyFuncShow.h"

using namespace cv;


cv::Point MyWindow::pt_LButtonDown = cv::Point(0, 0);
cv::Point MyWindow::pt_LButtonUp = cv::Point(0, 0);
cv::Point MyWindow::pt_MouseMove = cv::Point(0, 0);
cv::Point MyWindow::selectStart = cv::Point(0, 0);
int MyWindow::selectId = -1;
int MyWindow::srcHeight = 0;
int MyWindow::srcWidth = 0;
int MyWindow::selectRectThickness = 1;
bool MyWindow::isMoving = false;
bool MyWindow::isSelecting = false;
bool MyWindow::b_showRGB = false;
std::string MyWindow::name = "image";
cv::Point2d MyWindow::pt_MoveStart = cv::Point2d(0, 0);
cv::Point2d MyWindow::pt_MoveDone = cv::Point2d(0, 0);
cv::Rect MyWindow::selectRect = cv::Rect(0, 0, 1, 1);
std::vector<int> MyWindow::pt_flag;
std::vector<cv::Point2d> MyWindow::inputPts;
bool MyWindow::doDel = false;							// 执行删除操作
int MyWindow::crossLen = 0;								// 点绘制十字长度
int MyWindow::crossThickness = 0;						// 点绘制十字粗细
MyFuncShow* MyWindow::mf = NULL;



double MyWindow::scale = 1.0;
cv::Vec4d MyWindow::roi = cv::Vec4d(0, 0, 0, 0);		// 当前窗口显示的roi的参数
cv::Mat MyWindow::showimg = cv::Mat();					// 当前绘制的图像
cv::Mat MyWindow::showimg_copy = cv::Mat();				// 当前绘制的图像




// 获取图像的ROI 
// ROI的参数在Vec4f中
// 将ROI放大至与图像的尺寸大小相同
cv::Mat MyWindow::getROI2SameSize(cv::Mat src){
	if (roi[0] < 0.001) roi[0] = 0;
	if (roi[1] < 0.001) roi[1] = 0;
	if (roi[2] >= srcWidth) roi[2] = srcWidth;
	if (roi[3] >= srcHeight) roi[3] = srcHeight;
	if (roi[0] + roi[2] >= srcWidth) roi[0] = srcWidth - roi[2] - 2;
	if (roi[1] + roi[3] >= srcHeight) roi[1] = srcHeight - roi[3] - 2;
	if (roi[0] < 0.001) roi[0] = 0;
	if (roi[1] < 0.001) roi[1] = 0;

	int srcW = src.cols;
	int srcH = src.rows;
	int roix = floor(roi[0]);
	int roiy = floor(roi[1]);
	int roiw = ceil(roi[0] + roi[2]) - roix;
	int roih = ceil(roi[1] + roi[3]) - roiy;
	/*cout << "roi = " << roi << endl;
	cout << "roix = " << roix << endl;
	cout << "roiy = " << roiy << endl;
	cout << "roiw = " << roiw << endl;
	cout << "roih = " << roih << endl;*/

	cv::Mat copy = convert2BGR(src);

	for (int i = 0; i < (int)inputPts.size(); i++){
		if (pt_flag[i] == 1)
			drawCross(copy, inputPts[i], crossLen, MC_GREEN, crossThickness);
		else if (pt_flag[i] == 0 && !doDel){
			rectangle(copy, cv::Rect(cv::Point2d(inputPts[i].x - crossLen / 2 - 1, inputPts[i].y - crossLen / 2 - 1),
				cv::Point2d(inputPts[i].x + crossLen / 2 + 2, inputPts[i].y + crossLen / 2 + 2)), MC_RED, crossThickness, 8, 0);
			drawCross(copy, inputPts[i], crossLen, MC_GREEN, crossThickness);
		}
		else if (pt_flag[i] == 0 && doDel){
			pt_flag[i] = -1;
		}
		//	drawCross(copy, inputPts[i], 5, MC_RED, 1);
	}

	if (doDel) doDel = false;

	cv::Mat tmp = copy(cv::Rect(roix, roiy, roiw, roih)).clone();
	//cout << "tmp.size() = " << tmp.size() << endl;
	cv::Mat r1;
	resize(tmp, r1, src.size(), 0, 0, 0);
	//resizeGPU(tmp, r1, src.size(), 0, 0, 0);
	//cout << "tmp.size() = " << tmp.size() << endl;

	roix = srcW *(roi[0] - roix) / roiw;
	roiy = srcH * (roi[1] - roiy) / roih;
	roiw = srcW * roi[2] / roiw;
	roih = srcH *roi[3] / roih;

	/*cout << "roi = " << roi << endl;
	cout << "roix = " << roix << endl;
	cout << "roiy = " << roiy << endl;
	cout << "roiw = " << roiw << endl;
	cout << "roih = " << roih << endl;*/

	cv::Mat r2 = r1(cv::Rect(roix, roiy, roiw, roih)).clone();
	cv::Mat result;
	resize(r2, result, src.size(), 0, 0, 0);

	return result;
}

// 获取roi的参数
void MyWindow::updateRoiParms(int x, int y, bool wheelUp/* = true*/){
	double curx = x / (double)(srcWidth - 1) * roi[2] + roi[0];
	double cury = y / (double)(srcHeight - 1) * roi[3] + roi[1];
	/*pt_MouseMove.x = floor(curx);
	pt_MouseMove.y = floor(cury);
	printf("curx = %lf cury = %lf\n", curx, cury);*/

	if (wheelUp)
	{
		if (scale > 100.0 || roi[2] < 20 || roi[3] < 20)
			return;
		scale *= 1.25;
		roi[0] = roi[0] + (curx - roi[0]) *0.2;
		roi[1] = roi[1] + (cury - roi[1]) *0.2;
		roi[2] = roi[2] / 1.25;
		roi[3] = roi[3] / 1.25;
	}
	else{
		if (scale == 1.0)
			return;
		scale /= 1.25;
		roi[0] = roi[0] - (curx - roi[0]) *0.25;
		roi[1] = roi[1] - (cury - roi[1]) *0.25;
		roi[2] = roi[2] * 1.25;
		roi[3] = roi[3] * 1.25;
	}
}


ostream &operator<<(ostream &out, RANGE value){
	out << "minV = " << value.minV << endl;
	out << "maxV = " << value.maxV << endl;
	out << "spanV = " << value.spanV;
	return out;
}

ostream &operator<<(ostream &out, LIMIT value){
	out << "BottomV = " << value.BottomV << endl;
	out << "TopV = " << value.TopV << endl;
	out << "spanL = " << value.spanL;
	return out;
}

MyFuncShow::MyFuncShow()
	:height(800)
	, width(1280)
	, nChannel(0)	// 信号的通道数默认为0
{
	Mat A(height, width, CV_8UC3, Scalar(255, 255, 255));
	A.copyTo(canvas);
}

MyFuncShow::MyFuncShow(vector<double> val, cv::Scalar color, int N_ /*= -1*/){
	resizeChannels(1);
	if (N_ == -1)
		N[0] = (int)val.size();
	else if (N_ > 0)
		N[0] = N_;

	onlyYdata = true;

	dataX.clear();
	dataY.clear();

	colorFunc[0] = color;
	colorEle[0] = color;
	linewidth.push_back(1);
	eleSize.push_back(10);

	vector<double> inputX = linspace(1, N[0], N[0]);

	dataX.push_back(inputX);
	dataY.push_back(val);

	Init();
}

MyFuncShow::MyFuncShow(vector<double> inputX, vector<double>inputY, int N_/* = -1*/){
	resizeChannels(1);
	onlyYdata = false;
	if (N_ == -1)
		N[0] = (int)inputX.size();
	else if (N_ > 0)
		N[0] = N_;

	dataX.clear();
	dataY.clear();

	dataX.push_back(inputX);
	dataY.push_back(inputY);	

	Init();
}


MyFuncShow::MyFuncShow(double *inputX, double *inputY, int length){
	resizeChannels(1);
	onlyYdata = false;

	dataX[0].reserve(length);
	dataY[0].reserve(length);

	dataX[0].insert(dataX[0].begin(), &inputX[0], &inputX[length - 1]);
	dataY[0].insert(dataY[0].begin(), &inputY[0], &inputY[length - 1]);

	Init();
}

MyFuncShow::~MyFuncShow()
{
	for (int i = 0; i < nChannel; i++){
		dataX[i].clear();
		dataY[i].clear();
	}
}

void MyFuncShow::UpdateDrawPoints(){
	drawPoint.clear();
	for (int i = 0; i < nChannel; i++){
		vector<Point> tempPts;
		for (int j = 0; j < N[i]; j++){
			int x = (int)((dataX[i][j] - totalLimitX.BottomV) / totalLimitX.spanL * rect_Work.width);
			int tmpy = (int)((dataY[i][j] - totalLimitY.BottomV) / totalLimitY.spanL * rect_Work.height);
			int y = (rect_Work.height - 1) - ((tmpy<rect_Work.height) ? tmpy : --tmpy);

			tempPts.push_back(Point(x, y));
		}		
		drawPoint.push_back(tempPts);
	}
	//printf("UpdateDrawPoints Done!\n");
}

void MyFuncShow::showGrid(bool visible){
	this->ifShowGrid = visible;
}

// 绘制元素
void MyFuncShow::drawEle(){
	for (int i = 0; i < nChannel; i++){
		int n = (int)drawPoint[i].size();
		for (int j = 0; j < n; j++){
			if (ele_mode[i] == ELE_CROSS)
				drawCross(workArea, drawPoint[i][j], eleSize[i], colorEle[i], 1, 8, 0);
			else if (ele_mode[i] == ELE_CIRCLE)
				circle(workArea, drawPoint[i][j], eleSize[i] / 2.0, colorEle[i], 1, 8, 0);
			else if (ele_mode[i] == ELE_SKEW_CROSS)
				drawSkewCross(workArea, drawPoint[i][j], eleSize[i] *sqrt(2.0), colorEle[i], 1);
			else if (ele_mode[i] == ELE_SQUARE)
				drawSquare(workArea, drawPoint[i][j], eleSize[i]*0.8, colorEle[i], 1);
			else if (ele_mode[i] == ELE_STAR)
				drawStar(workArea, drawPoint[i][j], eleSize[i]*0.56, colorEle[i], 1);
			else if (ele_mode[i] == ELE_TRIANGLE)
				triangle(workArea, drawPoint[i][j], eleSize[i] * 0.67, colorEle[i], 1);
			else if (ele_mode[i] == ELE_TRIANGLE_SOLID)
				triangle(workArea, drawPoint[i][j], eleSize[i] * 0.67, colorEle[i], -1);
			else if (ele_mode[i] == ELE_SQUARE_SOLID)
				drawSquare(workArea, drawPoint[i][j], eleSize[i] * 0.8, colorEle[i], -1);
			else if (ele_mode[i] == ELE_STAR_SOLID)
				drawStar(workArea, drawPoint[i][j], eleSize[i] * 0.56, colorEle[i], -1);
		}
	}
}

Mat MyFuncShow::UpdateShow(/*DRAW_MODE mode*/){	
	UpdateDrawPoints();
	canvas.setTo(colorBK);
	workArea.setTo(colorWorkBK);

	//cout << drawPoint[0][0] << endl;

	//line(workArea, drawPoint[0][0], drawPoint[0][(int)drawPoint[0].size()-1], MC_GREEN, 1, 8, 0);
	for (int i = 0; i < nChannel; i++){
		if (draw_mode[i] == LINE){
			int n = (int)drawPoint[i].size();
			for (int j = 1; j < n; j++)
			{
				line(workArea, drawPoint[i][j - 1], drawPoint[i][j], colorFunc[i], linewidth[i], 8, 0);
			}
		}
		else if (draw_mode[i] == STEM){
			workArea.row(axis_pos[i].second).setTo(MC_BLACK);
			for (int j = 0; j < drawPoint[i].size(); j++){
				line(workArea, drawPoint[i][j], Point(drawPoint[i][j].x, axis_pos[i].second), colorFunc[i], linewidth[i], 8, 0);
				circle(workArea, drawPoint[i][j], 3, colorFunc[i], linewidth[i], 8, 0);
			}
		}
		else if (draw_mode[i] == STAIRS){
			for (int j = 1; j < drawPoint[i].size(); j++){
				//cout << "linewidth = " << linewidth[i] << endl;
				line(workArea, drawPoint[i][j - 1], Point(drawPoint[i][j].x, drawPoint[i][j - 1].y), colorFunc[i], linewidth[i], 8, 0);
				if (j < drawPoint[i].size() - 1)
					line(workArea, drawPoint[i][j], Point(drawPoint[i][j].x, drawPoint[i][j - 1].y), colorFunc[i], linewidth[i], 8, 0);
			}
			/*for (int j = 1; j < drawPoint[i].size(); j++){
				line(workArea, drawPoint[i][j], Point(drawPoint[i][j - 1].x, drawPoint[i][j].y), colorFunc, linewidth, 8, 0);
				}*/
		}
		else if (draw_mode[i] == SCATTER){
			for (int j = 0; j < drawPoint[i].size(); j++){
				//line(workArea, drawPoint[i][j], Point(drawPoint[i][j].x, axis_pos[i].second), colorFunc[i], linewidth, 8, 0);
				circle(workArea, drawPoint[i][j], 3, colorFunc[i], linewidth[i], 8, 0);
			}
		}
		else if (draw_mode[i] == HAAR){
			int axis_stepX = (int)(workArea.cols / drawPoint[i].size());
			for (int j = 0; j < drawPoint[i].size(); j++){
				line(workArea, Point(j*axis_stepX, drawPoint[i][j].y),
					Point((j + 1)*axis_stepX, drawPoint[i][j].y), colorFunc[i], linewidth[i], 8, 0);
				if (j < drawPoint[i].size() - 1)
					line(workArea, Point((j + 1)*axis_stepX, drawPoint[i][j].y),
					Point((j + 1)*axis_stepX, drawPoint[i][j + 1].y), colorFunc[i], linewidth[i], 8, 0);
			}
			/*for (int j = 1; j < drawPoint[i].size(); j++){
			line(workArea, drawPoint[i][j], Point(drawPoint[i][j - 1].x, drawPoint[i][j].y), colorFunc, linewidth, 8, 0);
			}*/

		}
	}
	drawEle();				// 绘制元素
	if (ifShowGrid)
		drawGrids();

#ifdef USE_MFC
	// 添加坐标轴区间 文字
	// x轴文字
	drawString(canvas, double2string(totalLimitX.BottomV, 3), Point(rect_Work.x, rect_Work.y + rect_Work.height + 20), MC_BLACK, 20);
	drawString(canvas, double2string(totalLimitX.TopV, 3), Point(rect_Work.x + rect_Work.width - 30, rect_Work.y + rect_Work.height + 20), MC_BLACK, 20);

	// y轴文字
	drawString(canvas, double2string(totalLimitY.TopV, 3), Point(rect_Work.x - 60, rect_Work.y - 10), MC_BLACK, 20);
	drawString(canvas, double2string(totalLimitY.BottomV, 3), Point(rect_Work.x - 60, rect_Work.y + rect_Work.height - 10), MC_BLACK, 20);
#endif
	return canvas;
}

void MyFuncShow::Init(){
	size_canvas = Size(1400, 800);
	rect_Work = Rect(100, 100, 1200, 600);

	colorDefault = Scalar(0, 0, 255);
	colorGrid = MC_GRAY;
	colorBK = Scalar(190, 190, 190);
	colorWorkBK = MC_WHITE;
	colorFont = MC_BLACK;

	canvas = Mat::zeros(size_canvas, CV_8UC3);
	canvas.setTo(colorBK);

	draw_mode[0] = LINE;
	ele_mode[0] = ELE_NULL;
	
	ifShowGrid = false;									// 默认不绘制网格
	workArea = canvas(Rect(100, 100, rect_Work.width, rect_Work.height));
	workArea.setTo(colorWorkBK);
	workArea.row(workArea.rows - 1).setTo(colorGrid);
	workArea.col(0).setTo(colorGrid);	

	title = "";
	labelXAxis = "";
	labelYAxis = "";

	getRanges();
	if (onlyYdata)
		ratio_x.push_back(1.0);
	else
		ratio_x.push_back(resolution(rangeX[0].spanV / dataX[0].size()));
	ratio_y.push_back(resolution(rangeY[0].spanV / dataY[0].size()));
	getLimits();
	//printf("Init Done!\n");
}

void MyFuncShow::getRanges(){
	rangeX.resize(nChannel);
	rangeY.resize(nChannel);
	for (int i = 0; i < nChannel; i++){
		double minV = 0.0, maxV = 0.0;
		int minIndex = 0, maxIndex = 0;

		minMaxVector(dataX[i], minV, maxV, minIndex, maxIndex, N[i]);
		rangeX[i].minV = minV;
		rangeX[i].maxV = maxV;
		rangeX[i].spanV = maxV - minV;

		minMaxVector(dataY[i], minV, maxV, minIndex, maxIndex, N[i]);
		rangeY[i].minV = minV;
		rangeY[i].maxV = maxV;
		rangeY[i].spanV = maxV - minV;
		
		//printf("minx[%d] = %.13lf maxx[%d] = %.13lf miny[%d] = %.13lf maxy[%d] = %.13lf\n", i,rangeX[i].minV, i,rangeX[i].maxV, i,rangeY[i].minV, i,rangeY[i].maxV);
	}	
	// 计算总的totalRange
	totalRangeX.minV = DBL_MAX;
	totalRangeX.maxV = DBL_MIN;

	totalRangeY.minV = DBL_MAX;
	totalRangeY.maxV = DBL_MIN;
	for (int i = 0; i < nChannel; i++){
		if (rangeX[i].minV < totalRangeX.minV)
			totalRangeX.minV = rangeX[i].minV;
		if (rangeY[i].minV < totalRangeY.minV)
			totalRangeY.minV = rangeY[i].minV;
		if (rangeX[i].maxV > totalRangeX.maxV)
			totalRangeX.maxV = rangeX[i].maxV;
		if (rangeY[i].maxV > totalRangeY.maxV)
			totalRangeY.maxV = rangeY[i].maxV;
	}
	totalRangeX.spanV = totalRangeX.maxV - totalRangeX.minV;
	totalRangeY.spanV = totalRangeY.maxV - totalRangeY.minV;
	//cout << totalRangeX << endl;
	//cout << totalRangeY << endl;
}

bool MyFuncShow::setTitle(string title){
#ifdef USE_MFC
	this->title = title;
	int space = (int)title.size() / 2;
	drawString(canvas, string2pChar(this->title), Point((int)(rect_Work.x + rect_Work.width / 2 - space * 17.8), 39),
		colorFont, 36);
	return true;
#else
	printf("Please define USE_MFC first!\n");
	return false;
#endif
}

// 设置x轴名称
bool MyFuncShow::xlabel(string str){
#ifdef USE_MFC
	this->labelXAxis = str;
	int space = (int)str.size() / 2;
	drawString(canvas, string2pChar(this->labelXAxis), Point((int)(rect_Work.x + rect_Work.width / 2 - space * 17.8), 720),
		colorFont, 36);
	return true;
#else
	printf("Please define USE_MFC first!\n");
	return false;
#endif
}

// 设置y轴名称
bool MyFuncShow::ylabel(string str){
#ifdef USE_MFC
	this->labelYAxis = str;
	int space = (int)str.size() / 2;
	drawString(canvas, string2pChar(this->labelYAxis), Point(30, rect_Work.y+rect_Work.height / 2),-90,
		colorFont, 36);
	return true;
#else
	printf("Please define USE_MFC first!\n");
	return false;
#endif
}

// 绘制栅格线
void MyFuncShow::drawGrids(){
	double baseX = calTopNum(limitX[0].BottomV, ratio_x[0] * 10);
	double baseY = calTopNum(limitY[0].BottomV, ratio_y[0] * 10);
		
	for (int i = 0; ; i++){
		int tempCol = (int)((baseX + ratio_x[0]*100 * i - limitX[0].BottomV) / limitX[0].spanL * rect_Work.width);
		if (tempCol < workArea.cols)
			//drawDottedLineCol(workArea, tempCol, colorGrid, 1);
			workArea.col(tempCol).setTo(colorGrid);
		else
			break;
	}

	for (int i = 0;; i++){
		int tempRow = (int)((baseY + ratio_y[0] * 100 * i - limitY[0].BottomV) / limitY[0].spanL * rect_Work.height);
		tempRow = rect_Work.height - tempRow;
		if (tempRow < workArea.rows && tempRow >= 0)
			workArea.row(tempRow).setTo(colorGrid);
			//drawDottedLineRow(workArea, tempRow, colorGrid, 1);			
		else
			break;
	}

	// 两横
	line(canvas, Point(100, 101 + rect_Work.height), Point(rect_Work.width + 101, 101 + rect_Work.height), MC_BLACK, 1);
	line(canvas, Point(100, 99), Point(101 + rect_Work.width, 99), MC_BLACK, 1);
	// 两竖
	line(canvas, Point(101 + rect_Work.width, 99), Point(rect_Work.width + 101, 101 + rect_Work.height), MC_BLACK, 1);
	line(canvas, Point(100, 99), Point(100, 101 + rect_Work.height), MC_BLACK, 1);
}

// 获取信号的Limit 为range考虑分辨率ratio之后的结果
void MyFuncShow::getLimits(){
	limitX.resize(nChannel);
	limitY.resize(nChannel);
	axis_pos.resize(nChannel);
	for (int i = 0; i < nChannel; i++){	
		limitX[i].TopV = calTopNum(rangeX[i].maxV, ratio_x[i]);
		limitX[i].BottomV = calFloorNum(rangeX[i].minV, ratio_x[i]);
		limitX[i].spanL = limitX[i].TopV - limitX[i].BottomV;

		limitY[i].TopV = calTopNum(rangeY[i].maxV, ratio_y[i]);
		limitY[i].BottomV = calFloorNum(rangeY[i].minV, ratio_y[i]);
		limitY[i].spanL = limitY[i].TopV - limitY[i].BottomV;
	
		int x_pos = (int)((0 - limitX[i].BottomV) / limitX[i].spanL * rect_Work.width);
		int tmpy_pos = (int)((0.0 - limitY[i].BottomV) / limitY[i].spanL * rect_Work.height);
		int y_pos = (rect_Work.height - 1) - ((tmpy_pos<rect_Work.height) ? tmpy_pos : --tmpy_pos);
		axis_pos[i] = make_pair(x_pos, y_pos);
	}

	// 计算总的totalRange
	totalLimitX.BottomV = DBL_MAX;
	totalLimitX.TopV = DBL_MIN;

	totalLimitY.BottomV = DBL_MAX;
	totalLimitY.TopV = DBL_MIN;
	for (int i = 0; i < nChannel; i++){
		if (limitX[i].BottomV < totalLimitX.BottomV)
			totalLimitX.BottomV = limitX[i].BottomV;
		if (limitY[i].BottomV < totalLimitY.BottomV)
			totalLimitY.BottomV = limitY[i].BottomV;
		if (limitX[i].TopV > totalLimitX.TopV)
			totalLimitX.TopV = limitX[i].TopV;
		if (limitY[i].TopV > totalLimitY.TopV)
			totalLimitY.TopV = limitY[i].TopV;
	}
	totalLimitX.spanL = totalLimitX.TopV - totalLimitX.BottomV;
	totalLimitY.spanL = totalLimitY.TopV - totalLimitY.BottomV;
	//cout << totalLimitX << endl;
	//cout << totalLimitY << endl;
}

// 添加一路信号
bool MyFuncShow::AddSignal(vector<double> y, Scalar color){
	nChannel++;
	linewidth.push_back(1);
	eleSize.push_back(5);

	if (color[0] == -1){
		colorFunc.push_back(colorList[nChannel - 1]);
		colorEle.push_back(colorList[nChannel - 1]);
	}
	else {
		colorFunc.push_back(color);
		colorEle.push_back(color);
	}

	N.push_back((int)y.size());

	vector<double> t = linspace(0, N[N.size() - 1] - 1, N[N.size() - 1]);
	dataX.push_back(t);
	dataY.push_back(y);
	draw_mode.push_back(MyFuncShow::LINE);
	ele_mode.push_back(MyFuncShow::ELE_CIRCLE);

	stepX.push_back(0.1);
	stepY.push_back(0.1);

	getRanges();
	ratio_x.push_back(resolution(rangeX[nChannel - 1].spanV / dataX[nChannel - 1].size()));
	ratio_y.push_back(resolution(rangeY[nChannel - 1].spanV / dataY[nChannel - 1].size()));
	getLimits();
	return true;
}

// 添加一路信号
bool MyFuncShow::AddSignal(vector<double> t, vector<double> y, Scalar color){
	CV_Assert(t.size() == y.size());
	nChannel++;

	if (color[0] == -1){
		colorFunc.push_back(colorList[nChannel - 1]);
	}
	else {
		colorFunc.push_back(color);
	}

	N.push_back((int)t.size());
	dataX.push_back(t);
	dataY.push_back(y);

	stepX.push_back(0.1);
	stepY.push_back(0.1);

	getRanges();
	ratio_x.push_back(resolution(rangeX[nChannel - 1].spanV / dataX[nChannel-1].size()));
	ratio_y.push_back(resolution(rangeY[nChannel - 1].spanV / dataY[nChannel-1].size()));
	getLimits();
	return true;
}



void MyFuncShow::resizeChannels(int size){
	nChannel = size;
	N.resize(size);
	dataX.resize(size);
	dataY.resize(size);

	stepX.resize(size);
	stepY.resize(size);
	
	draw_mode.resize(size);
	ele_mode.resize(size);
	axis_pos.resize(size);

	colorFunc.resize(size);
	colorEle.resize(size);
}

// 设置线宽
void MyFuncShow::setLineWidth(int lw, int index/* = -1*/){
	if (index == -1){
		for (int i = 0; i < nChannel; i++)
			linewidth[i] = lw;
	}
	else {
		linewidth[index] = lw;
	}
}

// 设置元素尺寸
void MyFuncShow::setEle(int mode, int size, int index/* = -1*/, cv::Scalar color/* = cv::Scalar(-1, -1, -1)*/){
	if (index == -1){
		for (int i = 0; i < nChannel; i++){
			eleSize[i] = size;
		}
	}
	else {
		eleSize[index] = size;
		if(color[0] >= 0)
			colorEle[index] = color;
		if(mode!=ELE_DEFAULT)
			ele_mode[index] = mode;
	}
}

// 设置绘图模式
void MyFuncShow::setDrawMode(DRAW_MODE mode, int index/* = -1*/){
	if (index == -1){
		for (int i = 0; i < nChannel; i++)
			draw_mode[i] = mode;
	}
	else 
		draw_mode[index] = mode;
}
// 设置X轴范围 Y轴范围
void MyFuncShow::setAxis(double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax, int index){
	if (index == -1){
		limitX[0].BottomV = xAxisMin;
		limitX[0].TopV = xAxisMax;
		limitX[0].spanL = limitX[0].TopV - limitX[0].BottomV;

		limitY[0].BottomV = yAxisMin;
		limitY[0].TopV = yAxisMax;
		limitY[0].spanL = limitY[0].TopV - limitY[0].BottomV;
	}
	else {
		LIMIT tmpLimitX, tmpLimitY;
		tmpLimitX.BottomV = xAxisMin;
		tmpLimitX.TopV = xAxisMax;
		tmpLimitX.spanL = tmpLimitX.TopV - tmpLimitX.BottomV;

		tmpLimitY.BottomV = yAxisMin;
		tmpLimitY.TopV = yAxisMax;
		tmpLimitY.spanL = tmpLimitY.TopV - tmpLimitY.BottomV;

		if (index == nChannel - 1){
			limitX[index].BottomV = tmpLimitX.BottomV;
			limitX[index].BottomV = tmpLimitX.BottomV;
			limitX[index].BottomV = tmpLimitX.BottomV;
			limitX[index].BottomV = tmpLimitX.BottomV;
			limitX[index].BottomV = tmpLimitX.BottomV;
			limitX[index].BottomV = tmpLimitX.BottomV;
		}

		limitX[index].BottomV = xAxisMin;
		limitX[index].TopV = xAxisMax;
		limitX[index].spanL = limitX[index].TopV - limitX[index].BottomV;
		
		limitY[index].BottomV = yAxisMin;
		limitY[index].TopV = yAxisMax;
		limitY[index].spanL = limitY[index].TopV - limitY[index].BottomV;
	}
}

// 设置函数绘制区域背景色
void MyFuncShow::setWorkAreaColor(cv::Scalar color){
	colorWorkBK = color;
	workArea.setTo(color);
}

// 设置X轴分辨率
void MyFuncShow::setRatioX(double rX, int index/* = -1*/){
	CV_Assert(index >= -1 && index < nChannel);
	if (index == -1)
		ratio_x[0] = rX;
	else {
		ratio_x[index] = rX;
	}
	getLimits();
}

// 设置Y轴分辨率
void MyFuncShow::setRatioY(double rY, int index/* = -1*/){
	CV_Assert(index >= -1 && index < nChannel);
	if (index == -1){
		ratio_y[0] = rY;
	}
	else {
		ratio_y[index] = rY;
	}
	getLimits();
}

// 设置函数绘制颜色
void MyFuncShow::setFuncColor(cv::Scalar color, int index/* = -1*/){
	CV_Assert(index >= -1 && index < nChannel);
	if (index == -1)
		colorFunc[0] = color;
	else {
		colorFunc[index] = color;
	}
}

int MyFuncShow::getDataFromPixelCoord(const float &xpos, const float &ypos, const float &distThresh/* = 5*/)
{
	float newx = xpos - 100;
	float newy = ypos - 100;

	int n = (int)drawPoint[0].size();
	std::vector<double> ds(drawPoint[0].size());
#pragma omp parallel for
	for (int j = 0; j < n; j++)
	{
		float deltax = newx - drawPoint[0][j].x;
		float deltay = newy - drawPoint[0][j].y;

		ds[j] = sqrt(deltax * deltax + deltay * deltay);
	}
	int index = minIndexV<double>(ds);
	if (ds[index] < distThresh)
	{
		return index;
	}
	else
		return -1;
}

