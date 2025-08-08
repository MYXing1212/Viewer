// EpipolarTestDlg.cpp : implementation file
//
#include "stdafx.h"
#include "EpipolarTest.h"
#include "EpipolarTestDlg.h"
#include "afxdialogex.h"
#include"MyFuncShow.h"
#include"MyConfig.h"
#include "VectorRenderer.h"
#include"fbomanager.h"
#include"BigImageRenderer.h"

#define WM_MY_MESSAGE (WM_USER+101)

float lastX = 0;
float lastY = 0;
bool flag1d = false;
bool firstMouse = true;
bool leftCamFlag = true;
bool xcoordFlag = true;
int SCR_WIDTH, SCR_HEIGHT;

// 绘制灰度变化曲线对应的范围
float lowThreshY = 0.0f, highThreshY = 1.0f;
float lowThreshY_right = 0.0f, highThreshY_right = 1.0f;

float showThreshLow = 0;
float showThreshHigh = 1.0f;
float showThreshLow_right = 0, showThreshHigh_right = 1.0f;

bool syncFlag = false;
bool syncLRFlag = false;

bool isSaving = false;

std::vector<float> graysRow, graysCol;

//cv::Mat matchedPtCol;						// 用于左右同步的查找表
//glm::vec2 matchedPixel;						// 匹配的右相机的像素点坐标
//glm::vec2 matchedPixelInTexture;			// 匹配的右相机的点在纹理坐标系下的坐标
std::vector<cv::Mat> dataPrjMap;

BigImageRenderer patternLeft;
BigImageRenderer patternRight;
ElementGeoRenderer ele;
VectorRenderer vecr;

GLFWwindow *window;

glm::vec3 grayL, grayR;

cv::Mat imgLeft = cv::Mat::zeros(100, 100, CV_8UC3);


//#define WM_MESSAGE_UPDATE_POS (WM_USER+1022)
const UINT WM_MESSAGE_UPDATE_POS = ::RegisterWindowMessage(_T("Update_Pos")); // 接收方

void updateRender(GLFWwindow *window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void drop_callback(GLFWwindow* window, int count, const char** paths);

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CAboutDlg dialog used for App About

std::vector<glm::vec2> pts2D;

bool savePts2D(std::string filepath, std::vector<glm::vec2> pts)
{
	std::ofstream r(filepath);
	for (int i = 0; i < (int)pts.size(); i++)
	{
		r << std::setprecision(12) << pts[i].x << " " << pts[i].y << std::endl;
	}
	r.close();
	return true;
}

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CEpipolarTestDlg dialog
CEpipolarTestDlg::CEpipolarTestDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CEpipolarTestDlg::IDD, pParent)
	, m_str_info(_T(""))
	, m_edit_thresh_low(lowThreshY) 
	, m_edit_thresh_low_right(lowThreshY_right)
	, m_edit_thresh_high(highThreshY)
	, m_edit_thresh_high_right(highThreshY_right)
	, m_edit_show_thresh_low(-3.0f)
	, m_edit_show_thresh_high(3.0f)
	, m_edit_show_thresh_low_right(-3.0f)
	, m_edit_show_thresh_high_right(3.0f)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CEpipolarTestDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_THRESH_LOW, m_edit_thresh_low);
	DDX_Text(pDX, IDC_EDIT_THRESH_HIGH, m_edit_thresh_high);
	DDX_Text(pDX, IDC_EDIT_SHOW_THRESH_LOW, m_edit_show_thresh_low);
	DDX_Text(pDX, IDC_EDIT_SHOW_THRESH_HIGH, m_edit_show_thresh_high);

	DDX_Text(pDX, IDC_EDIT_THRESH_LOW_RIGHT, m_edit_thresh_low_right);
	DDX_Text(pDX, IDC_EDIT_THRESH_HIGH_RIGHT, m_edit_thresh_high_right);
	DDX_Text(pDX, IDC_EDIT_SHOW_THRESH_LOW_RIGHT, m_edit_show_thresh_low_right);
	DDX_Text(pDX, IDC_EDIT_SHOW_THRESH_HIGH_RIGHT, m_edit_show_thresh_high_right);
}

BEGIN_MESSAGE_MAP(CEpipolarTestDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_LOAD1, &CEpipolarTestDlg::OnBnClickedBtnLoad1)
	ON_WM_SHOWWINDOW()
	//ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_BUTTON_APPLY, &CEpipolarTestDlg::OnBnClickedButtonApply)
	ON_BN_CLICKED(IDC_BUTTON_APPLY2, &CEpipolarTestDlg::OnBnClickedButtonApply)
	ON_REGISTERED_MESSAGE(WM_MESSAGE_UPDATE_POS, OnUpdatePOS)
	ON_WM_TIMER()
	ON_WM_COPYDATA()
	ON_BN_CLICKED(IDC_CHECK_SYNC, &CEpipolarTestDlg::OnBnClickedCheckSync)
	ON_BN_CLICKED(IDC_CHECK_SYNC_LEFT_RIGHT, &CEpipolarTestDlg::OnBnClickedCheckSyncLeftRight)
	ON_BN_CLICKED(IDC_CHECK_PSEU_LEFT, &CEpipolarTestDlg::OnBnClickedCheckPseuLeft)
	ON_BN_CLICKED(IDC_CHECK_PSEU_RIGHT, &CEpipolarTestDlg::OnBnClickedCheckPseuRight)
	ON_EN_CHANGE(IDC_EDIT_SHOW_THRESH_LOW, &CEpipolarTestDlg::OnEnChangeEditShowThreshLow)
	ON_EN_CHANGE(IDC_EDIT_SHOW_THRESH_HIGH, &CEpipolarTestDlg::OnEnChangeEditShowThreshHigh)
	ON_MESSAGE(WM_MY_MESSAGE, &CEpipolarTestDlg::OnMyMessage)
	ON_BN_CLICKED(IDC_BUTTON_COPYTORIGHT, &CEpipolarTestDlg::OnBnClickedButtonCopytoright)
	ON_BN_CLICKED(IDC_BUTTON_COPYTOLEFT, &CEpipolarTestDlg::OnBnClickedButtonCopytoleft)
END_MESSAGE_MAP()


// CEpipolarTestDlg message handlers
BOOL CEpipolarTestDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	AllocConsole();
	SetConsoleTitle(_T("调试输出"));
	FILE *pf;
	freopen_s(&pf, "CONOUT$", "w",
		stdout);

	// Init GLFW glfw: initialize and configure
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//没有边框和标题栏 
	glfwWindowHint(GLFW_DECORATED, GL_FALSE);

	Config::setParameterFile("myconfig.yaml");
	std::string datapath = Config::get<std::string>("datapath");
	cout << "datapath = " << datapath << endl;

	//matchedPtCol = readMatInBinaryFloat("matchedPtCol.bin");


	//for (int p = 16; p <= 36; p++)
	//{
	//	cv::Mat tmpPrjMap = readMatInBinaryFloat("E:\\data\\DEFOCUS\\slant 0000\\prjMap" + to_string(p) + ".bin");
	//	dataPrjMap.push_back(tmpPrjMap);
	//}

	//std::string strBase = "G:\\Test Data\\2019-11-14 Phone\\Data20191114102810\\3\\";
	//for (int i = 0; i < 6; i++)
	//{
	//	cv::Mat tmpPrjMap = readMatInBinaryFloat(strBase + "coordR_" + to_string(i+1) + ".bin");
	//	if(tmpPrjMap.data)
	//		dataPrjMap.push_back(tmpPrjMap);
	//}

	//std::string strBase = "G:\\Test Data\\2019-11-14 Phone\\Data20191114102810\\5\\";
	//for (int i = 0; i < 6; i++)
	//{
	//	cv::Mat tmpPrjMap = readMatInBinaryFloat(strBase + "coordR_" + to_string(i + 1) + ".bin");
	//	if (tmpPrjMap.data)
	//		dataPrjMap.push_back(tmpPrjMap);
	//}

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CEpipolarTestDlg::OnBnClickedBtnLoad1()
{
}

LRESULT CEpipolarTestDlg::OnMyMessage(WPARAM wParam, LPARAM lParam)
{
	UpdateData(TRUE);
	printf("MyMessage!\n");

	m_edit_thresh_high = showThreshHigh;
	m_edit_thresh_low = showThreshLow;

	m_edit_show_thresh_low = showThreshLow;
	m_edit_show_thresh_high = showThreshHigh;

	m_edit_show_thresh_low_right = showThreshLow_right;
	m_edit_show_thresh_high_right = showThreshHigh_right;
	UpdateData(FALSE);
	return 0;
}


// 拖入文件 不支持中文路径
void drop_callback(GLFWwindow* window, int count, const char** paths)
{
	std::string filename = paths[0];
	std::string str = getFileSuffix(filename);

	if (str != "bmp" && str != "jpg" && str != "png" && str != "tiff" && str != "xml" && str != "yaml" && str!="bin" && str!="txt")
	{
		printf("ERROR: 无法显示该图片!\n");
	}
	cv::Mat img;
	if (str == "bmp" || str == "jpg" || str == "png" || str == "tiff" || str =="jpeg")
	{
		img = cv::imread(filename, -1);
		//img = DFT(img);
	}
	else if (str == "xml" || str == "yaml")
	{
		img = readMat(filename);
		img.convertTo(img, CV_32FC1);
		for (int i = 0; i < img.total(); i++)
		{
			if (isnan(img.ptr<float>(0)[i]))
				img.ptr<float>(0)[i] = 0.0f;
		}
	
		//printf("range = %f %f\n", minM<float>(img), maxM<float>(img));
		//pattern.setNormalizeRange(glm::vec2(500.0f, 900.0f));
		//pattern.setNormalizeRange(glm::vec2(minM<float>(img), maxM<float>(img)));
	}
	else if (str == "bin")
	{
		img = readMatInBinaryFloat(filename);
		if (img.cols == 1)
			img = img.t();

		for (int i = 0; i < img.rows * img.cols * img.channels(); i++)
		{
			if (isnan(img.ptr<float>(0)[i]))
				img.ptr<float>(0)[i] = 0.0f;
		}	
	}
	else if (str == "txt")
	{
		pts2D = loadPt2d_GL(filename);
		cout << "pts2D.size() = " << pts2D.size() << endl;
		for (int i = 0; i < pts2D.size(); i++)
		{
			printf("%f %f\n", pts2D[i].x, pts2D[i].y);
		}
		return;
	}

	//printf("channel = %d\n", img.channels());
	double lv, hv;
	minMaxLoc(img, &lv, &hv);	
	printf("lv = %lf, hv = %lf\n", lv, hv);
	theApp.m_pMainWnd->UpdateData(TRUE);

	if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
	{
		if (img.rows == 1)
		{
			flag1d = true;
			lowThreshY = lv;
			highThreshY = hv;
		}
		else
		{
			flag1d = false;
			lowThreshY = -1;
			highThreshY = 1;
			showThreshLow = lv;
			showThreshHigh = hv;
		}

		patternLeft.setFileInfo(filename);
		theApp.m_pMainWnd->GetDlgItem(IDC_STATIC_TITLE_LEFT)->SetWindowTextW(_T("左图像: ") +
			string2CString(filename + " " + to_string(img.cols) + "×" + to_string(img.rows)));
		printf("左图像: %s w: %d h: %d channel = %d\n", filename.c_str(), img.cols, img.rows, img.channels());
		patternLeft.reset();
		patternLeft.updateImg(img, true);
		for (int i = 0; i < 1; i++)
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			if(!flag1d)
				patternLeft.RenderPattern();
			else
			{
				glViewport(0, SCR_HEIGHT / 2, SCR_WIDTH / 2, SCR_HEIGHT / 2);
				vecr.setData(3, patternLeft.texture);
				vecr.render(glm::vec2(patternLeft.xminTexture, patternLeft.selectPixelInTexture.y),
					glm::vec2(patternLeft.xmaxTexture, patternLeft.selectPixelInTexture.y), glm::vec3(1.0f, 0.0f, 1.0f),
					lowThreshY, highThreshY);
			}
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}

	if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
	{
		if (img.rows == 1)
		{
			flag1d = true;
			lowThreshY_right = lv;
			highThreshY_right = hv;
		}
		else
		{
			flag1d = false;
			lowThreshY_right = -1;
			highThreshY_right = 1;
			showThreshLow_right = lv;
			showThreshHigh_right = hv;
		}

		theApp.m_pMainWnd->GetDlgItem(IDC_STATIC_TITLE_RIGHT)->SetWindowTextW(_T("右图像: ")
			+ string2CString(filename + " " + to_string(img.cols) + "×" + to_string(img.rows)));
		printf("右图像: %s channel = %d\n", filename.c_str(), img.channels());
		patternRight.reset();
		patternRight.updateImg(img, true);
		for (int i = 0; i < 2; i++)
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			patternRight.RenderPattern();
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}
	PostMessageW(theApp.m_pMainWnd->m_hWnd, WM_MY_MESSAGE, 1, 1);
	//PostMessageW(theApp.m_pMainWnd->m_hWnd, WM_MY_MESSAGE, NULL, (LPARAM)((void*)&threshCombo));
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)						// this bool variable is initially set to true
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;		// reversed since y-coordinates range from bottom to top

	lastX = xpos;
	lastY = ypos;
	patternLeft.setMousePos(lastX, lastY);
	patternRight.setMousePos(lastX, lastY);
	//printf("lastX = %f lastY = %f\n", lastX, lastY);

	//pattern2.setMousePos(lastX, SCR_HEIGHT - 1 - lastY);

	//slider.processInput(xpos, ypos, xoffset);
	//slider2.processInput(xpos, ypos, xoffset);
	//slider3.processInput(xpos, ypos, xoffset);
	//slider4.processInput(xpos, ypos, xoffset);

	////camera.ProcessMouseMovement(xoffset, yoffset);

	//if (menu.enableShow)
	//{
	//	menu.selectTest(xpos, ypos);
	//	//printf("selected id = %d\n",menu.selectTest(xpos, ypos) );
	//}

	if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
	{
		patternLeft.queryImageCoord(lastX, lastY);
		if (syncLRFlag)
			patternRight.copyROI(&patternLeft);
		glm::vec3 rgb = patternLeft.queryValue(lastX, lastY);

		/*if (patternLeft.isSelectingRoi)
		{
			patternLeft.updateRoi(lastX, lastY);
		}*/
		//grayL = rgb;
		//printf("Left R %.5f  G %.5f  B %.5f\n", rgb[0], rgb[1], rgb[2]);*/
	}
	if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
	{
		patternRight.queryImageCoord(lastX, lastY);
		if (syncLRFlag)
			patternLeft.copyROI(&patternRight);
		glm::vec3 rgb = patternRight.queryValue(lastX, lastY);
		/*grayR = rgb;
		printf("Right R %.5f  G %.5f  B %.5f\n", rgb[0], rgb[1], rgb[2]);
		printf("delta = %f\n", grayR.x - grayL.x);*/
		/*if (patternRight.isSelectingRoi)
		{
			patternRight.updateRoi(lastX, lastY);
		}*/
	}	

	if (patternLeft.isTranslating)
	{
		patternLeft.updateOffset(xoffset, yoffset);
		if (syncLRFlag)
			patternRight.copyROI(&patternLeft);
	}
	if (patternRight.isTranslating)
	{
		patternRight.updateOffset(xoffset, yoffset);
		if(syncLRFlag)	
			patternLeft.copyROI(&patternRight);
	}
	updateRender(window);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	if (yoffset < 0)
	{
		if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
		{
			if (patternLeft.isSelectingRoi)
				return;
			patternLeft.zoomOut();
			if (syncLRFlag)
				patternRight.copyROI(&patternLeft);
		}
		else if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
		{
			if (patternRight.isSelectingRoi)
				return;
			patternRight.zoomOut();
			if (syncLRFlag)
				patternLeft.copyROI(&patternRight);
		}
	}
	else if (yoffset > 0)
	{
		if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
		{
			patternLeft.zoomIn();
			if (syncLRFlag)
				patternRight.copyROI(&patternLeft);
		}
		else if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
		{
			patternRight.zoomIn();
			if (syncLRFlag)
				patternLeft.copyROI(&patternRight);
		}
	}
	updateRender(window);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	//printf("key = %d\n", key);
	if (action == GLFW_PRESS && key == GLFW_KEY_C)
	{		
	}
	if (action == GLFW_PRESS && key == GLFW_KEY_S && mods == GLFW_MOD_CONTROL)
	{
	
	}
	updateRender(window);
}

void updateInfo(std::string LRflag, glm::vec2 pixel, glm::vec3 rgb)
{
	CString info;
	if (LRflag == "L") 
	{
		printf("Left    R %.9f  G %.9f  B %.9f\n", rgb[0], rgb[1], rgb[2]);
		info.Format(_T("左相片:\r\npos: [%d, %d]\r\nR: %f\r\nG %f\r\nB %f"), int(pixel.x), int(pixel.y), rgb.x, rgb.y, rgb.z);
		theApp.m_pMainWnd->GetDlgItem(IDC_INFO_LEFT)->SetWindowTextW(info);
	}
	else if (LRflag == "R")
	{
		printf("Right    R %.9f  G %.9f  B %.9f\n", rgb[0], rgb[1], rgb[2]);
		info.Format(_T("右相片:\r\npos: [%d, %d]\r\nR: %f\r\nG %f\r\nB %f"), int(pixel.x), int(pixel.y), rgb.x, rgb.y, rgb.z);
		theApp.m_pMainWnd->GetDlgItem(IDC_INFO_RIGHT)->SetWindowTextW(info);
	}
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{

	if (action == GLFW_PRESS)
	{
		if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
		{
			glm::vec3 rgb = patternLeft.queryValue(lastX, lastY);
			grayL = rgb;
			updateInfo("L", patternLeft.selectPixel, rgb);

			if (syncLRFlag)
			{
				rgb = patternRight.queryPixelValue(patternLeft.selectPixel);
				grayR = rgb;
				updateInfo("R", patternLeft.selectPixel, rgb);
			}			
		}
		if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
		{
			glm::vec3 rgb = patternRight.queryValue(lastX, lastY);
			grayR = rgb;
			printf("delta = %f\n", grayR.x - grayL.x);
			updateInfo("R", patternRight.selectPixel, rgb);

			if (syncLRFlag)
			{
				rgb = patternLeft.queryPixelValue(patternRight.selectPixel);
				grayL = rgb;
				updateInfo("L", patternRight.selectPixel, rgb);
			}
		}

		glm::vec2 ptLeft = patternLeft.queryImageCoord(lastX, lastY);
		glm::vec2 ptRight = patternRight.queryImageCoord(lastX, lastY);
		
		int cn = 15;

		switch (button)
		{
		case GLFW_MOUSE_BUTTON_LEFT:
			if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
			{
				std::vector<double> tmpData;
				for (int i = 0; i < dataPrjMap.size(); i++)
				{
					tmpData.push_back(dataPrjMap[i].at<float>(ptLeft.y, ptLeft.x));
				}
				if (!tmpData.empty())
				{
					MyFuncShow mf(tmpData, MC_GREEN);
					imshow("mf", mf.UpdateShow());
					cv::waitKey(0);
				}
			
			}
			else if(patternRight.isInWindow(glm::vec2(lastX, lastY)))
			{
			}	
			/*if (slider.isAdjacent)
				slider.isSelected = true;
			if (slider2.isAdjacent)
				slider2.isSelected = true;
			if (slider3.isAdjacent)
				slider3.isSelected = true;
			if (slider4.isAdjacent)
				slider4.isSelected = true;*/
			break;
		//case GLFW_MOUSE_BUTTON_MIDDLE:
		//	if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
		//	{
		//		patternLeft.initSelectRoi(lastX, lastY);
		//	}
		//	if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
		//	{
		//		patternRight.initSelectRoi(lastX, lastY);
		//	}
		//	break;
		case GLFW_MOUSE_BUTTON_RIGHT:
			if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
			{
				if (!patternLeft.isTranslating)
				{
					patternLeft.isTranslating = true;
				}
			}
			if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
			{
				if (!patternRight.isTranslating)
				{
					patternRight.isTranslating = true;
				}
			}
			//menu.reset();
			/*menu.enableShow = true;
			menu.setPos(lastX, lastY);*/
			break;
		default:
			break;
		}
	}
	if (action == GLFW_RELEASE)
	{
		glm::mat4 r_model = glm::mat4();// = sphere.rotateState;//  glm::mat4();
		cv::Mat mask, mask0;
		int selectId = 0;
		switch (button)
		{
		case GLFW_MOUSE_BUTTON_LEFT:
			//if (menu.enableShow)
			//{				
			//}
			//menu.reset();

			
			//slider.isSelected = false;
			//slider2.isSelected = false;
			//slider3.isSelected = false;
			//slider4.isSelected = false;
			
			//pattern2.isTranslating = false;
			break;
		//case GLFW_MOUSE_BUTTON_MIDDLE:
		//	if (patternLeft.isSelectingRoi)
		//	{
		//		patternLeft.isSelectingRoi = false;
		//		string savepath = patternLeft.locFolder;
		//		int flag1 = savepath.find_last_of('\\');
		//		int flag2 = savepath.find_last_of('/');
		//		flag1 = std::max(flag1, flag2);
		//		printf("%d %d %d %d\n", savepath.length(), flag1, savepath.find_last_of('\\'), savepath.find_last_of('/'));
		//		cout << "save = " << savepath.substr(0, flag1) << endl;
		//		imwrite(savepath.substr(0, flag1) + "\\roi.bmp", patternLeft.roiMask);
		//		patternRight.updateImg(patternLeft.roiMask);
		//	}
		//	if (patternRight.isSelectingRoi)
		//	{
		//		patternRight.isSelectingRoi = false;
		//		string savepath = CString2string(selectFolder());
		//		imwrite(savepath + "\\roi.bmp", patternRight.roiMask);
		//	}
		//	break;
		case GLFW_MOUSE_BUTTON_RIGHT:
			patternLeft.isTranslating = false;
			patternRight.isTranslating = false;
			break;
		default:
			break;
		}
	}
	updateRender(window);
	return;
}


void CEpipolarTestDlg::OnShowWindow(BOOL bShow, UINT nStatus)
{
	CDialogEx::OnShowWindow(bShow, nStatus);

	CRect showRect;
	(this->GetDlgItem(IDC_PIC_SHOW))->GetWindowRect(&showRect);
	int cntMonitors = 0;
	GLFWmonitor** pMonitor = glfwGetMonitors(&cntMonitors);
	printf("cntMonitors = %d\n", cntMonitors);

	window = glfwCreateWindowEx(showRect.Width(), showRect.Height(), "Breakout", NULL, NULL,
		(int)((this->GetDlgItem(IDC_PIC_SHOW))->m_hWnd));

	SCR_WIDTH = showRect.Width();
	SCR_HEIGHT = showRect.Height();

	lastX = SCR_WIDTH / 2.0f;
	lastY = SCR_HEIGHT / 2.0f;

	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetDropCallback(window, drop_callback);

	//openGL生产商及版本
	const char* vendorName = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
	const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
	printf("OpenGL实现的版本号：%s\n", version);

	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	printf("GL Vendor : %s\n", vendor);
	printf("GL Renderer : %s\n", renderer);
	printf("GL Version(string) : %s\n", version);
	printf("GL Version(integer) : %d.%d\n", major, minor);
	printf("GLSL Version : %s\n", glslVersion);
	glewInit();

	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// OpenGL要求所有的纹理都是4字节对齐的，即纹理的大小永远是4字节的倍数
	// 通过将纹理解压对齐参数设为1，这样才能确保不会有对齐问题。
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	GLint internalFormat = GL_R32F;
	GLint max_size;
	glGetInternalformativ(GL_TEXTURE_2D, internalFormat, GL_MAX_WIDTH, 1, &max_size);
	glGetInternalformativ(GL_TEXTURE_2D, internalFormat, GL_MAX_HEIGHT, 1, &max_size);

	std::string datapath = Config::get<std::string>("datapath");

	cv::Mat img0 = cv::Mat::zeros(100, 100, CV_8UC3);

	patternLeft.init(glm::ivec2(0, SCR_HEIGHT / 2), glm::ivec2(SCR_WIDTH / 2, SCR_HEIGHT / 2), 1, SCR_WIDTH, SCR_HEIGHT);
	patternLeft.loadData(img0);
	patternLeft.updateImg(img0);

	

	patternRight.init(glm::ivec2(SCR_WIDTH / 2, SCR_HEIGHT / 2), glm::ivec2(SCR_WIDTH / 2, SCR_HEIGHT / 2), 2, SCR_WIDTH, SCR_HEIGHT);
	patternRight.loadData(img0);
	patternRight.updateImg(img0);
	
	ele.init(SCR_WIDTH, SCR_HEIGHT);
	vecr.init(SCR_WIDTH, SCR_HEIGHT);

	//text.initWindow(SCR_WIDTH - SCR_WIDTH / 2, SCR_HEIGHT);
	//text.Load("E:\\MyClass\\fonts\\calibri.ttf", 32);

	//openglInitDone = true;
}

void updateRender(GLFWwindow *window)
{
	if (isSaving)
		return;

	patternLeft.queryImageCoord(lastX, lastY);
	patternRight.queryImageCoord(lastX, lastY);

	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	patternLeft.setNormalizeRange(showThreshLow, showThreshHigh);	// 归一化的上下限
	patternRight.setNormalizeRange(showThreshLow_right, showThreshHigh_right);	// 归一化的上下限
	
	if (!flag1d)
	{
		patternLeft.RenderPattern();
	}

	if (!pts2D.empty())
	{
		glViewport(0, SCR_HEIGHT / 2, SCR_WIDTH / 2, SCR_HEIGHT / 2);
		ele.RenderPtsUseCross2D(patternLeft.convert2RenderCS(pts2D), 0.01, GLM_RED);
	}
	patternRight.RenderPattern();

	
	if (!flag1d)
	{
		//printf("glViewport\n");
		glViewport(0, 0, SCR_WIDTH / 2, SCR_HEIGHT / 2);
		ele.setGraySeqTexture(1, patternLeft.texture);

		//printf("lowThreshY = %f\n", lowThreshY);
		//printf("highThreshY = %f\n", highThreshY);
		ele.RenderGraySeq(glm::vec2(patternLeft.xminTexture, patternLeft.selectPixelInTexture.y),
			glm::vec2(patternLeft.xmaxTexture, patternLeft.selectPixelInTexture.y), glm::vec3(1.0f, 0.0f, 1.0f),
			lowThreshY, highThreshY);

		//float xpos = (patternLeft.selectPixelInTexture.x - patternLeft.xminTexture) / (patternLeft.xmaxTexture - patternLeft.xminTexture) *2.0f - 1.0f;
		//ele.RenderLine2D(glm::vec2(xpos, -1.0f), glm::vec2(xpos, 1.0f), GLM_GREEN);
	}
	else
	{
		glViewport(0, SCR_HEIGHT / 2, SCR_WIDTH / 2, SCR_HEIGHT / 2);
		vecr.setData(3, patternLeft.texture);
		vecr.render(glm::vec2(patternLeft.xminTexture, patternLeft.selectPixelInTexture.y),
			glm::vec2(patternLeft.xmaxTexture, patternLeft.selectPixelInTexture.y), glm::vec3(1.0f, 0.0f, 1.0f),
			lowThreshY + int((highThreshY - lowThreshY) *patternLeft.yminTexture), int(highThreshY * patternLeft.ymaxTexture));
	}

	glViewport(SCR_WIDTH / 2, 0, SCR_WIDTH / 2, SCR_HEIGHT / 2);
	ele.setGraySeqTexture(2, patternRight.texture);
	
	//if (false)
	/*	if (syncFlag)
	{
		if (matchedPtCol.empty())
			return;
		float col = matchedPtCol.at<float>(int(patternLeft.selectPixel.y), int(patternLeft.selectPixel.x));
		float xspan = patternLeft.xmaxTexture - patternLeft.xminTexture;
		if (col != -1)
		{
			matchedPixel = glm::vec2(col, patternLeft.selectPixel.y);
			matchedPixelInTexture = glm::vec2(col / patternLeft.cols, patternLeft.selectPixelInTexture.y);
			patternRight.scale = patternLeft.scale;
			patternRight.scaleLevel = patternLeft.scaleLevel;
			patternRight.offset = glm::vec2(matchedPixelInTexture.x - xspan * 0.5f, patternLeft.yminTexture);
			patternRight.updateRenderParas();
		}
		ele.RenderGraySeq(glm::vec2(matchedPixelInTexture.x - xspan*0.5f, matchedPixelInTexture.y),
			glm::vec2(matchedPixelInTexture.x + xspan * 0.5f, matchedPixelInTexture.y), glm::vec3(1.0f, 0.0f, 0.0f),
			lowThreshY_right, highThreshY_right);
		ele.RenderLine2D(glm::vec2(0, -1.0f), glm::vec2(0, 1.0f), GLM_GREEN);
	}
	else*/
	//{		
		ele.RenderGraySeq(glm::vec2(patternRight.xminTexture, patternRight.selectPixelInTexture.y),
			glm::vec2(patternRight.xmaxTexture, patternRight.selectPixelInTexture.y), glm::vec3(1.0f, 0.0f, 0.0f),
			lowThreshY_right, highThreshY_right);
	//}

	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT / 2);
	ele.RenderLine2D(glm::vec2(-1.0f, 1.0f), glm::vec2(1.0f, 1.0f), GLM_YELLOW);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, SCR_WIDTH / 2, SCR_HEIGHT);
	ele.RenderLine2D(glm::vec2(1.0f, -1.0f), glm::vec2(1.0f, 1.0f), GLM_YELLOW);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

void CEpipolarTestDlg::OnBnClickedButtonApply()
{
	//printf("Apply\n");
	UpdateData(TRUE);
	showThreshLow = m_edit_show_thresh_low;
	showThreshHigh = m_edit_show_thresh_high;
	lowThreshY = m_edit_thresh_low;
	highThreshY = m_edit_thresh_high;

	showThreshLow_right = m_edit_show_thresh_low_right;
	showThreshHigh_right = m_edit_show_thresh_high_right;
	lowThreshY_right = m_edit_thresh_low_right;
	highThreshY_right = m_edit_thresh_high_right;
	UpdateData(FALSE);
}

BOOL CEpipolarTestDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO:  在此添加专用代码和/或调用基类
	if (pMsg->message == WM_MESSAGE_UPDATE_POS) 
	{
		printf("hello1\n");
		return CDialog::PreTranslateMessage(pMsg);
	}
	if (pMsg->message == WM_KEYDOWN && pMsg->wParam == VK_ESCAPE)		return TRUE;
	if (pMsg->message == WM_KEYDOWN && pMsg->wParam == VK_RETURN)		return TRUE;

	// 按住Ctrl+S
	if (pMsg->message == WM_KEYDOWN && pMsg->wParam == 83 && (GetAsyncKeyState(VK_CONTROL) < 0))
	{
		printf("save!\n");
		isSaving = true;

		GLuint textureSave = generateTextureRGB(SCR_WIDTH / 2, SCR_HEIGHT / 2);
		FBOManager fbo(GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureSave);
		fbo.bindFBO();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, SCR_WIDTH / 2, SCR_HEIGHT / 2);

		if (patternLeft.isInWindow(glm::vec2(lastX, lastY)))
		{
			patternLeft.shader.use();
			patternLeft.RenderPattern(false);
		}

		if (patternRight.isInWindow(glm::vec2(lastX, lastY)))
		{
			patternRight.shader.use();
			patternRight.RenderPattern(false);
		}

		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		cv::Mat imgSave = cv::Mat::zeros(SCR_HEIGHT / 2, SCR_WIDTH / 2, CV_8UC3);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureSave);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, imgSave.data);

		if (!imgSave.empty())
		{
			cvtColor(imgSave, imgSave, CV_RGB2BGR);
			flip(imgSave, imgSave, 0);
			//imshow("imgLeft", imgLeft);
			//cv::waitKey(0);

			std::string filename = selectSavePath("图片", "bmp", "image");
			if (filename.find(".bmp") == -1)
				filename += ".bmp";
			cv::imwrite(filename, imgSave);
		}

		fbo.unbindFBO();
		glBindTexture(GL_TEXTURE_2D, 0);
		isSaving = false;
	}
	return CDialog::PreTranslateMessage(pMsg);
}

//void CEpipolarTestDlg::OnCancel() { this->close; }
void CEpipolarTestDlg::OnOK(){}
//void CEpipolarTestDlg::OnClose(){OnCancel();}
void CEpipolarTestDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}
void CEpipolarTestDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}
HCURSOR CEpipolarTestDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

LRESULT CEpipolarTestDlg::OnUpdatePOS(WPARAM wParam, LPARAM lParam)
{
	return 1;
}

void CEpipolarTestDlg::OnTimer(UINT_PTR nIDEvent)
{
	CDialogEx::OnTimer(nIDEvent);
}


BOOL CEpipolarTestDlg::OnCopyData(CWnd* pWnd, COPYDATASTRUCT* pCopyDataStruct)
{
	return CDialogEx::OnCopyData(pWnd, pCopyDataStruct);
}


void CEpipolarTestDlg::OnBnClickedCheckSync()
{
	syncFlag = IsDlgButtonChecked(IDC_CHECK_SYNC);
	if (syncLRFlag == true)
	{
		syncLRFlag = false;
		((CButton*)GetDlgItem(IDC_CHECK_SYNC_LEFT_RIGHT))->SetCheck(syncLRFlag);
	}
}


void CEpipolarTestDlg::OnBnClickedCheckSyncLeftRight()
{
	syncLRFlag = IsDlgButtonChecked(IDC_CHECK_SYNC_LEFT_RIGHT);
	if (syncFlag == true)
	{
		syncFlag = false;
		((CButton*)GetDlgItem(IDC_CHECK_SYNC))->SetCheck(syncFlag);
	}
}


void CEpipolarTestDlg::OnBnClickedCheckPseuLeft()
{
	patternLeft.usePseudoColor = IsDlgButtonChecked(IDC_CHECK_PSEU_LEFT);
	updateRender(window);
}


void CEpipolarTestDlg::OnBnClickedCheckPseuRight()
{
	patternRight.usePseudoColor = IsDlgButtonChecked(IDC_CHECK_PSEU_RIGHT);
	updateRender(window);
}


void CEpipolarTestDlg::OnEnChangeEditShowThreshLow()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CEpipolarTestDlg::OnEnChangeEditShowThreshHigh()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CEpipolarTestDlg::OnBnClickedButtonCopytoright()
{
	UpdateData(TRUE);
	showThreshLow = m_edit_show_thresh_low;
	showThreshHigh = m_edit_show_thresh_high;
	lowThreshY = m_edit_thresh_low;
	highThreshY = m_edit_thresh_high;

	showThreshLow_right = m_edit_show_thresh_low_right = showThreshLow;
	showThreshHigh_right = m_edit_show_thresh_high_right = showThreshHigh;
	lowThreshY_right = m_edit_thresh_low_right = lowThreshY;
	highThreshY_right = m_edit_thresh_high_right = highThreshY;
	UpdateData(FALSE);
}


void CEpipolarTestDlg::OnBnClickedButtonCopytoleft()
{
	UpdateData(TRUE);
	showThreshLow_right = m_edit_show_thresh_low_right;
	showThreshHigh_right = m_edit_show_thresh_high_right;
	lowThreshY_right = m_edit_thresh_low_right;
	highThreshY_right = m_edit_thresh_high_right;

	showThreshLow = m_edit_show_thresh_low = showThreshLow_right;
	showThreshHigh = m_edit_show_thresh_high = showThreshHigh_right;
	lowThreshY = m_edit_thresh_low = lowThreshY_right;
	highThreshY = m_edit_thresh_high = highThreshY_right;
	UpdateData(FALSE);
}
