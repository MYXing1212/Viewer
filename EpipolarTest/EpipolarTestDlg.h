
// EpipolarTestDlg.h : header file
//

#pragma once
#include<opencv2/opencv.hpp>
#include"MyImage.h"
#include"MyMatrix.h"
#include"MyPCP.h"

#include<GL/glew.h>
#ifdef USE_FREEGLUT
#include<GL/freeglut.h>
#elif defined USE_GLFW
#include<GLFW/glfw3.h>
#endif

#include"ImageRenderer.h"
#include"ElementGeoRenderer.h"

using namespace cv;
using namespace std;


// CEpipolarTestDlg dialog
class CEpipolarTestDlg : public CDialogEx
{
// Construction
public:
	CEpipolarTestDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_EPIPOLARTEST_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnLoad1();
	afx_msg void OnBnClickedBtnLoad2();

	cv::Mat pic1, pic2;
	int w, h;
	cv::Mat F;

	CString m_str_info;

	float m_edit_thresh_low_right;
	float m_edit_thresh_high_right;
	float m_edit_show_thresh_low_right;
	float m_edit_show_thresh_high_right;

	afx_msg void OnShowWindow(BOOL bShow, UINT nStatus);
	float m_edit_thresh_low;
	float m_edit_thresh_high;
	float m_edit_show_thresh_low;
	float m_edit_show_thresh_high;
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	//virtual void OnCancel();
	virtual void OnOK();
	//afx_msg void OnClose();

	afx_msg void OnBnClickedButtonApply();

	afx_msg LRESULT OnUpdatePOS(WPARAM wParam, LPARAM lParam);
	//afx_msg LRESULT OnUpdatePOS(WPARAM wParam, LPARAM lParam);
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg BOOL OnCopyData(CWnd* pWnd, COPYDATASTRUCT* pCopyDataStruct);
	afx_msg void OnBnClickedCheckSync();
	afx_msg void OnBnClickedCheckSyncLeftRight();
	afx_msg void OnBnClickedCheckPseuLeft();
	afx_msg void OnBnClickedCheckPseuRight();
	afx_msg void OnEnChangeEditShowThreshLow();
	afx_msg void OnEnChangeEditShowThreshHigh();

	afx_msg LRESULT OnMyMessage(WPARAM wParam, LPARAM lParam);
	afx_msg void OnBnClickedButtonCopytoright();
	afx_msg void OnBnClickedButtonCopytoleft();
};
