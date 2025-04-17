#define _AFXDLL
#pragma once
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
//#include<Windows.h>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<time.h>
#include<iostream>
#include<afx.h>

using namespace std;

class MyTime
{
public: //外部接口
	MyTime();
	MyTime(int h,int m,int s);

	void SetBegin();                  // 设置待测程序段的时间起点
	//void SetStartPoint();             // 设置起始时间点

	double GetDuration();             // 计算程序用时，单位ms
	//double GetDurTime_us();           // 计算算法耗时

	//LONG64 getTime();                 // 获得CPU当前精确时间

	bool IfBissextile(int year);      // 判断是否为闰年

	CString getTimeCString();		  // 获取现在的时间	

	MyTime operator+(MyTime& time);

	friend bool operator>(MyTime &t1, MyTime &t2)
	{
		if(t1.hours > t2.hours)
			return true;
		else if(t1.minutes > t2.minutes && t1.hours == t2.hours)
			return true;
		else if(t1.seconds > t2.seconds && t1.minutes == t2.minutes && t1.hours == t2.hours)
			return true;
		else 
			return false;
	}
	friend bool operator<(MyTime &t1, MyTime &t2)
	{
		return t2>t1;
	}
	friend bool operator==(MyTime &t1, MyTime &t2)
	{
		if(t1.hours == t2.hours && t1.minutes == t2.minutes && t1.seconds == t2.seconds)
			return true;
		else 
			return false;
	}

private:
	double duration;
	int hours,minutes,seconds;
	LONG64 start_time, end_time;
};

#ifndef _XSLEEP_H_
#define _XSLEEP_H_

void XSleep(int nWaitInMSecs);

#endif // _XSLEEP_H_