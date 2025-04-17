#include"stdafx.h"
//#include"stdafx.h"
#include"MyTime.h"

using namespace cv;

MyTime::MyTime(void)
	:duration(0),
	start_time(0),
	end_time(0)
{
}

MyTime::MyTime(int h,int m,int s)
{
	hours = h;
	minutes = m;
	seconds = s;
}

MyTime MyTime::operator +(MyTime& time)
{
	int h,m,s;
	s=time.seconds+seconds;
	m=time.minutes+minutes+s/60;
	h=time.hours+m/60+hours;
	MyTime result(h,m%60,s%60);
	return result;
}

// 设置待测程序段的起始时间点
void MyTime::SetBegin()
{
	duration = static_cast<double>(getTickCount());
}

// 计算程序用时，以ms为单位
double MyTime::GetDuration()
{
	duration = static_cast<double>(getTickCount())-duration;
	duration /= getTickFrequency();// 运行时间,以s为单位
	duration *= 1000.0;
	return duration;
}

// 判断是否为闰年
bool MyTime::IfBissextile(int year)
{
	if(year % 400 == 0)
		return true;
	else
	{
		if((year % 4 == 0) && (year % 100 != 0))
			return true;
		else 
			return false;
	}
	return false;
}

//LONG64 MyTime::getTime()            // 获取CPU当前精确时间
//{
//	LARGE_INTEGER litmp;
//	LONG64 QPart;
//	QueryPerformanceCounter(&litmp);     // 获取当前时间
//	QPart = litmp.QuadPart;              // 获取longlong型数据
//	return QPart;
//}

//void MyTime::SetStartPoint()
//{
//	end_time = 0;
//	start_time = getTime();
//}

//double MyTime::GetDurTime_us()     // 得到算法运行时间，单位us
//{
//	end_time = getTime();
//	if (start_time == 0){
//		cout << "尚未设定开始时间点!" << endl;
//		return 0; 
//	}
//	else
//		return (double)(end_time - start_time);
//}

// 获取当前时间 格式 2015/6/11 22:46:24
CString MyTime::getTimeCString()
{
	CTime time = CTime::GetCurrentTime();
	return time.Format(_T("%Y/%m/%d %H:%M:%S "));
}

