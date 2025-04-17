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

// ���ô������ε���ʼʱ���
void MyTime::SetBegin()
{
	duration = static_cast<double>(getTickCount());
}

// ���������ʱ����msΪ��λ
double MyTime::GetDuration()
{
	duration = static_cast<double>(getTickCount())-duration;
	duration /= getTickFrequency();// ����ʱ��,��sΪ��λ
	duration *= 1000.0;
	return duration;
}

// �ж��Ƿ�Ϊ����
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

//LONG64 MyTime::getTime()            // ��ȡCPU��ǰ��ȷʱ��
//{
//	LARGE_INTEGER litmp;
//	LONG64 QPart;
//	QueryPerformanceCounter(&litmp);     // ��ȡ��ǰʱ��
//	QPart = litmp.QuadPart;              // ��ȡlonglong������
//	return QPart;
//}

//void MyTime::SetStartPoint()
//{
//	end_time = 0;
//	start_time = getTime();
//}

//double MyTime::GetDurTime_us()     // �õ��㷨����ʱ�䣬��λus
//{
//	end_time = getTime();
//	if (start_time == 0){
//		cout << "��δ�趨��ʼʱ���!" << endl;
//		return 0; 
//	}
//	else
//		return (double)(end_time - start_time);
//}

// ��ȡ��ǰʱ�� ��ʽ 2015/6/11 22:46:24
CString MyTime::getTimeCString()
{
	CTime time = CTime::GetCurrentTime();
	return time.Format(_T("%Y/%m/%d %H:%M:%S "));
}

