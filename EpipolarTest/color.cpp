#if defined USE_PCH	// 是否使用预编译头
#include"stdafx.h"
#endif
#include "color.h"

ColorList colorList[COLLIST_MAX]=
{
	MC_RED4,
	MC_RED,
	MC_DEEP_PINK3,
	MC_PINK1,
	MC_ORANGE,
	MC_YELLOW,
	MC_PALE_GREEN,
	MC_GREEN,
	MC_FORESET_GREEN,
	MC_BLUE,
	MC_PURPLE,
	MC_DARK_ORCHID4,
	MC_GREY,
	MC_NAVY_BLUE	
};

// 获取随机颜色
cv::Scalar getRandomColor(int s/* = -1*/)
{
	static int m = 0;
	if(s >0)
		srand((unsigned)time(NULL) + s*12345 + m*1117);

	//cout << (unsigned)time(NULL) << endl;

	uchar r = rand() % 256;
	uchar g = rand() % 256;
	uchar b = rand() % 256;

	if(s>0)
		m = (m + 361) % 111717;

	return cv::Scalar(b, g, r);
}

// 返回颜色条对应的颜色 ratio从0到1变化
cv::Scalar getSeqColor(float ratio){
	if (ratio < 0 || ratio > 1){
		printf("color.cpp getSeqColor 输入参数有误 请输入0到1之间的浮点型数据！\n");
		return cv::Scalar::all(0);
	}

	float tempL = ratio * 10.0f;
	tempL = tempL - floor(tempL);
	float tempR = 1.0f - tempL;
	
	int index = (int)(floor(ratio * 10.0f));
	
	if (index < 10){
		int b = (int)(colorList[index + 1][0] * tempL + colorList[index][0] * tempR);
		int g = (int)(colorList[index + 1][1] * tempL + colorList[index][1] * tempR);
		int r = (int)(colorList[index + 1][2] * tempL + colorList[index][2] * tempR);
		return cv::Scalar(b, g, r);
	}
	else {
		return MC_PURPLE;
	}
}

cv::Scalar convertToColor_cv(glm::vec3 c)
{
	return cv::Scalar(
		255 * c.b,
		255 * c.g,
		255 * c.r,
		255
		);
}



