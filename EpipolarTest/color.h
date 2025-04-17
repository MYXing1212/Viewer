#pragma once

//#include "windows.h"
#include<time.h>
#include<opencv2/opencv.hpp>
#ifdef USE_GLM
#include <glm/glm.hpp>
#endif

#define MC_BLACK			(cv::Scalar(0, 0, 0))
#define MC_WHITE			(cv::Scalar(255, 255, 255))

#define MC_RED4				(cv::Scalar(0, 0, 139))
#define MC_RED				(cv::Scalar(0, 0, 255))
#define MC_DEEP_PINK3		(cv::Scalar(118, 16, 205))
#define MC_PINK1			(cv::Scalar(197, 181, 255))
#define MC_ORANGE			(cv::Scalar(0, 165, 255))
#define MC_YELLOW			(cv::Scalar(0, 255, 255))
#define MC_PALE_GREEN		(cv::Scalar(152, 251, 152))
#define MC_GREEN			(cv::Scalar(0, 255, 0))
#define MC_FORESET_GREEN	(cv::Scalar(34, 139, 34))
#define MC_BLUE				(cv::Scalar(255, 0, 0))
#define MC_PURPLE			(cv::Scalar(240, 32, 160))
#define MC_DARK_ORCHID4		(cv::Scalar(139, 34, 104))
#define MC_GREY				(cv::Scalar(190, 190, 190))
#define MC_NAVY_BLUE		(cv::Scalar(128, 0, 0))
#define MC_CORNFLOWER_BLUE	(cv::Scalar(237, 149, 100))
#define MC_LIGHT_BLUE		(cv::Scalar(230,216, 173))				// 亮蓝色			LightBlue		173 216 230	#ADD8E6
#define MC_GREEN3			(cv::Scalar(0, 205, 0))
#define MC_DARK_SLATE_GRAY	(cv::Scalar(47, 79, 79))				// 深灰色
#define MC_WHITE_SMOKE		(cv::Scalar(245, 245, 245))				// 白烟
#define MC_DEEP_SKY_BLUE	(cv::Scalar(255, 191, 0))				// 天蓝色			DeepSkyBlue
#define MC_GRAY11			(cv::Scalar(28, 28, 28))				// 接近黑的深灰色		grey11			 28  28  28	#1C1C1C
#define MC_PALE_TURQUOISE	(cv::Scalar(238, 238, 175))				// 浅蓝色			PaleTurquoise	175 238 238
#define MC_GRAY41			(cv::Scalar(105, 105, 105))				// 深灰色			grey41			105 105 105	#696969
#define MC_GRAY31			(cv::Scalar(79, 79, 79))				// 深灰色			grey31			 79  79  79	#4F4F4F

#define MC_GRAY				(cv::Scalar(190, 190, 190))
#define MC_DODGER_BLUE		(cv::Scalar(255, 144, 30))
#define MC_CYAN				(cv::Scalar(255, 255, 0))

#ifdef USE_GLM
#define GLM_RED				(glm::vec3(1.0f, 0.0f, 0.0f))
#define GLM_GREEN			(glm::vec3(0.0f, 1.0f, 0.0f))
#define GLM_BLUE			(glm::vec3(0.0f, 0.0f, 1.0f))
#define GLM_BLACK			(glm::vec3(0.0f))
#define GLM_WHITE			(glm::vec3(1.0f))
#define GLM_YELLOW			(glm::vec3(1.0f, 1.0f, 0.0f))
#define GLM_GREY			(glm::vec3(0.745f, 0.745f, 0.745f))
#define GLM_PALE_TURQUOISE	(glm::vec3(0.68f, 0.93f, 0.93f))		// 浅蓝色			PaleTurquoise	175 238 238
#define GLM_DEEP_SKY_BLUE	(glm::vec3(0.0f, 0.75f, 1.0f))			// 天蓝色			DeepSkyBlue
#define GLM_NAVY_BLUE		(glm::vec3(0.0f, 0.0f, 0.5f))			
#define GLM_ORANGE			(glm::vec3(1.0f, 0.647f,0.0f))			// 橘黄色
#define GLM_LIGHT_GREY		(glm::vec3(241/255.0f, 241/255.0f, 241/255.0f))	// 亮灰色
#define GLM_LIGHT_BLUE		(glm::vec3(142/255.0f, 203/255.0f, 250/255.0f))	// 亮灰色
#endif

typedef cv::Scalar ColorList;

#define COLLIST_MAX 25

extern ColorList colorList[COLLIST_MAX];

// 获取随机颜色
cv::Scalar getRandomColor(int s = -1);

// 返回颜色条对应的颜色 ratio从0到1变化
cv::Scalar getSeqColor(float ratio);

cv::Scalar convertToColor_cv(glm::vec3 c);

#ifdef USE_GLM
// 获取伪彩色 输入scalar 为0~1 float
inline void getPseudoColor(const float &scalar, glm::vec3 &color)
{
	if (scalar < 0.5f)
		color.r = 0.0f;
	else if (scalar < 0.75f)
		color.r = 4.0f * (scalar - 0.5f);
	else
		color.r = 1.0f;

	if (scalar < 0.25f)
		color.g = 4.0f * scalar;
	else if (scalar < 0.75f)
		color.g = 1.0f;
	else
		color.g = 1.0f - 4.0f * (scalar - 0.75f);

	if (scalar < 0.25f)
		color.b = 1.0f;
	else if (scalar < 0.5f)
		color.b = 1.0f - 4.0f * (scalar - 0.25f);
	else
		color.b = 0.0f;

	color = glm::max(glm::vec3(0.0f), color);
	color = glm::min(glm::vec3(1.0f), color);
}
#endif
