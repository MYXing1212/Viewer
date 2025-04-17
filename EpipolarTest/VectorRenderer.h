#pragma once

#include<opencv2/opencv.hpp>

#include<map>

#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif
#include<glm/glm.hpp>
#include<vector>
//#include"color.h"
#include"Shader.h"
#include<glm/ext.hpp>


class VectorRenderer
{
public:
	// Shader used for text rendering 
	Shader shader;
	
	// Constructor
	VectorRenderer();
	~VectorRenderer();

	void init(int screenWidth, int screenHeight);

	void render(const std::vector<float> grays, glm::vec3 color, 
		const float &lowThresh = 0, const float &highThresh = 255.0f);

	void render(const glm::vec2 &startPt, const glm::vec2 &endPt, glm::vec3 color,
		const float &lowThresh = 0, const float &highThresh = 255.0f);

	void setScreenRatio(const float &ratio)
	{
		screenRatio = ratio;
	}

	// 给出鼠标位置，返回归一化后的坐标点
	glm::vec2 getNormalizedCoord(const float &xpos, const float &ypos);

	// 清空背景色 默认为黑色
	void clear(glm::vec3 bkColor = glm::vec3(0.0f));
	
	// 设置绘制灰度变化曲线参考的纹理的id
	void setData(int GL_TEXTURE_id, int texture);

private:
	float screenRatio = 1.0f;
	float screenWidth = 0, screenHeight = 0;

	// RenderState
	GLuint VAO = -1, VBO=-1;
	GLfloat window_width, window_height;
	
	void initLineData();								// 初始化绘制直线数据

	int graySeqTextureId = 0;

	cv::Mat lineData;									// 绘制直线的数据
};

