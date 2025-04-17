#ifndef ELEMENT_GEO_H
#define ELEMENT_GEO_H

#include<opencv2/opencv.hpp>

#include<map>

#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif
#include<glm/glm.hpp>
#include<vector>
#include"TextRenderer.h"
//#include"color.h"
#include"Shader.h"
#include<glm/ext.hpp>

cv::Point3f getSpherePoint(float u, float v);

class ElementGeoRenderer
{
public:
	// Shader used for text rendering 
	Shader shader;
	
	// Constructor
	ElementGeoRenderer();
	~ElementGeoRenderer();	
	
	// 绘制球
	void RenderPts(glm::vec3 pts[], int cnt, const float &pointSize, glm::vec3 color);
	void RenderPts(std::vector<glm::vec3> pts, const float &pointSize, glm::vec3 color, bool enableBlend = false);
	void RenderPtsUseCross2D(std::vector<glm::vec2> pts, const float &len, glm::vec3 color, bool enableBlend = false);
	void RenderPtsUseCross2D(std::vector<glm::vec2> pts, std::vector<int> ptsFlags, 
		const float &len, const float &width, 
		glm::vec3 color, bool enableBlend = false);
	
	// 绘制二维环
	void RenderLoop2D(std::vector<glm::vec2> pts, glm::vec3 color, bool enableBlend = false);

	void RenderSphere(float radius, glm::vec3 pos, glm::vec3 color);
	// 半径为归一化值 在[-1,1]坐标系下 pos也在[-1,1]坐标系下
	void RenderSphere2D(float radius, glm::vec2 pos, glm::vec3 color);
	void RenderLine(glm::vec3 start, glm::vec3 end, glm::vec3 color);
	void RenderHalfLine(glm::vec3 start, glm::vec3 dir, float length, glm::vec3 color);


	void RenderLine2D(glm::vec2 start, glm::vec2 end, glm::vec3 color);

	void RenderCircle(float radius, glm::vec3 color, bool bFilled = true);
	void RenderCircle(float radius, glm::vec3 cen, glm::vec3 norm, glm::vec3 color, bool bFilled = true, bool dashed = false);
	void RenderCylinder(float radius, float h, glm::vec3 color);
	void RenderCone(float radius, float h, glm::vec3 color);

	// 给出起点 给出终点  给出条带宽度
	void RenderPlane2D(glm::vec2 start, glm::vec2 end, float width, glm::vec3 color, bool bFilled = true);
	void RenderPlane2D(glm::vec2 tl, glm::vec2 br, glm::vec3 color, bool bFilled = true);

	void RenderPlane(glm::vec2 topleft, glm::vec2 size, glm::vec3 color);
	void RenderPlane(glm::vec3 cen, glm::vec3 norm, glm::vec2 size, glm::vec3 color, float alpha = 1.0);
	void RenderPlane(glm::vec3 cen, glm::vec3 u, glm::vec3 v, glm::vec2 size, glm::vec3 color, float alpha = 1.0f);
	
	// 空间小面元的4个点顺次连接成一个小面片
	void RenderPlane(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 D, glm::vec3 color, float alpha = 1.0f);

	void RenderBackground();

	void init(int screenWidth, int screenHeight);


	// 绘制四棱锥台
	// 返回包围盒的大小 6个数分别为xmin, xmax, ymin, ymax, zmin, zmax
	cv::Vec6f RenderFrustum(glm::vec3 start, glm::vec3 dir, float nearPlane, 
		float farPlane, float angleWidth, float angleHeight, glm::vec3 color, float alpha = 1.0f);

		// 以底面为xoy平面 底面中点为坐标原点
	void RenderCube(glm::vec3 pos, glm::vec3 size, glm::vec3 color, bool bFilled = true);
	void RenderBoundingBox(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, glm::vec3 color);
	void RenderSquareColumn(glm::vec3 pos, float height, glm::vec2 size);

	void RenderGraySeq(const std::vector<float> grays, glm::vec3 color, 
		const float &lowThresh = 0, const float &highThresh = 255.0f);

	void RenderGraySeq(const glm::vec2 &startPt, const glm::vec2 &endPt, glm::vec3 color,
		const float &lowThresh = 0, const float &highThresh = 255.0f);


	// 八个点的顺序是 上顶面左上角点-右上角点-右下角点-左下角点  下底面左上角点-右上角点-右下角点-左下角点
	void RenderBoundingBox(std::vector<glm::vec3> vertex, glm::vec3 color);

	void setPrjViewMatrix(glm::mat4 projection, glm::mat4 view);
	void setModelMatrix(glm::mat4 model, bool updateModel = true);

	void setScreenRatio(const float &ratio)
	{
		screenRatio = ratio;
	}

	// 给出鼠标位置，返回归一化后的坐标点
	glm::vec2 getNormalizedCoord(const float &xpos, const float &ypos);

	// 清空背景色 默认为黑色
	void clear(glm::vec3 bkColor = glm::vec3(0.0f));

	void setPointSize(const GLfloat &size)
	{
		shader.use();
		shader.setFloat("pointSize", size);
	}

	glm::mat4 getTransformMatrix();
	
	glm::mat4 model, projection, view;

	std::vector<glm::vec2> pts2D_record;				// 记录的2维点集

	// 设置绘制灰度变化曲线参考的纹理的id
	void setGraySeqTexture(int GL_TEXTURE_id, int texture);

	void startSelectROI(const glm::vec2 &startPt);
	void updateROI(const glm::vec2 &pt);
	void endSelectROI(const glm::vec2 &endPt);

	bool isSelectingROI = false;
	glm::vec2 roiStart;									// roi起点
	glm::vec2 roiEnd;									// roi终点

private:
	float screenRatio = 1.0f;
	float screenWidth = 0, screenHeight = 0;

	// RenderState
	GLuint VAO = -1, VBO=-1;
	GLfloat window_width, window_height;
	
	void initLineData();								// 初始化绘制直线数据
	void initCircleData(float radius);					// 初始化圆绘制数据
	void initSphereData(float radius);					// 初始化球绘制数据
	void initCylinderData(float radius, float h);		// 初始化圆柱数据
	void initConeData(float radius, float h);			// 初始化圆柱数据

	float R_circle;
	float R_sphere;

	float R_cylinder;									// 圆柱的半径
	float h_cylinder;									// 圆柱的高
	
	float R_cone;										// 圆锥的半径
	float h_cone;

	int graySeqTextureId = 0;

	cv::Mat sphereData;
	cv::Mat cylinderData;
	cv::Mat circleData;
	cv::Mat coneData;
	cv::Mat lineData;									// 绘制直线的数据
};

#endif