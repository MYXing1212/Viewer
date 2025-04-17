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
	
	// ������
	void RenderPts(glm::vec3 pts[], int cnt, const float &pointSize, glm::vec3 color);
	void RenderPts(std::vector<glm::vec3> pts, const float &pointSize, glm::vec3 color, bool enableBlend = false);
	void RenderPtsUseCross2D(std::vector<glm::vec2> pts, const float &len, glm::vec3 color, bool enableBlend = false);
	void RenderPtsUseCross2D(std::vector<glm::vec2> pts, std::vector<int> ptsFlags, 
		const float &len, const float &width, 
		glm::vec3 color, bool enableBlend = false);
	
	// ���ƶ�ά��
	void RenderLoop2D(std::vector<glm::vec2> pts, glm::vec3 color, bool enableBlend = false);

	void RenderSphere(float radius, glm::vec3 pos, glm::vec3 color);
	// �뾶Ϊ��һ��ֵ ��[-1,1]����ϵ�� posҲ��[-1,1]����ϵ��
	void RenderSphere2D(float radius, glm::vec2 pos, glm::vec3 color);
	void RenderLine(glm::vec3 start, glm::vec3 end, glm::vec3 color);
	void RenderHalfLine(glm::vec3 start, glm::vec3 dir, float length, glm::vec3 color);


	void RenderLine2D(glm::vec2 start, glm::vec2 end, glm::vec3 color);

	void RenderCircle(float radius, glm::vec3 color, bool bFilled = true);
	void RenderCircle(float radius, glm::vec3 cen, glm::vec3 norm, glm::vec3 color, bool bFilled = true, bool dashed = false);
	void RenderCylinder(float radius, float h, glm::vec3 color);
	void RenderCone(float radius, float h, glm::vec3 color);

	// ������� �����յ�  �����������
	void RenderPlane2D(glm::vec2 start, glm::vec2 end, float width, glm::vec3 color, bool bFilled = true);
	void RenderPlane2D(glm::vec2 tl, glm::vec2 br, glm::vec3 color, bool bFilled = true);

	void RenderPlane(glm::vec2 topleft, glm::vec2 size, glm::vec3 color);
	void RenderPlane(glm::vec3 cen, glm::vec3 norm, glm::vec2 size, glm::vec3 color, float alpha = 1.0);
	void RenderPlane(glm::vec3 cen, glm::vec3 u, glm::vec3 v, glm::vec2 size, glm::vec3 color, float alpha = 1.0f);
	
	// �ռ�С��Ԫ��4����˳�����ӳ�һ��С��Ƭ
	void RenderPlane(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 D, glm::vec3 color, float alpha = 1.0f);

	void RenderBackground();

	void init(int screenWidth, int screenHeight);


	// ��������׶̨
	// ���ذ�Χ�еĴ�С 6�����ֱ�Ϊxmin, xmax, ymin, ymax, zmin, zmax
	cv::Vec6f RenderFrustum(glm::vec3 start, glm::vec3 dir, float nearPlane, 
		float farPlane, float angleWidth, float angleHeight, glm::vec3 color, float alpha = 1.0f);

		// �Ե���Ϊxoyƽ�� �����е�Ϊ����ԭ��
	void RenderCube(glm::vec3 pos, glm::vec3 size, glm::vec3 color, bool bFilled = true);
	void RenderBoundingBox(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, glm::vec3 color);
	void RenderSquareColumn(glm::vec3 pos, float height, glm::vec2 size);

	void RenderGraySeq(const std::vector<float> grays, glm::vec3 color, 
		const float &lowThresh = 0, const float &highThresh = 255.0f);

	void RenderGraySeq(const glm::vec2 &startPt, const glm::vec2 &endPt, glm::vec3 color,
		const float &lowThresh = 0, const float &highThresh = 255.0f);


	// �˸����˳���� �϶������Ͻǵ�-���Ͻǵ�-���½ǵ�-���½ǵ�  �µ������Ͻǵ�-���Ͻǵ�-���½ǵ�-���½ǵ�
	void RenderBoundingBox(std::vector<glm::vec3> vertex, glm::vec3 color);

	void setPrjViewMatrix(glm::mat4 projection, glm::mat4 view);
	void setModelMatrix(glm::mat4 model, bool updateModel = true);

	void setScreenRatio(const float &ratio)
	{
		screenRatio = ratio;
	}

	// �������λ�ã����ع�һ����������
	glm::vec2 getNormalizedCoord(const float &xpos, const float &ypos);

	// ��ձ���ɫ Ĭ��Ϊ��ɫ
	void clear(glm::vec3 bkColor = glm::vec3(0.0f));

	void setPointSize(const GLfloat &size)
	{
		shader.use();
		shader.setFloat("pointSize", size);
	}

	glm::mat4 getTransformMatrix();
	
	glm::mat4 model, projection, view;

	std::vector<glm::vec2> pts2D_record;				// ��¼��2ά�㼯

	// ���û��ƻҶȱ仯���߲ο��������id
	void setGraySeqTexture(int GL_TEXTURE_id, int texture);

	void startSelectROI(const glm::vec2 &startPt);
	void updateROI(const glm::vec2 &pt);
	void endSelectROI(const glm::vec2 &endPt);

	bool isSelectingROI = false;
	glm::vec2 roiStart;									// roi���
	glm::vec2 roiEnd;									// roi�յ�

private:
	float screenRatio = 1.0f;
	float screenWidth = 0, screenHeight = 0;

	// RenderState
	GLuint VAO = -1, VBO=-1;
	GLfloat window_width, window_height;
	
	void initLineData();								// ��ʼ������ֱ������
	void initCircleData(float radius);					// ��ʼ��Բ��������
	void initSphereData(float radius);					// ��ʼ�����������
	void initCylinderData(float radius, float h);		// ��ʼ��Բ������
	void initConeData(float radius, float h);			// ��ʼ��Բ������

	float R_circle;
	float R_sphere;

	float R_cylinder;									// Բ���İ뾶
	float h_cylinder;									// Բ���ĸ�
	
	float R_cone;										// Բ׶�İ뾶
	float h_cone;

	int graySeqTextureId = 0;

	cv::Mat sphereData;
	cv::Mat cylinderData;
	cv::Mat circleData;
	cv::Mat coneData;
	cv::Mat lineData;									// ����ֱ�ߵ�����
};

#endif