#ifndef IMAGE_RENDERER_H
#define IMAGE_RENDERER_H

#include<opencv2/opencv.hpp>

#include<map>
#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif
#include<glm/glm.hpp>
#include<vector>
#include"Shader.h"
#include<condition_variable>
#include<mutex>

// A renderer class for renderering text displayed by a font loaded using the 
// Freetype library. A single font is loaded, processed into a list of Character
// items for later rendering.
class ImageRenderer
{
public:
	// Shader used for text rendering 
	Shader shader;

	void setFileInfo(std::string path);
	std::string filepath;
	std::string locFolder;
	std::string filename;
	std::vector<cv::Mat> rgb;

	// Constructor
	ImageRenderer();
	void init(glm::ivec2 pos, glm::ivec2 size, GLint textureId, int screenW = 0, int screenH = 0);
	~ImageRenderer();

	void copyROI(ImageRenderer *render);

	void setRenderUsePseudoColor(bool flag);

	// 输入图像点像素坐标，
	// 返回渲染窗口坐标 [-1,1]
	glm::vec2 convert2RenderCS(glm::vec2 uv);
	std::vector<glm::vec2> convert2RenderCS(std::vector<glm::vec2> uv);

	// 返回图像的像素坐标
	glm::vec2 queryImageCoord(const float &xpos, const float &ypos);

	void loadData(const cv::Mat &data);
	void updateImg(const cv::Mat &img, const bool &flipY = false);

	void setMousePos(const float &posx, const float &posy);
	void RenderPattern(bool updateViewport = true);

	void setNormalizeRange(const float &low, const float &high);

	void zoomIn();
	void zoomOut();

	void updateOffset(const float &offsetX, const float &offsetY);

	glm::vec3 queryValue(const float &x, const float &y);
	glm::vec3 queryPixelValue(glm::vec2 pixel);

	GLuint texture;
	bool isInWindow(const glm::vec2 &pos, bool updateActiveStatus = false);
	bool isTranslating = false;
	bool render8bitImage = true;

	void updateViewPort()
	{
		glViewport(pos.x, pos.y, width, height);
	}

	void updateRenderParas();


	bool active = false;
	void reset();			// 清空

	cv::Mat imgGray;		// 灰度图像

	bool usePseudoColor = false;

	bool isSelectingRoi = false;
	void initSelectRoi(double xpos, double ypos);
	void updateRoi(double xpos, double ypos);
	cv::Rect roi;
	cv::Mat roiMask;


#ifdef RENDER_USE_OPENGL
	void updateTargets(const std::vector<glm::vec2> &ts)
	{
		targets.clear();
		flagsTarget.clear();
		targets.assign(ts.begin(), ts.end());
		flagsTarget.assign(ts.size(), 1);
	}

	// 更新选中的标志点
	void updateSelectTargets(const glm::vec2 &startPt, const glm::vec2 &endPt);
	void deleteSelectedTargets();
	void resetSelectTargets();

	std::vector<glm::vec2> getNormalizedTargetCoords()
	{
		std::vector<glm::vec2> result(targets.size());
		for (int i = 0; i < result.size(); i++)
			result[i] = convert2RenderCS(targets[i]);
		return result;
	}

	std::vector<glm::vec2> targets;		// 标志点
	std::vector<int> flagsTarget;		// 标志点状态标志位

	bool showFlag = false;
#endif
	int rowMin, rowMax;
	int colMin, colMax;
	float xminTexture, xmaxTexture;
	float yminTexture, ymaxTexture;
	glm::vec2 selectPixel;
	glm::vec2 selectPixelInTexture = glm::vec2(0);

	float scale = 1.0;
	int scaleLevel = 0;			// 图片的放大倍率   实际放大的倍数是pow(1.25, scale) 对应图片的面积大小就是pow(0.8, scale);
	glm::vec2 offset = glm::vec2(0, 0);

	int rows = 0, cols = 0;		// 图像分辨率 不会随着滚轮放缩图像而改变
	cv::Mat pic;

private:
	void updateRoi();

	cv::Point roiPt1, roiPt2;

	glm::vec2 mousePos;

	// RenderState
	GLuint VAO, VBO, EBO;

	// 渲染窗口的尺寸
	int screenWidth, screenHeight;

	int width, height;		// 窗口的尺寸
	glm::ivec2 pos;			// 窗口的左上角坐标
	int textureId = 0;
};

#endif