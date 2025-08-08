#pragma once

#include<opencv2/opencv.hpp>

#include<map>
#include<GL/glew.h>
#include<glm/glm.hpp>
#include<vector>
#include"Shader.h"
#include<condition_variable>
#include<mutex>

#define UNDEFINED_TEXTURE 999

// 图像块结构
struct ImageTile {
	GLuint textureID = UNDEFINED_TEXTURE;
	int x, y;           // 块坐标
	int width, height;  // 块尺寸
	bool loaded;
};

struct ProjectRange
{
	float minX, maxX;
	float minY, maxY;
	float xspan;
	float yspan;
};

enum FillStrategy
{
	FILL_WIDTH = 0,
	FILL_HEIGHT = 1,
};

// A renderer class for renderering text displayed by a font loaded using the 
// Freetype library. A single font is loaded, processed into a list of Character
// items for later rendering.
class BigImageRenderer
{
public:
	// Shader used for text rendering 
	Shader shader;

	void setFileInfo(std::string path);
	std::string filepath;
	std::string locFolder;
	std::string filename;
	std::vector<cv::Mat> rgb;

	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 modelMatrix;
	glm::vec2 viewPosition;
	float zoomLevel=1.0f;

	const int tileSize = 4096;
	//const int tileSize = 4096;
	std::vector<ImageTile> tiles;
	std::thread loaderThread;
	std::mutex tileMutex;
	std::atomic<bool> running;
	float imageRatio = 1.0f;
	float ratioWindow = 1.0f; // 窗口的横纵比

	ProjectRange scope;

	// Constructor
	BigImageRenderer();
	void init(glm::ivec2 pos, glm::ivec2 size, GLint textureId, int screenW = 0, int screenH = 0);
	~BigImageRenderer();

	void setRenderUsePseudoColor(bool flag);

	void updateProjectionMatrix();

	// 输入图像点像素坐标，
	// 返回渲染窗口坐标 [-1,1]
	glm::vec2 convert2RenderCS(glm::vec2 uv);
	std::vector<glm::vec2> convert2RenderCS(std::vector<glm::vec2> uv);
	void copyROI(BigImageRenderer* render);

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
	void createTiles();
	void loadVisibleTiles();
	void loadTile(ImageTile& tile);

	bool active = false;
	void reset();			// 清空

	cv::Mat imgGray;		// 灰度图像

	bool usePseudoColor = false;

	bool isSelectingRoi = false;
	void initSelectRoi(double xpos, double ypos);
	void updateRoi(double xpos, double ypos);
	cv::Rect roi;
	cv::Mat roiMask;

	int rowMin, rowMax;
	int colMin, colMax;
	float xminTexture, xmaxTexture;
	float yminTexture, ymaxTexture;
	glm::vec2 selectPixel;
	glm::vec2 selectPixelInTexture = glm::vec2(0);

	float scale = 1.0;
	int scaleLevel = 0;			// 图片的放大倍率   实际放大的倍数是pow(1.25, scale) 对应图片的面积大小就是pow(0.8, scale);
	glm::vec2 offset = glm::vec2(0, 0);

	float ratioWidth = 1.0f, ratioHeight = 1.0f;
	FillStrategy mFillStrategy = FillStrategy::FILL_WIDTH;

	int rows = 0, cols = 0;		// 图像分辨率 不会随着滚轮放缩图像而改变
	int imgType = CV_8UC1;
	cv::Mat pic;

private:
	void updateRoi();
	void updateFillStrategy();

	cv::Point roiPt1, roiPt2;

	glm::vec2 mousePos;

	// RenderState
	GLuint VAO, VBO;

	// 渲染窗口的尺寸
	int screenWidth, screenHeight;

	int width, height;		// 窗口的尺寸
	glm::ivec2 pos;			// 窗口的左上角坐标
	int textureId = 0;
};