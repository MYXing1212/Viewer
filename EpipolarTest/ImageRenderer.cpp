#if defined USE_PCH
#include"stdafx.h"
#endif

#include<iostream>

#include "ImageRenderer.h"
#include"resource_manager.h"
#include"Texture2D.h"

using namespace std;

void ImageRenderer::setFileInfo(std::string path)
{
	filepath = path;
	int flag1 = path.find_last_of('\\');
	int flag2 = path.find_last_of('/');
	if (flag1 == -1 && flag2 == -1)
	{
		locFolder = "";
		filename = path;
	}
	else
	{
		int flag = std::max(flag1, flag2);
		locFolder = path.substr(0, flag);
		filename = path.substr(flag + 1, path.length() - flag);
	}
	cout << "filepath = " << filepath << endl;
	cout << "filename = " << filename << endl;
	cout << "locFolder = " << locFolder << endl;
}

ImageRenderer::ImageRenderer()
{
	VAO = 0;
	VBO = 0;
	EBO = 0;
}

void ImageRenderer::init(glm::ivec2 pos, glm::ivec2 size, GLint textureId, int screenW, int screenH)
{
	screenWidth = ((screenW == 0) ? size.x : screenW);
	screenHeight = ((screenH == 0) ? size.y : screenH);
	glViewport(pos.x, pos.y, width, height);
	this->pos = pos;
	width = size.x;
	height = size.y;
	this->shader = ResourceManager::LoadShader("shaders\\image.vert",
		"shaders\\image.frag", nullptr, "image");
	this->shader.use();
	shader.setVec2("offset", glm::vec2(0, 0));
	shader.setFloat("scale", 1.0f);
	shader.setInt("texture1", textureId);
	this->textureId = textureId;

	float vertices[] = {
		// positions			// texture coords
		1.0f, 1.0f, 0.0f,		1.0f, 1.0f,			// 右上角
		1.0f, -1.0f, 0.0f,		1.0f, 0.0f,			// 右下角
		-1.0f, -1.0f, 0.0f,		0.0f, 0.0f,			// 左下角
		-1.0f, 1.0f, 0.0f,		0.0f, 1.0f			// 左上角
	};

	unsigned int indices[] = {
		0, 1, 3,		// first triangle
		1, 2, 3		// second triangle
	};

	glGenVertexArrays(1, &this->VAO);
	glGenBuffers(1, &this->VBO);
	glGenBuffers(1, &this->EBO);

	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindVertexArray(0);
}

void ImageRenderer::loadData(const cv::Mat &data)
{
	pic = loadTexture(texture, data);
}

void ImageRenderer::RenderPattern(bool updateViewport/* = true*/)
{
	// Activate corresponding render state
	if(updateViewport)
		glViewport(pos.x, pos.y, width, height);
	GLboolean blendEnabled = glIsEnabled(GL_BLEND);
	glDisable(GL_BLEND);
	shader.use();
	glActiveTexture(GL_TEXTURE0 + textureId);
	glBindTexture(GL_TEXTURE_2D, texture);
	shader.setInt("texture1", textureId);
	shader.setBool("render8bitImage", render8bitImage);
	shader.setBool("usePseudoColor", usePseudoColor);

	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	if (blendEnabled)
		glEnable(GL_BLEND);
	glBindTexture(GL_TEXTURE_2D, 0);
	shader.disable();
}

ImageRenderer::~ImageRenderer()
{
	if (VAO)
	{
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);
	}	
}

void ImageRenderer::copyROI(ImageRenderer *render)
{
	if (this->rows != render->rows || this->cols != render->cols)
		return;
	this->rowMax = render->rowMax;
	this->rowMin = render->rowMin;
	this->colMax = render->colMax;
	this->colMin = render->colMin;
	this->xminTexture = render->xminTexture;
	this->xmaxTexture = render->xmaxTexture;
	this->yminTexture = render->yminTexture;
	this->ymaxTexture = render->ymaxTexture;
	this->selectPixel = render->selectPixel;
	this->selectPixelInTexture = render->selectPixelInTexture;

	this->scale = render->scale;
	this->scaleLevel = render->scaleLevel;
	this->offset = render->offset;
	
	shader.use();
	shader.setFloat("scale", scale);
	shader.setVec2("offset", offset);
}

void ImageRenderer::setMousePos(const float &posx, const float &posy)
{
	//printf("xpos = %f ypos = %f\n", posx, posy);
	float tmpY = screenHeight - 1.0f - posy;
	// 如果在窗体内，那么就更新鼠标的位置
	if (posx >= this->pos.x && posx < this->pos.x + width
		&& tmpY >= this->pos.y && tmpY < this->pos.y + height)
	{
		mousePos = glm::vec2(posx, tmpY);
	}
}

void ImageRenderer::zoomIn()
{
	if (scaleLevel < 32)
	{
		scaleLevel++;
		shader.use();
		scale = pow(0.8, scaleLevel);
		shader.setFloat("scale", scale);
		offset.x += (mousePos.x - pos.x) / (float)width * scale * 0.25;
		offset.y += (mousePos.y - pos.y) / (float)height * scale * 0.25;
		updateOffset(0, 0);
		//updateRoi();
	}
}

void ImageRenderer::zoomOut()
{
	//printf("zoomOut!\n");
	if (scaleLevel >=1)
	{
		scaleLevel--;
		shader.use();
		scale = pow(0.8, scaleLevel);
		shader.setFloat("scale", scale);
		offset.x -= (mousePos.x - pos.x) / (float)width * scale * 0.2;
		offset.y -= (mousePos.y - pos.y) / (float)height * scale * 0.2;
		updateOffset(0, 0);
	}
}

bool ImageRenderer::isInWindow(const glm::vec2 &pos, bool updateActiveStatus/* = false*/)
{
	float tmpY = screenHeight - 1.0f - pos.y;
	//printf("tmpY = %f \n pos.y = %f \n", tmpY, pos.y);  //tmpY上大，下小
	bool flag = false;
	if (pos.x >= this->pos.x && pos.x < this->pos.x + width
		&& tmpY >= this->pos.y && tmpY < this->pos.y + height)
	{
		flag = true;
	}
	else
		flag = false;
	if (updateActiveStatus)
	{
		active = flag;
	}
	return flag;
}


void ImageRenderer::updateOffset(const float &offsetX, const float &offsetY)
{
	shader.use();
	glm::vec2 tmp;
	tmp.x = offset.x - offsetX / (float)width * 1.0f * scale;
	tmp.y = offset.y - offsetY / (float)height * 1.0f * scale;
	
	if (tmp.x <= 0.0)
		tmp.x = 0.0;
	if (tmp.y <= 0.0)
		tmp.y = 0.0;

	if (tmp.x + scale > 1.0f)
		tmp.x = 1.0f - scale;
	if (tmp.y + scale > 1.0f)
		tmp.y = 1.0f - scale;

	offset = tmp;
	shader.setVec2("offset", offset);
	updateRoi();
}

glm::vec3 ImageRenderer::queryValue(const float &x, const float &y)
{
	if (imgGray.data == NULL)
		return glm::vec3(-1.0f, -1.0f, -1.0f);

	float tmpy = screenHeight - 1.0f - y;
	// float转int时强制向下取整
	int row = pic.rows - ((tmpy - pos.y) / (float)height*scale + offset.y) * pic.rows;
	int col = ((x - pos.x) / (float)width*scale + offset.x) * pic.cols;
	//printf("row = %d col = %d\n", row, col);
	selectPixel = glm::vec2(col, row);

	if (imgGray.empty())
		return glm::vec3(0);
	if (row < 0 || row >= imgGray.rows || col < 0 || col >= imgGray.cols)
		return glm::vec3(0);

	if (imgGray.type() == CV_8UC3)
	{
		cv::Vec3b val = imgGray.at<cv::Vec3b>(row, col);
		return glm::vec3(val[2], val[1], val[0]);
	}
	else if (imgGray.type() == CV_8UC1)
	{
		uchar val = imgGray.at<uchar>(row, col);
		return glm::vec3(val, val, val);
	}
	else if(imgGray.type() == CV_32FC1)
	{
		float val = imgGray.at<float>(row, col);
		return glm::vec3(val, val, val);
	}
	else if (imgGray.type() == CV_32FC3)
	{
		float r = imgGray.at<cv::Vec3f>(row, col)[0];
		float g = imgGray.at<cv::Vec3f>(row, col)[1];
		float b = imgGray.at<cv::Vec3f>(row, col)[2];
		return glm::vec3(r, g, b);
	}
	//return pic.at<cv::Vec3b>(pic.rows - 1 - row, col);
}

glm::vec3 ImageRenderer::queryPixelValue(glm::vec2 pixel)
{
	selectPixel = pixel;
	if (imgGray.type() == CV_8UC3)
	{
		cv::Vec3b val = imgGray.at<cv::Vec3b>((int)selectPixel.y, (int)selectPixel.x);
		return glm::vec3(val[2], val[1], val[0]);
	}
	else if (imgGray.type() == CV_8UC1)
	{
		uchar val = imgGray.at<uchar>((int)selectPixel.y, (int)selectPixel.x);
		return glm::vec3(val, val, val);
	}
	else if (imgGray.type() == CV_32FC1)
	{
		float val = imgGray.at<float>((int)selectPixel.y, (int)selectPixel.x);
		return glm::vec3(val, val, val);
	}
}

void ImageRenderer::updateRoi()
{
	//glm::vec2 tl = queryImageCoord(pos.x, screenHeight - 1 - (pos.y + height));
	//glm::vec2 br = queryImageCoord(pos.x + width, screenHeight - pos.y);
	glm::vec2 tl = queryImageCoord(pos.x, screenHeight - (pos.y + height));
	glm::vec2 br = queryImageCoord(pos.x + width-1.0f, screenHeight - 1.0f - pos.y);
	/*printf("tl = %d %d\n", pos.x, screenHeight - (pos.y + height));
	printf("br = %d %d\n", pos.x + width, screenHeight - pos.y);
	printf("tl = %f %f\n", tl.x, tl.y);
	printf("br = %f %f\n", br.x, br.y);*/

	if (tl.x == -1.0f && tl.y == -1.0f) return;
	if (br.x == -1.0f && br.y == -1.0f) return;

	rowMin = std::min(std::max(0, (int)tl.y), imgGray.rows - 1);
	rowMax = std::min(std::max(0, (int)br.y), imgGray.rows - 1);

	colMin = std::min(std::max(0, (int)tl.x), imgGray.cols - 1);
	colMax = std::min(std::max(0, (int)br.x), imgGray.cols - 1);

	float x1 = tl.x / (float)pic.cols;
	float y1 = (pic.rows - tl.y) / (float)pic.rows;
	float x2 = br.x / (float)pic.cols;
	float y2 = (pic.rows - br.y) / (float)pic.rows;
	xminTexture = std::min(std::max(0.0f, std::min(x1, x2)), 1.0f);
	xmaxTexture = std::min(std::max(0.0f, std::max(x1, x2)), 1.0f);

	yminTexture = std::min(std::max(0.0f, std::min(y1, y2)), 1.0f);
	ymaxTexture = std::min(std::max(0.0f, std::max(y1, y2)), 1.0f);
	/*printf("xminTexture = %f\n", xminTexture);
	printf("xmaxTexture = %f\n", xmaxTexture);
	printf("yminTexture = %f\n", yminTexture);
	printf("ymaxTexture = %f\n", ymaxTexture);*/
}

void ImageRenderer::updateRoi(double xpos, double ypos)
{
	if (imgGray.data == NULL)
		return;
	float tmpy = screenHeight - 1.0f - ypos;
	// float转int时强制向下取整
	int row = pic.rows - ((tmpy - pos.y) / (float)height*scale + offset.y) * pic.rows;
	int col = ((xpos - pos.x) / (float)width*scale + offset.x) * pic.cols;
	//printf("row = %d col = %d\n", row, col);
	roiPt2 = cv::Point(col, row);

	int w = abs(roiPt2.x - roiPt1.x) + 1;
	int h = abs(roiPt2.y - roiPt1.y) + 1;
	int tlx = std::min(roiPt1.x, roiPt2.x);
	int tly = std::min(roiPt1.y, roiPt2.y);

	roiMask = cv::Mat::zeros(imgGray.size(), CV_8UC1);
	roiMask(cv::Rect(tlx, tly, w, h)).setTo(255);
}

void ImageRenderer::updateImg(const cv::Mat &img, const bool &flipY/* = true*/)
{
	if(img.empty())
		return ;

#ifdef RENDER_USE_OPENGL
	showFlag = true;
#endif
	rowMin = 0;	rowMax = img.rows-1;
	colMin = 0;	colMax = img.cols - 1;

	xminTexture = 0; xmaxTexture = 1.0f;
	yminTexture = 0; ymaxTexture = 1.0f;

	//if (img.channels() == 3)
	//	cvtColor(img, imgGray, CV_BGR2GRAY);
	//else
	imgGray = img.clone();

	rows = imgGray.rows;
	cols = imgGray.cols;

	glActiveTexture(GL_TEXTURE0 + textureId);
	glBindTexture(GL_TEXTURE_2D, texture);

	if (!img.empty() && (img.type() == CV_8UC1 || img.type() == CV_8UC3))
	{
		shader.use();
		shader.setBool("normalizeValue", false);
		render8bitImage = true;

		cv::Mat fy;
		// 沿Y轴进行翻转
		cv::flip(img, fy, 0);
		if (fy.channels() == 3)
			cv::cvtColor(fy, pic, cv::COLOR_BGR2RGB);
		else
			cv::cvtColor(fy, pic, cv::COLOR_GRAY2BGR);
		//imwrite("E:\\pic.bmp", pic);
		printf("RGB8\n");
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pic.cols, pic.rows, GL_RGB, GL_UNSIGNED_BYTE, (uchar*)pic.data);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pic.cols, pic.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, (uchar*)pic.data);
		//cv::cvtColor(fy, pic, cv::COLOR_GRAY2BGR);
		//render8bitImage = false;
		////pic.convertTo(pic, CV_32FC3);
		//double minValue, maxValue;
		//minMaxLoc(img, &minValue, &maxValue);
		//shader.setVec2("normalizeRange", minValue, maxValue);
		//printf("minValue = %f maxValue = %f\n", minValue, maxValue);
		//split(pic, rgb);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, pic.cols, pic.rows, 0, GL_RED, GL_FLOAT, (float*)rgb[0].data);
		//gluBuild2DMipmaps(GL_TEXTURE_2D, 3, pic.cols, pic.rows, GL_RGB, GL_UNSIGNED_BYTE, (uchar*)pic.data);
	}
	else if (!img.empty() && img.type() == CV_32FC1)
	{
		shader.use();
		shader.setBool("normalizeValue", true);
		double minValue, maxValue;
		minMaxLoc(img, &minValue, &maxValue);
		shader.setVec2("normalizeRange", minValue, maxValue);
		printf("minValue = %f maxValue = %f\n", minValue, maxValue);
		if (flipY)
		{
			cv::flip(img, pic, 0);
		}
		else
			pic = img.clone();
		render8bitImage = false;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, pic.cols, pic.rows, 0, GL_RED, GL_FLOAT, (float*)pic.data);
	}
	else if (!img.empty() && img.type() == CV_32FC3)
	{
		printf("update iMG 32FC3\n");
		shader.use();
		shader.setBool("normalizeValue", true);
		double minValue, maxValue;
		minMaxLoc(img, &minValue, &maxValue);
		shader.setVec2("normalizeRange", minValue, maxValue);
		printf("minValue = %f maxValue = %f\n", minValue, maxValue);
		if (flipY)
		{
			cv::flip(img, pic, 0);
		}
		else
			pic = img.clone();
		render8bitImage = false;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, pic.cols, pic.rows, 0, GL_RGB, GL_FLOAT, (float*)pic.data);
	}
}

glm::vec2 ImageRenderer::convert2RenderCS(glm::vec2 uv)
{
	float gx = ((float)(uv.x + 0.5) / pic.cols - offset.x) / scale *2.0 - 1.0;
	float gy = ((float)(pic.rows - (uv.y + 0.5)) / pic.rows - offset.y) / scale *2.0 - 1.0;
	return glm::vec2(gx, gy);
}

std::vector<glm::vec2> ImageRenderer::convert2RenderCS(std::vector<glm::vec2> uv)
{
	std::vector<glm::vec2> result;
	for (auto p : uv)
	{
		result.push_back(convert2RenderCS(p));
	}
	return result;
}

void ImageRenderer::setRenderUsePseudoColor(bool flag)
{
	usePseudoColor = flag;
	//shader.use();
	//shader.setBool("usePseudoColor", flag);
}


void ImageRenderer::updateRenderParas()
{
	shader.use();
	shader.setVec2("offset", offset);
	shader.setFloat("scale", scale);
}

void ImageRenderer::reset()
{
	//pixelPos = glm::ivec2(0, 0);
	scaleLevel = 0;
	scale = 1.0;

	offset = glm::vec2(0, 0);

	shader.use();
	shader.setVec2("offset", offset);
	shader.setFloat("scale", scale);
}

void ImageRenderer::initSelectRoi(double xpos, double ypos)
{
	isSelectingRoi = true;
	if (imgGray.data == NULL)
		return;
	float tmpy = screenHeight - 1.0f - ypos;
	// float转int时强制向下取整
	int row = pic.rows - ((tmpy - pos.y) / (float)height*scale + offset.y) * pic.rows;
	int col = ((xpos - pos.x) / (float)width*scale + offset.x) * pic.cols;
	//printf("row = %d col = %d\n", row, col);
	roiPt1 = cv::Point(col, row);
}

// 返回图像的像素坐标
glm::vec2 ImageRenderer::queryImageCoord(const float &xpos, const float &ypos)
{
	if (!isInWindow(glm::vec2(xpos, ypos)))
		return glm::vec2(-1.0f, -1.0f);
	float tmpy = screenHeight - 1.0f - ypos;
	float row = pic.rows - ((tmpy - pos.y) / (float)height*scale + offset.y) * pic.rows;
	float col = ((xpos - pos.x) / (float)width*scale + offset.x) * pic.cols;
	//printf("ROW = %d COL = %d\n", row, col);
	selectPixel = glm::vec2(col, row);
	selectPixelInTexture = glm::vec2(col / (float)pic.cols, (pic.rows - row) / (float)pic.rows);
	return glm::vec2(col, row);
}

void ImageRenderer::setNormalizeRange( const float &low, const float &high )
{
	shader.use();
	shader.setVec2("normalizeRange", low, high);
}

#ifdef RENDER_USE_OPENGL
void ImageRenderer::updateSelectTargets(const glm::vec2 &startPt, const glm::vec2 &endPt)
{
	float minx = std::min(startPt.x, endPt.x);
	float maxx = std::max(startPt.x, endPt.x);
	float miny = std::min(startPt.y, endPt.y);
	float maxy = std::max(startPt.y, endPt.y);

	for (int i = 0; i < targets.size(); i++)
	{
		if (flagsTarget[i] != 1)
			continue;
		glm::vec2 normalizedCoord = convert2RenderCS(targets[i]);
		float x = normalizedCoord.x;
		float y = normalizedCoord.y;

		if (x < maxx && x > minx && y < maxy && y > miny)
		{
			flagsTarget[i] = 0;
		}
	}
}

void ImageRenderer::deleteSelectedTargets()
{
	//printf("deleteSelected targets!\n");
	for (int i = 0; i < targets.size(); i++)
	{
		if (flagsTarget[i] == 0)
			flagsTarget[i] = -1;
	}
}

void ImageRenderer::resetSelectTargets()
{
	for (int i = 0; i < targets.size(); i++)
		flagsTarget[i] = 1;
}
#endif
