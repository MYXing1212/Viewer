#include<iostream>

#include "BigImageRenderer.h"
#include"resource_manager.h"
#include"Texture2D.h"

using namespace std;

void BigImageRenderer::setFileInfo(std::string path)
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

BigImageRenderer::BigImageRenderer()
{
	VAO = 0;
	VBO = 0;
}

void BigImageRenderer::init(glm::ivec2 pos, glm::ivec2 size, GLint textureId, int screenW, int screenH)
{
	screenWidth = ((screenW == 0) ? size.x : screenW);
	screenHeight = ((screenH == 0) ? size.y : screenH);
	glViewport(pos.x, pos.y, width, height);
	this->pos = pos;
	width = size.x;
	height = size.y;
	this->shader = ResourceManager::LoadShader("shaders\\bigimage.vert",
		"shaders\\bigimage.frag", nullptr, "bigimage");
	this->shader.use();
	//shader.setVec2("offset", glm::vec2(0, 0));
	//shader.setFloat("scale", 1.0f);
	//shader.setInt("texture1", textureId);
	this->textureId = textureId;

	float vertices[] = {
		// 位置       // 纹理坐标
		0.0f, 0.0f, 0.0f, 0.0f,  // 左下
		 1.0f, 0.0f, 1.0f, 0.0f,  // 右下
		0.0f,  1.0f, 0.0f, 1.0f,  // 左上
		 1.0f,  1.0f, 1.0f, 1.0f   // 右上
	};

	glGenVertexArrays(1, &this->VAO);
	glGenBuffers(1, &this->VBO);

	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	ratioWindow = float(size.x) / size.y;

	// 更新投影矩阵
	scope.minX = 0.5f - ratioWindow / 2.0f;
	scope.maxX = 0.5f + ratioWindow / 2.0f;
	scope.minY = 0.0f;
	scope.maxY = 1.0f;
	updateProjectionMatrix();

	// 更新视图矩阵 (平移和缩放)
	view = glm::mat4(1.0f);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindVertexArray(0);
}

void BigImageRenderer::loadData(const cv::Mat &data)
{
	pic = loadTexture(texture, data);
}

void BigImageRenderer::RenderPattern(bool updateViewport/* = true*/)
{
	// Activate corresponding render state
	if(updateViewport)
		glViewport(pos.x, pos.y, width, height);
	//glClearColor(0.3, 0.3, 0.3, 1.0);
	//// 清除颜色缓冲区
	//glClear(GL_COLOR_BUFFER_BIT);
	GLboolean blendEnabled = glIsEnabled(GL_BLEND);
	glDisable(GL_BLEND);
	shader.use();
	shader.setMat4("projection", projection);
	shader.setMat4("view", view); // view 在这里始终等于单位矩阵
	shader.setInt("imgType", imgType);
	shader.setInt("imageTexture", 0);


	//std::lock_guard<std::mutex> lock(tileMutex);
	// tile的size以像素为单位，tile最大尺寸为4096
	for (const auto& tile : tiles)
	{
		if (tile.loaded)
		{
			// 计算模型矩阵 (位置和缩放)
			// 在OpenGL和glm中，矩阵乘法是从右向左进行的，也就是说，最后应用的变换实际上是第一个被乘的矩阵。
			// 这里是把Tile变换到NDC，所以需要先缩放，再平移。

			// 策略，先把图片（所有tiles范围）归一化到[0,1],然后再乘/除图片自身的横纵比，让图片横纵不失真
			// 为实现先把图片（所有tiles范围）归一化到[0,1]，那么针对每一片Tile，需要tile.width/cols,tile.height/rows
			// 注意：最后一行Tile，或者最后一列Tile，可能不足tileSize，所以最终归一化的范围肯能比1小一些
			// imageRatio = (float)cols / (float)rows;

			glm::mat4 model = glm::mat4(1.0f);
			if (mFillStrategy == FillStrategy::FILL_WIDTH) // 胖型，优先保证X方向，Y方向 / imageRatio
			{
				model = glm::translate(model, glm::vec3(
					tile.x / (float)cols,
					tile.y / (float)rows / imageRatio,
					0.0f));
				model = glm::scale(model, glm::vec3(
					tile.width / (float)cols,
					tile.height / (float)rows / imageRatio,
					1.0f));
			}
			else // 瘦型，优先保证Y方向, 即Y坐标不需要处理，X坐标*imageRatio
			{
				model = glm::translate(model, glm::vec3(
					tile.x / (float)cols * imageRatio,
					tile.y / (float)rows,
					0.0f));
				model = glm::scale(model, glm::vec3(
					tile.width / (float)cols  * imageRatio,
					tile.height / (float)rows,
					1.0f));
			}


			shader.setMat4("model", model);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, tile.textureID);

			glBindVertexArray(VAO);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		}
	}


	/*glActiveTexture(GL_TEXTURE0 + textureId);
	glBindTexture(GL_TEXTURE_2D, texture);
	shader.setInt("texture1", textureId);
	shader.setBool("render8bitImage", render8bitImage);
	shader.setBool("usePseudoColor", usePseudoColor);

	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	if (blendEnabled)
		glEnable(GL_BLEND);
	glBindTexture(GL_TEXTURE_2D, 0);*/
	shader.disable();
}

BigImageRenderer::~BigImageRenderer()
{
	if (VAO)
	{
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
	}	

	for (auto& tile : tiles)
	{
		if (tile.loaded)
		{
			glDeleteTextures(1, &tile.textureID);
		}
	}
}

// 创建图像分块
void BigImageRenderer::createTiles() 
{
	int tilesX = (cols + tileSize - 1) / tileSize;
	int tilesY = (rows + tileSize - 1) / tileSize;

	tiles.resize(tilesX * tilesY);

	for (int y = 0; y < tilesY; ++y) {
		for (int x = 0; x < tilesX; ++x) {
			int idx = y * tilesX + x;
			tiles[idx].x = x * tileSize;
			tiles[idx].y = y * tileSize;
			tiles[idx].width = std::min(tileSize, cols - x * tileSize);
			tiles[idx].height = std::min(tileSize, rows - y * tileSize);
			tiles[idx].loaded = false;
		}
	}
}

// 加载可见区域的纹理块
void BigImageRenderer::loadVisibleTiles()
{
	// 计算当前可见区域
	glm::vec2 visibleMin = viewPosition - glm::vec2(1.0f / zoomLevel);
	glm::vec2 visibleMax = viewPosition + glm::vec2(1.0f / zoomLevel);

	//std::lock_guard<std::mutex> lock(tileMutex);

	for (auto& tile : tiles) {
		// 检查块是否在可见区域内
		glm::vec2 tileMin(tile.x / (float)cols, tile.y / (float)rows);
		glm::vec2 tileMax((tile.x + tile.width) / (float)cols,
			(tile.y + tile.height) / (float)rows);

		bool isVisible = !(tileMax.x < visibleMin.x || tileMin.x > visibleMax.x ||
			tileMax.y < visibleMin.y || tileMin.y > visibleMax.y);

		if (isVisible && !tile.loaded) {
			loadTile(tile);
		}
		else if (!isVisible && tile.loaded) {
			// 可选: 卸载不可见块以节省内存
			// glDeleteTextures(1, &tile.textureID);
			// tile.loaded = false;
		}
	}
}

// 加载单个纹理块
void BigImageRenderer::loadTile(ImageTile& tile) 
{
	// 这里应该是实际的图像加载代码
	// 示例使用空白纹理代替

	cv::Mat tmpImg = imgGray(cv::Rect(tile.x, tile.y, tile.width, tile.height)).clone();
	if (tile.textureID == UNDEFINED_TEXTURE)
	{
		glGenTextures(1, &tile.textureID);
	}
	glBindTexture(GL_TEXTURE_2D, tile.textureID);

	// 为8位灰度或32位浮点图像设置适当的格式
	if (imgType == CV_8UC1)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8,
			tile.width, tile.height, 0,
			GL_RED, GL_UNSIGNED_BYTE, tmpImg.data);
	}
	else if (imgType == CV_8UC3)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8,
			tile.width, tile.height, 0,
			GL_RGB, GL_UNSIGNED_BYTE, tmpImg.data);
	}
	else if (imgType == CV_32FC3)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,
			tile.width, tile.height, 0,
			GL_RGB, GL_FLOAT, tmpImg.data);
	}
	else if(imgType == CV_32FC1)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
			tile.width, tile.height, 0,
			GL_RED, GL_FLOAT, tmpImg.data);
	}
	
	// 设置纹理参数
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	tile.loaded = true;
}

void BigImageRenderer::setMousePos(const float &posx, const float &posy)
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

void BigImageRenderer::copyROI(BigImageRenderer* render)
{
	if (this->rows != render->rows || this->cols != render->cols)
		return;
	this->scope = render->scope;
	updateProjectionMatrix();
}

void BigImageRenderer::zoomIn()
{
		float x_ndc = 2.0 * (mousePos.x - pos.x) / (float)width - 1.0f;
		float y_ndc = 2.0 * (mousePos.y - pos.y) / (float)height - 1.0f;

		auto pos_scope = glm::inverse(projection) * glm::vec4(x_ndc, -y_ndc, 0.0f, 1.0f);

		scope.minX = pos_scope.x - (pos_scope.x - scope.minX) * 0.8;
		scope.minY = pos_scope.y - (pos_scope.y - scope.minY) * 0.8;
		
		scope.maxX = pos_scope.x + (scope.maxX - pos_scope.x) * 0.8;
		scope.maxY = pos_scope.y + (scope.maxY - pos_scope.y) * 0.8;
		updateProjectionMatrix();
}

void BigImageRenderer::zoomOut()
{
	float x_ndc = 2.0 * (mousePos.x - pos.x) / (float)width - 1.0f;
	float y_ndc = 2.0 * (mousePos.y - pos.y) / (float)height - 1.0f;
	auto pos_scope = glm::inverse(projection) * glm::vec4(x_ndc, -y_ndc, 0.0f, 1.0f);

	scope.minX = pos_scope.x - (pos_scope.x - scope.minX) * 1.25;
	scope.minY = pos_scope.y - (pos_scope.y - scope.minY) * 1.25;

	scope.maxX = pos_scope.x + (scope.maxX - pos_scope.x) * 1.25;
	scope.maxY = pos_scope.y + (scope.maxY - pos_scope.y) * 1.25;
	updateProjectionMatrix();
}

bool BigImageRenderer::isInWindow(const glm::vec2 &pos, bool updateActiveStatus/* = false*/)
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


void BigImageRenderer::updateOffset(const float &offsetX, const float &offsetY)
{
	scope.minX = scope.minX - offsetX / width *  scope.xspan;
	scope.maxX = scope.maxX - offsetX / width * scope.xspan;
	scope.minY = scope.minY + offsetY / height * scope.yspan;
	scope.maxY = scope.maxY + offsetY / height * scope.yspan;
	updateProjectionMatrix();
}

glm::vec3 BigImageRenderer::queryValue(const float &x, const float &y)
{
	if (imgGray.data == NULL)
		return glm::vec3(-1.0f, -1.0f, -1.0f);

	float tmpy = screenHeight - 1.0f - y;

	float x_ndc = (x - pos.x) / (float)width * 2.0f - 1.0f;
	float y_ndc = (tmpy - pos.y) / (float)height * 2.0f - 1.0f;

	auto pos_normal = glm::inverse(projection * view) * glm::vec4(x_ndc, -y_ndc, 0.0f, 1.0f);

	int row = 0, col = 0;
	if (mFillStrategy == FillStrategy::FILL_WIDTH) // 胖型，优先保证X方向，Y方向 / imageRatio
	{
		row = pos_normal.y * imageRatio * rows ; //+ rows;
		col = pos_normal.x * cols;
	}
	else if(mFillStrategy == FillStrategy::FILL_HEIGHT) // 瘦型，优先保证Y方向, X方向*imageRatio
	{
		row = pos_normal.y * rows; //+ rows;
		col = pos_normal.x / imageRatio * cols;
	}

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
}

glm::vec3 BigImageRenderer::queryPixelValue(glm::vec2 pixel)
{
	selectPixel = pixel;
	if ((int)selectPixel.x >= imgGray.cols || (int)selectPixel.x < 0 || (int)selectPixel.y >= imgGray.rows || (int)selectPixel.y < 0)
		return glm::vec3(-1.0f, -1.0f, -1.0f);
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

void BigImageRenderer::updateRoi()
{
	glm::vec2 tl = queryImageCoord(pos.x, screenHeight - (pos.y + height));
	glm::vec2 br = queryImageCoord(pos.x + width-1.0f, screenHeight - 1.0f - pos.y);

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
}

void BigImageRenderer::updateFillStrategy()
{
	ratioWidth = (float)width / cols;
	ratioHeight = (float)height / rows;
	if (ratioWidth > ratioHeight)
	{
		mFillStrategy = FillStrategy::FILL_HEIGHT;
	}
	else
	{
		mFillStrategy = FillStrategy::FILL_WIDTH;
	}
}

void BigImageRenderer::updateRoi(double xpos, double ypos)
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

void BigImageRenderer::updateImg(const cv::Mat &img, const bool &flipY/* = true*/)
{
	if(img.empty())
		return ;

	rowMin = 0;	rowMax = img.rows-1;
	colMin = 0;	colMax = img.cols - 1;

	xminTexture = 0; xmaxTexture = 1.0f;
	yminTexture = 0; ymaxTexture = 1.0f;

	imgGray = img.clone();

	rows = imgGray.rows;
	cols = imgGray.cols;

	imageRatio = (float)cols / (float)rows;

	imgType = img.type();
	updateFillStrategy();

	createTiles();
	loadVisibleTiles();

	if (mFillStrategy == FillStrategy::FILL_WIDTH)
	{
		scope.minX = 0.0f;
		scope.maxX = 1.0f;
		scope.minY = 0;
		scope.maxY = 1.0 / ratioWindow;
	}
	else if (mFillStrategy == FillStrategy::FILL_HEIGHT)
	{
		// 要保证xspan = ratioWindow * yspan
		// 这里ysan = 1.0
		// 所以xspan = rationWindow
		scope.minX = 0;
		scope.maxX = ratioWindow;
		scope.minY = 0.0f;
		scope.maxY = 1.0f;
	}
	updateProjectionMatrix();
}

glm::vec2 BigImageRenderer::convert2RenderCS(glm::vec2 uv)
{
	float gx = ((float)(uv.x + 0.5) / pic.cols - offset.x) / scale *2.0 - 1.0;
	float gy = ((float)(pic.rows - (uv.y + 0.5)) / pic.rows - offset.y) / scale *2.0 - 1.0;
	return glm::vec2(gx, gy);
}

std::vector<glm::vec2> BigImageRenderer::convert2RenderCS(std::vector<glm::vec2> uv)
{
	std::vector<glm::vec2> result;
	for (auto p : uv)
	{
		result.push_back(convert2RenderCS(p));
	}
	return result;
}

void BigImageRenderer::setRenderUsePseudoColor(bool flag)
{
	usePseudoColor = flag;
}

void BigImageRenderer::updateProjectionMatrix()
{
	scope.xspan = scope.maxX - scope.minX;
	scope.yspan = scope.maxY - scope.minY;
	projection = glm::ortho(scope.minX, scope.maxX, scope.minY, scope.maxY, -1.0f, 1.0f);
}

void BigImageRenderer::updateRenderParas()
{
	shader.use();
	shader.setVec2("offset", offset);
	shader.setFloat("scale", scale);
}

void BigImageRenderer::reset()
{
	scaleLevel = 0;
	scale = 1.0;
	offset = glm::vec2(0, 0);
	shader.use();
}

void BigImageRenderer::initSelectRoi(double xpos, double ypos)
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
glm::vec2 BigImageRenderer::queryImageCoord(const float &xpos, const float &ypos)
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

void BigImageRenderer::setNormalizeRange( const float &low, const float &high )
{
	shader.use();
	shader.setVec2("normalizeRange", low, high);
}

