#if defined USE_PCH
#include"stdafx.h"
#endif

#include<iostream>

#include "VectorRenderer.h"
#include"resource_manager.h"


VectorRenderer::VectorRenderer()
{
	
}

VectorRenderer::~VectorRenderer()
{
	if (VAO != -1)
	{
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
	}	
}


void VectorRenderer::initLineData()
{
	int num = 2000;
	std::vector<cv::Point3f> dat(num);
	for (int i = 0; i < 2000; i++)
	{
		dat[i].x = i / (float)num;
		dat[i].y = 0;
		dat[i].z = 0;
	}

	cv::Mat b = cv::Mat(dat);
	cv::Mat c = b.clone();
	lineData = c.reshape(1, (int)dat.size());

}

void VectorRenderer::init(int screenWidth, int screenHeight)
{
	this->screenWidth = screenWidth;
	this->screenHeight = screenHeight;
	
	shader = ResourceManager::LoadShader("shaders\\vecr_geo.vert", "shaders\\vecr_geo.frag", nullptr, "points");
	shader.use();
	shader.setVec3("cen", glm::vec3(0.0f));
	shader.setVec2("screenSize", screenWidth, screenHeight);

	// Configure VAO/VBO for texture quads
	glGenVertexArrays(1, &this->VAO);
	glGenBuffers(1, &this->VBO);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(sphereData), (float*)sphereData, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	initLineData();
}

// 给出鼠标位置，返回归一化后的坐标点
glm::vec2 VectorRenderer::getNormalizedCoord(const float &xpos, const float &ypos)
{
	return glm::vec2(
		xpos / (float)screenWidth * 2.0f - 1.0f,
		-1.0f * (ypos / (float)screenHeight * 2.0f - 1.0f)
		);
}


void VectorRenderer::render(const std::vector<float> grays, glm::vec3 color,
	const float &lowThresh/* = 0*/, const float &highThresh/* = 255.0f*/)
{
	if (grays.empty())
		return;
	int N = (int)grays.size();
	if (N == 0)
		return;
	float *ld = new float[N * 2 * 3];
	for (int i = 0; i < N; i++)
	{
		ld[3 * i * 2 + 0] = -1.0 + (2.0 / N)*i;
		ld[3 * i * 2 + 1] = ((grays[i] - lowThresh) / highThresh)*2.0 - 1.0f;
		ld[3 * i * 2 + 2] = 0.0f;

		ld[3 * (2 * i + 1) + 0] = -1.0 + (2.0 / N)*(i + 1);
		ld[3 * (2 * i + 1) + 1] = ((grays[i] - lowThresh) / highThresh)*2.0 - 1.0f;
		ld[3 * (2 * i + 1) + 2] = 0.0f;
	}

	shader.use();
	shader.setBool("render2D", true);
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * N * 2 * 3, ld, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_LINE_STRIP, 0, N * 2);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	shader.setBool("render2D", false);

	delete[] ld;
}

void VectorRenderer::render(const glm::vec2 &startPt, const glm::vec2 &endPt, glm::vec3 color,
	const float &lowThresh/* = 0*/, const float &highThresh/* = 255.0f*/)
{
	shader.use();
	shader.setVec2("startPt", startPt);
	shader.setVec2("endPt", endPt);
	shader.setVec2("grayRange", lowThresh, highThresh);
	shader.setVec3("color", color);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * lineData.total(), (float*)lineData.data, GL_STATIC_DRAW);
	glBindVertexArray(this->VAO);
	glDrawArrays(GL_LINE_STRIP, 0, lineData.rows);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VectorRenderer::setData(int GL_TEXTURE_id, int texture)
{
	glActiveTexture(GL_TEXTURE0+GL_TEXTURE_id);
	glBindTexture(GL_TEXTURE_2D, texture);
	shader.use();
	shader.setInt("graySeqTexture", GL_TEXTURE_id);
}


