#if defined USE_PCH
#include"stdafx.h"
#endif

#include<iostream>

#include<ft2build.h>
#include FT_FREETYPE_H

#include "TextRenderer.h"
#include"resource_manager.h"

wchar_t* string2pwchar_t(const std::string &pKey)
{
	const char* pCStrKey = pKey.c_str();
	int pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
	wchar_t *pWCStrKey = new wchar_t[pSize];
	MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
	return pWCStrKey;
}

TextRenderer::TextRenderer(GLuint width, GLuint height)
{
	init();
	initWindow(width, height);
}

void TextRenderer::initWindow()
{
	init();
}

void TextRenderer::initWindow(GLuint width, GLuint height){
	init();
	this->window_width = width;
	this->window_height = height;
	this->TextShader.use();
	this->projection = glm::ortho(0.0f, static_cast<GLfloat>(width), 0.0f, static_cast<GLfloat>(height), 0.0001f, 2.0f);
	this->view = glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	this->TextShader.setMat4("projection", this->projection);
	this->TextShader.setMat4("view", this->view);
}

void TextRenderer::init()
{
	//glShadeModel(GL_SMOOTH | GL_FLAT);       // Enable Smooth Shading
	//glEnable(GL_COLOR_MATERIAL_FACE);
	//glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	this->TextShader = ResourceManager::LoadShader("shaders\\text.vert", "shaders\\text.frag", nullptr, "text");
	this->TextShader.use();
	this->TextShader.setInt("text", 0);
	glm::mat4 model = glm::mat4();
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.9998f));
	this->TextShader.setMat4("model", model);
	// Configure VAO/VBO for texture quads
	glGenVertexArrays(1, &this->VAO);
	glGenBuffers(1, &this->VBO);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void TextRenderer::SetPrjMatrix(glm::mat4 projection){
	TextShader.use();
	TextShader.setMat4("projection", projection);
	this->projection = projection;
}

void TextRenderer::SetModelMatrix(glm::mat4 model){
	TextShader.use();
	TextShader.setMat4("model", model);
}

void TextRenderer::Load(std::string font, GLuint fontSize)
{
	std::ifstream file(font);
	if (!file)
	{
		printf("字体文件不存在!\n");
		return;
	}
	file.close();
	g_FreeTypeLib.load(font.c_str(), fontSize, fontSize);

	// 第一个参数表示设置像素的对齐值 第二个参数表示实际设置为多少
	// 这里像素可以单字节对齐（实际上就是不使用对齐）
	// 双字节对齐（如果长度为奇数，则再补一个字节）
	// 四字节对齐（如果长度不是4的倍数，则补为4的倍数）
	// 八字节对齐
	// 分别对应alignment的值为1, 2, 4, 8。实际上，默认的值是4，正好与BMP文件的对齐方式相吻合
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	string text_H = "H";
	wchar_t* _strText = string2pwchar_t(text_H);
	H_render = g_FreeTypeLib.getTextChar(_strText[0]);		// 'H' 的字形
	cout << "H_render->m_delta_y = " << H_render->m_delta_y << endl;
	cout << "H_render->m_adv_y = " << H_render->m_adv_y << endl;
}

// 计算文字的尺寸和位置 左下角点的坐标 + width + height 以图像左下角点为坐标原点
glm::vec4 TextRenderer::calcTextSizePos(std::string text, float x, float y, float scale)
{
	wchar_t* _strText = string2pwchar_t(text);
	int sx = x;
	int sy = y;

	float top = DBL_MIN;
	float bottom = DBL_MAX;

	glm::vec4 size(0);

	int maxW = 15000;
	int maxH = 19500;
	size_t nLen = wcslen(_strText);

	for (int i = 0; i < nLen; i++)
	{
		if (_strText[i] == '/n')
		{
			sx = x; sy += maxH + 12;
			continue;
		}
		xCharTexture* pCharTex = g_FreeTypeLib.getTextChar(_strText[i]);

		float w = pCharTex->m_Width * scale;
		float h = pCharTex->m_Height * scale;

		float ch_x = sx + pCharTex->m_delta_x * scale;
		float ch_y = sy + (H_render->m_delta_y + 2 * pCharTex->m_delta_y - pCharTex->m_delta_y) * scale;
		
		if (size[1] < h)
		{
			size[1] = h;
		}

		if (maxH < h) maxH = h;
		if (top < ch_y + h) top = ch_y + h;
		if (bottom > ch_y) bottom = ch_y;

		GLfloat vertices[6][4] = {
			{ ch_x, ch_y + h, 0.0, 0.0 },
			{ ch_x, ch_y, 0.0, 1.0 },
			{ ch_x + w, ch_y, 1.0, 1.0 },

			{ ch_x, ch_y + h, 0.0, 0.0 },
			{ ch_x + w, ch_y, 1.0, 1.0 },
			{ ch_x + w, ch_y + h, 1.0, 0.0 }
		};
	
		sx += pCharTex->m_adv_x * scale;
		if (sx > maxW)
		{
			sx = 0; sy += maxH + 12;
		}
	}
	size[0] = x;
	size[1] = bottom;
	size[2] = sx - x;
	size[3] = top - bottom;
	return size;
}

// 以屏幕左下角为起点的
void TextRenderer::RenderText(std::string text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color, 
	bool centered/* = false*/, bool useDepthTest/* = true*/)
{
	bool enableDepthText = glIsEnabled(GL_DEPTH_TEST);
	if(!useDepthTest)
		glDisable(GL_DEPTH_TEST);
	if (centered)
	{
		glm::vec4 sz = calcTextSizePos(text, x, y, scale);
		x = x - sz[2] / 2.0f;
		y = y - sz[3] / 2.0f;
	}

	// Activate corresponding render state
	this->TextShader.use();
	this->TextShader.setVec3("textColor", color);
	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(this->VAO);

	wchar_t* _strText = string2pwchar_t(text);

	int sx = x;
	int sy = y;
	int maxW = 15000;
	int maxH = 19500;
	size_t nLen = wcslen(_strText);

	for (int i = 0; i < nLen; i++)
	{
		if (_strText[i] == '/n')
		{
			sx = x; sy += maxH + 12;
			continue;
		}
		xCharTexture* pCharTex = g_FreeTypeLib.getTextChar(_strText[i]);

		float w = pCharTex->m_Width * scale;
		float h = pCharTex->m_Height * scale;

		float ch_x = sx + pCharTex->m_delta_x * scale;
		float ch_y = sy + (H_render->m_delta_y + 2 * pCharTex->m_delta_y - pCharTex->m_delta_y) * scale;

		if (maxH < h) maxH = h;

		GLfloat vertices[6][4] = {
			{ ch_x, ch_y + h, 0.0, 0.0 },
			{ ch_x, ch_y, 0.0, 1.0 },
			{ ch_x + w, ch_y, 1.0, 1.0 },

			{ ch_x, ch_y + h, 0.0, 0.0 },
			{ ch_x + w, ch_y, 1.0, 1.0 },
			{ ch_x + w, ch_y + h, 1.0, 0.0 }
		};
		// Render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, pCharTex->m_texID);
		// Update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// Render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		sx += pCharTex->m_adv_x * scale;
		if (sx > x + maxW)
		{
			sx = x; sy += maxH + 12;
		}
	}
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	if (enableDepthText)
		glEnable(GL_DEPTH_TEST);
}

 