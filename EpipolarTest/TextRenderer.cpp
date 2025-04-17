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
		printf("�����ļ�������!\n");
		return;
	}
	file.close();
	g_FreeTypeLib.load(font.c_str(), fontSize, fontSize);

	// ��һ��������ʾ�������صĶ���ֵ �ڶ���������ʾʵ������Ϊ����
	// �������ؿ��Ե��ֽڶ��루ʵ���Ͼ��ǲ�ʹ�ö��룩
	// ˫�ֽڶ��루�������Ϊ���������ٲ�һ���ֽڣ�
	// ���ֽڶ��루������Ȳ���4�ı�������Ϊ4�ı�����
	// ���ֽڶ���
	// �ֱ��Ӧalignment��ֵΪ1, 2, 4, 8��ʵ���ϣ�Ĭ�ϵ�ֵ��4��������BMP�ļ��Ķ��뷽ʽ���Ǻ�
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	string text_H = "H";
	wchar_t* _strText = string2pwchar_t(text_H);
	H_render = g_FreeTypeLib.getTextChar(_strText[0]);		// 'H' ������
	cout << "H_render->m_delta_y = " << H_render->m_delta_y << endl;
	cout << "H_render->m_adv_y = " << H_render->m_adv_y << endl;
}

// �������ֵĳߴ��λ�� ���½ǵ������ + width + height ��ͼ�����½ǵ�Ϊ����ԭ��
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

// ����Ļ���½�Ϊ����
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

 