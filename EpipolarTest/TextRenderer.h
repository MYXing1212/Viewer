#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include<map>

#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif
#include<glm/glm.hpp>
#include<vector>

#include"xFreeTypeLib.h"

#include"Texture2D.h"
#include"Shader.h"

// A renderer class for renderering text displayed by a font loaded using the 
// Freetype library. A single font is loaded, processed into a list of Character
// items for later rendering.
class TextRenderer
{
public:
	TextRenderer(){}
	// Shader used for text rendering 
	Shader TextShader;
	// Constructor
	TextRenderer(GLuint width, GLuint height);
	void initWindow(GLuint width, GLuint height);
	void initWindow();
	// Pre-compiles a list of characters from the given font
	void Load(std::string font, GLuint fontSize);

	void SetPrjMatrix(glm::mat4 projection);
	void SetModelMatrix(glm::mat4 model);

	// 计算文字的尺寸和位置
	glm::vec4 calcTextSizePos(std::string text, float x, float y, float scale);

	// 当centered为false时 字体纹理的左下角点在屏幕坐标系下坐标 以屏幕左下角为起点的
	// 当centered为true时  字体纹理的左下角点在屏幕坐标系下坐标 以屏幕左下角为起点的
	// centered默认为false
	void RenderText(std::string text, GLfloat x, GLfloat y, GLfloat scale,
		glm::vec3 color = glm::vec3(1.0f), bool centered = false, bool useDepthTest = true);

	GLuint window_width, window_height;
	glm::mat4 projection, view;
private:
	// RenderState
	GLuint VAO, VBO;
	xFreeTypeLib g_FreeTypeLib;

	xCharTexture* H_render;		// 'H' 的字形
	void init();
};

#endif