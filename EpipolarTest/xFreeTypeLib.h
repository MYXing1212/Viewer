#pragma once

#include<windows.h>

#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif
#include"GLFW/glfw3.h"
#include<iostream>
#include<ft2build.h>
#include<freetype/freetype.h>
#include<freetype/ftglyph.h>
#include<freetype/ftoutln.h>
#include<freetype/fttrigon.h>
#include<glm/glm.hpp>
//#include<windows.h>

using namespace std;

#define MAX_NO_TEXTURE  1
#define CUBE_TEXTURE	0

struct xCharTexture
{
	GLuint m_texID;
	wchar_t m_chaID;
	int m_Width;
	int m_Height;

	int m_adv_x;
	int m_adv_y;
	int m_delta_x;
	int m_delta_y;
public:
	xCharTexture()
	{
		m_texID = 0;
		m_chaID = 0;
		m_Width = 0;
		m_Height = 0;
	}
};

class xFreeTypeLib
{
	FT_Library m_FT2Lib;
	FT_Face m_FT_Face;

public:
	int m_w;
	int m_h;
	void load(const char* font_file, int _w, int _h);

	GLuint loadChar(wchar_t ch);
	GLuint loadCharNew(wchar_t ch, xCharTexture &charTex);

	xCharTexture* getTextChar(wchar_t ch);
	
	~xFreeTypeLib();
};

// 参数lpcstr类型也可是char*
LPWSTR AnsiToUnicode(LPCSTR lpcstr);	