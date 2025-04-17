#include"xFreeTypeLib.h"

using namespace std;

GLuint texture_id[MAX_NO_TEXTURE];
static xCharTexture g_TexID[65536];

void xFreeTypeLib::load(const char* font_file, int _w, int _h)
{
	FT_Library library;
	if (FT_Init_FreeType(&library))
		exit(0);
	// 加载一个字体，取默认的Face，一般为Regualer
	if (FT_New_Face(library, font_file, 0, &m_FT_Face))
		exit(0);
	// 选择字符表
	FT_Select_Charmap(m_FT_Face, FT_ENCODING_UNICODE);
	m_w = _w; 
	m_h = _h;
	m_FT_Face->num_fixed_sizes;
	// 大小要乘64，这是规定，照做就可以了.
	// FT_Set_Char_Size(m_FT_Face, 0, m_w<<6, 96, 96);
	// 用来存放制定字符宽度和高度的特定数据
	FT_Set_Pixel_Sizes(m_FT_Face, m_w, m_h);

	for (GLubyte c = 0; c < 128; c++)
	{
		// Load character glyph
		if (FT_Load_Char(m_FT_Face, c, FT_LOAD_RENDER))
		{
			std::cout << "ERROR::FREETYPE: Failed to load Glyph" << std::endl;
			continue;
		}
		// Generate texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RED,
			m_FT_Face->glyph->bitmap.width,
			m_FT_Face->glyph->bitmap.rows,
			0,
			GL_RED,
			GL_UNSIGNED_BYTE,
			m_FT_Face->glyph->bitmap.buffer
			);
		// Set texture options;
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
}

xFreeTypeLib::~xFreeTypeLib()
{
	FT_Done_Face(m_FT_Face);
	FT_Done_FreeType(m_FT2Lib);
}



GLuint xFreeTypeLib::loadChar(wchar_t ch)
{
	//if (g_TexID[ch].m_texID)
	//	return g_TexID[ch].m_texID;
	// 装载字形图像到字形槽（将会抹掉先前的字形图像)
	if (FT_Load_Char(m_FT_Face, ch, FT_LOAD_RENDER))
		return 0;

	xCharTexture& charTex = g_TexID[ch];

	//得到字模  
	FT_Glyph glyph;
	//把字形图像从字形槽复制到新的FT_Glyph对象glyph中。这个函数返回一个错误码并且设置glyph。   
	if (FT_Get_Glyph(m_FT_Face->glyph, &glyph))
		return 0;

	//转化成位图  
	FT_Render_Glyph(m_FT_Face->glyph, FT_RENDER_MODE_LCD);//FT_RENDER_MODE_NORMAL  );   
	FT_Glyph_To_Bitmap(&glyph, ft_render_mode_normal, 0, 1);
	FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;

	//取道位图数据  
	FT_Bitmap& bitmap = bitmap_glyph->bitmap;

	//把位图数据拷贝自己定义的数据区里.这样旧可以画到需要的东西上面了。  
	int width = bitmap.width;
	int height = bitmap.rows;

	m_FT_Face->size->metrics.y_ppem;		//伸缩距离到设备空间  
	m_FT_Face->glyph->metrics.horiAdvance;  //水平文本排列  

	charTex.m_Width = width;
	charTex.m_Height = height;
	charTex.m_adv_x = m_FT_Face->glyph->advance.x / 64.0f;  //步进宽度  
	charTex.m_adv_y = m_FT_Face->size->metrics.y_ppem;        //m_FT_Face->glyph->metrics.horiBearingY / 64.0f;  
	charTex.m_delta_x = (float)bitmap_glyph->left;           //left:字形原点(0,0)到字形位图最左边象素的水平距离.它以整数象素的形式表示。   
	charTex.m_delta_y = (float)bitmap_glyph->top - height;   //Top: 类似于字形槽的bitmap_top字段。  

	glGenTextures(1, &charTex.m_texID);
	glBindTexture(GL_TEXTURE_2D, charTex.m_texID);

	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		width,
		height,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		m_FT_Face->glyph->bitmap.buffer);  //指定一个二维的纹理图片  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);                            //glTexParameteri():纹理过滤  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);                                //纹理进行混合  

	//delete[] pBuf;
	return charTex.m_chaID;
}

GLuint xFreeTypeLib::loadCharNew(wchar_t ch, xCharTexture &charTex)
{
	//if (g_TexID[ch].m_texID)
	//	return g_TexID[ch].m_texID;
	// 装载字形图像到字形槽（将会抹掉先前的字形图像)
	if (FT_Load_Char(m_FT_Face, ch, FT_LOAD_RENDER))
		return 0;

	//xCharTexture& charTex = g_TexID[ch];

	//得到字模  
	FT_Glyph glyph;
	//把字形图像从字形槽复制到新的FT_Glyph对象glyph中。这个函数返回一个错误码并且设置glyph。   
	if (FT_Get_Glyph(m_FT_Face->glyph, &glyph))
		return 0;

	//转化成位图  
	FT_Render_Glyph(m_FT_Face->glyph, FT_RENDER_MODE_LCD);//FT_RENDER_MODE_NORMAL  );   
	FT_Glyph_To_Bitmap(&glyph, ft_render_mode_normal, 0, 1);
	FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;

	//取道位图数据  
	FT_Bitmap& bitmap = bitmap_glyph->bitmap;

	//把位图数据拷贝自己定义的数据区里.这样旧可以画到需要的东西上面了。  
	int width = bitmap.width;
	int height = bitmap.rows;

	m_FT_Face->size->metrics.y_ppem;		//伸缩距离到设备空间  
	m_FT_Face->glyph->metrics.horiAdvance;  //水平文本排列  

	charTex.m_Width = width;
	charTex.m_Height = height;
	charTex.m_adv_x = m_FT_Face->glyph->advance.x / 64.0f;  //步进宽度  
	charTex.m_adv_y = m_FT_Face->size->metrics.y_ppem;        //m_FT_Face->glyph->metrics.horiBearingY / 64.0f;  
	charTex.m_delta_x = (float)bitmap_glyph->left;           //left:字形原点(0,0)到字形位图最左边象素的水平距离.它以整数象素的形式表示。   
	charTex.m_delta_y = (float)bitmap_glyph->top - height;   //Top: 类似于字形槽的bitmap_top字段。  

	glGenTextures(1, &charTex.m_texID);
	glBindTexture(GL_TEXTURE_2D, charTex.m_texID);

	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RED,
		width,
		height,
		0,
		GL_RED,
		GL_UNSIGNED_BYTE,
		m_FT_Face->glyph->bitmap.buffer);  //指定一个二维的纹理图片  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);                            //glTexParameteri():纹理过滤  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);                                //纹理进行混合  

	//delete[] pBuf;
	return charTex.m_chaID;
}


xCharTexture* xFreeTypeLib::getTextChar(wchar_t ch)
{
	if(g_TexID[ch].m_texID == 0)
		loadChar(ch);
	return &g_TexID[ch];
}