#include"xFreeTypeLib.h"

using namespace std;

GLuint texture_id[MAX_NO_TEXTURE];
static xCharTexture g_TexID[65536];

void xFreeTypeLib::load(const char* font_file, int _w, int _h)
{
	FT_Library library;
	if (FT_Init_FreeType(&library))
		exit(0);
	// ����һ�����壬ȡĬ�ϵ�Face��һ��ΪRegualer
	if (FT_New_Face(library, font_file, 0, &m_FT_Face))
		exit(0);
	// ѡ���ַ���
	FT_Select_Charmap(m_FT_Face, FT_ENCODING_UNICODE);
	m_w = _w; 
	m_h = _h;
	m_FT_Face->num_fixed_sizes;
	// ��СҪ��64�����ǹ涨�������Ϳ�����.
	// FT_Set_Char_Size(m_FT_Face, 0, m_w<<6, 96, 96);
	// ��������ƶ��ַ���Ⱥ͸߶ȵ��ض�����
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
	// װ������ͼ�����βۣ�����Ĩ����ǰ������ͼ��)
	if (FT_Load_Char(m_FT_Face, ch, FT_LOAD_RENDER))
		return 0;

	xCharTexture& charTex = g_TexID[ch];

	//�õ���ģ  
	FT_Glyph glyph;
	//������ͼ������β۸��Ƶ��µ�FT_Glyph����glyph�С������������һ�������벢������glyph��   
	if (FT_Get_Glyph(m_FT_Face->glyph, &glyph))
		return 0;

	//ת����λͼ  
	FT_Render_Glyph(m_FT_Face->glyph, FT_RENDER_MODE_LCD);//FT_RENDER_MODE_NORMAL  );   
	FT_Glyph_To_Bitmap(&glyph, ft_render_mode_normal, 0, 1);
	FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;

	//ȡ��λͼ����  
	FT_Bitmap& bitmap = bitmap_glyph->bitmap;

	//��λͼ���ݿ����Լ��������������.�����ɿ��Ի�����Ҫ�Ķ��������ˡ�  
	int width = bitmap.width;
	int height = bitmap.rows;

	m_FT_Face->size->metrics.y_ppem;		//�������뵽�豸�ռ�  
	m_FT_Face->glyph->metrics.horiAdvance;  //ˮƽ�ı�����  

	charTex.m_Width = width;
	charTex.m_Height = height;
	charTex.m_adv_x = m_FT_Face->glyph->advance.x / 64.0f;  //�������  
	charTex.m_adv_y = m_FT_Face->size->metrics.y_ppem;        //m_FT_Face->glyph->metrics.horiBearingY / 64.0f;  
	charTex.m_delta_x = (float)bitmap_glyph->left;           //left:����ԭ��(0,0)������λͼ��������ص�ˮƽ����.�����������ص���ʽ��ʾ��   
	charTex.m_delta_y = (float)bitmap_glyph->top - height;   //Top: ���������β۵�bitmap_top�ֶΡ�  

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
		m_FT_Face->glyph->bitmap.buffer);  //ָ��һ����ά������ͼƬ  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);                            //glTexParameteri():�������  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);                                //������л��  

	//delete[] pBuf;
	return charTex.m_chaID;
}

GLuint xFreeTypeLib::loadCharNew(wchar_t ch, xCharTexture &charTex)
{
	//if (g_TexID[ch].m_texID)
	//	return g_TexID[ch].m_texID;
	// װ������ͼ�����βۣ�����Ĩ����ǰ������ͼ��)
	if (FT_Load_Char(m_FT_Face, ch, FT_LOAD_RENDER))
		return 0;

	//xCharTexture& charTex = g_TexID[ch];

	//�õ���ģ  
	FT_Glyph glyph;
	//������ͼ������β۸��Ƶ��µ�FT_Glyph����glyph�С������������һ�������벢������glyph��   
	if (FT_Get_Glyph(m_FT_Face->glyph, &glyph))
		return 0;

	//ת����λͼ  
	FT_Render_Glyph(m_FT_Face->glyph, FT_RENDER_MODE_LCD);//FT_RENDER_MODE_NORMAL  );   
	FT_Glyph_To_Bitmap(&glyph, ft_render_mode_normal, 0, 1);
	FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;

	//ȡ��λͼ����  
	FT_Bitmap& bitmap = bitmap_glyph->bitmap;

	//��λͼ���ݿ����Լ��������������.�����ɿ��Ի�����Ҫ�Ķ��������ˡ�  
	int width = bitmap.width;
	int height = bitmap.rows;

	m_FT_Face->size->metrics.y_ppem;		//�������뵽�豸�ռ�  
	m_FT_Face->glyph->metrics.horiAdvance;  //ˮƽ�ı�����  

	charTex.m_Width = width;
	charTex.m_Height = height;
	charTex.m_adv_x = m_FT_Face->glyph->advance.x / 64.0f;  //�������  
	charTex.m_adv_y = m_FT_Face->size->metrics.y_ppem;        //m_FT_Face->glyph->metrics.horiBearingY / 64.0f;  
	charTex.m_delta_x = (float)bitmap_glyph->left;           //left:����ԭ��(0,0)������λͼ��������ص�ˮƽ����.�����������ص���ʽ��ʾ��   
	charTex.m_delta_y = (float)bitmap_glyph->top - height;   //Top: ���������β۵�bitmap_top�ֶΡ�  

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
		m_FT_Face->glyph->bitmap.buffer);  //ָ��һ����ά������ͼƬ  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);                            //glTexParameteri():�������  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);                                //������л��  

	//delete[] pBuf;
	return charTex.m_chaID;
}


xCharTexture* xFreeTypeLib::getTextChar(wchar_t ch)
{
	if(g_TexID[ch].m_texID == 0)
		loadChar(ch);
	return &g_TexID[ch];
}