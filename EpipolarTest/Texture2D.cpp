#include"Texture2D.h"

GLuint createTextureRectARB(int width, int height, int index/* = 0*/)
{
	// 创建纹理对象并绑定
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);

	// 设置纹理参数
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_NEAREST);

	// 将纹理关联到FBO
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + index, GL_TEXTURE_RECTANGLE_ARB, tex, 0);

	// 定义纹理数据单元类型
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);
	return tex;
}

GLuint loadTexture(std::string, GLint, GLint)
{

}

// 返回纹理图片
cv::Mat loadTexture(GLuint &textureId, const cv::Mat &img, 
	const bool &loadRGB/* = true*/,
	const GLint &fillMode,
	const GLint &interpolateMode)
{
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fillMode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fillMode);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, interpolateMode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, interpolateMode);

	if (!img.empty())
	{
		// 沿Y轴进行翻转
		cv::flip(img, img, 0);	
		if (loadRGB)
		{
			cv::Mat pic;
			if (img.channels() == 3)
				cv::cvtColor(img, pic, cv::COLOR_BGR2RGB);
			else
				cv::cvtColor(img, pic, cv::COLOR_GRAY2BGR);
			//cv::imshow("img", pic);
			//cv::waitKey(0);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pic.cols, pic.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, (uchar*)pic.data);
			return pic;
		}
		else
		{
			//printf("load Texture mask\n");
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, img.cols, img.rows, 0, GL_RED, GL_UNSIGNED_BYTE, (uchar*)img.data);
			return img;
		}
		
		//glGenerateMipmap(GL_TEXTURE_2D);
	}
	textureId = -1;
	return cv::Mat();
}

GLuint generateTextureRGB(GLint width, GLint height)
{
	GLuint textureColorbuffer;
	glGenTextures(1, &textureColorbuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	return textureColorbuffer;
}