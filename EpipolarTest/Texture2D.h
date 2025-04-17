/*******************************************************************
** This code is part of Breakout.
**
** Breakout is free software: you can redistribute it and/or modify
** it under the terms of the CC BY 4.0 license as published by
** Creative Commons, either version 4 of the License, or (at your
** option) any later version.
******************************************************************/
#ifndef TEXTURE_H
#define TEXTURE_H

//#include"GLFW/glfw3.h"
#ifdef USE_GLAD
#include<glad/glad.h>	// include glad to get all the required OpenGL headers
#elif defined USE_GLEW
#include<GL/glew.h>
#endif
#include<iostream>
#include<opencv2/opencv.hpp>

// Texture2D is able to store and configure a texture in OpenGL.
// It also hosts utility functions for easy management.
class Texture2D
{
public:
	// Holds the ID of the texture object, used for all texture operations to reference to this particlar texture
	GLuint ID;
	// Texture image dimensions
	GLuint Width, Height;	// Width and height of loaded image in pixels
	// Texture Format
	GLuint Internal_Format;	// Format of texture object
	GLuint Image_Format;	// Format of loaded image
	// Texture configuration;
	GLuint Wrap_S;			// Wrapping mode on S axis
	GLuint Wrap_T;			// Wrapping mode on T axis
	GLuint Filter_Min;		// Filtering mode if texture pixels < screen pixels
	GLuint Filter_Max;		// Filtering mode if texture pixels > screen pixels
	// Constructor (sets default texture modes);
	Texture2D()
		:Width(0), Height(0), Internal_Format(GL_RGB), Image_Format(GL_RGB), Wrap_S(GL_REPEAT), Wrap_T(GL_REPEAT), Filter_Min(GL_LINEAR), Filter_Max(GL_LINEAR)
	{
		glGenTextures(1, &this->ID);
	}

	// Generates texture from image data
	void Generate(GLuint width, GLuint height, unsigned char* data){
		this->Width = width;
		this->Height = height;
		// Create Texture
		glBindTexture(GL_TEXTURE_2D, this->ID);
		glTexImage2D(GL_TEXTURE_2D, 0, this->Internal_Format, width, height, 0, this->Image_Format, GL_UNSIGNED_BYTE, data);
		// Set Texture wrap and filter modes
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, this->Wrap_S);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, this->Wrap_T);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, this->Filter_Min);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, this->Filter_Max);
		// Unbind texture
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	// Binds the texture as the current active GL_TEXTURE_2D texture object
	void Bind() const
	{
		glBindTexture(GL_TEXTURE_2D, this->ID);
	}
};

GLuint createTextureRectARB(int width, int height, int index = 0);

GLuint loadTexture(std::string, GLint, GLint);

// ��������ͼƬ
cv::Mat loadTexture(GLuint &textureId, const cv::Mat &img, 
	const bool &loadRGB = true,
	const GLint &fillMode = GL_REPEAT, 
	const GLint &interpolateMode = GL_NEAREST);

GLuint generateTextureRGB(GLint width, GLint height);



#endif