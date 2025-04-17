/*******************************************************************
** This code is part of Breakout.
**
** Breakout is free software: you can redistribute it and/or modify
** it under the terms of the CC BY 4.0 license as published by
** Creative Commons, either version 4 of the License, or (at your
** option) any later version.
******************************************************************/
#include "resource_manager.h"

#include <iostream>
#include <sstream>
#include <fstream>

//#define STB_IMAGE_IMPLEMENTATION
//#include"stb_image.h"

// Instantiate static variables
std::map<std::string, Texture2D>    ResourceManager::Textures;
std::map<std::string, Shader>       ResourceManager::Shaders;


Shader ResourceManager::LoadShader(const GLchar *vShaderFile, const GLchar *fShaderFile, const GLchar *gShaderFile, std::string name)
{
	Shaders[name] = loadShaderFromFile(vShaderFile, fShaderFile, gShaderFile);
	return Shaders[name];
}

Shader ResourceManager::GetShader(std::string name)
{
	return Shaders[name];
}

//Texture2D ResourceManager::LoadTexture(const GLchar *file, GLboolean alpha, std::string name)
//{
//	Textures[name] = loadTextureFromFile(file, alpha);
//	return Textures[name];
//}

Texture2D ResourceManager::GetTexture(std::string name)
{
	return Textures[name];
}

void ResourceManager::Clear()
{
	// (Properly) delete all shaders	
	for (auto iter : Shaders)
		glDeleteProgram(iter.second.ID);
	// (Properly) delete all textures
	for (auto iter : Textures)
		glDeleteTextures(1, &iter.second.ID);
}

Shader ResourceManager::loadShaderFromFile(const GLchar *vShaderFile, const GLchar *fShaderFile, const GLchar *gShaderFile)
{
	return Shader(vShaderFile, fShaderFile, gShaderFile);
}

//Texture2D ResourceManager::loadTextureFromFile(const GLchar *file, GLboolean alpha)
//{
//	std::cout << "loadTextureFromFile " << std::string(file) << std::endl;
//	// Create Texture object
//	Texture2D texture;
//	if (alpha)
//	{
//		texture.Internal_Format = GL_RGBA;
//		texture.Image_Format = GL_RGBA;
//	}
//	// Load image
//	int width, height, nrComponents;
//
//	unsigned char *image = stbi_load(file, &width, &height, &nrComponents, 0);
//	std::cout << "width = " << width << std::endl;
//	std::cout << "height = " << height << std::endl;
//	std::cout << "nrComponents = " << nrComponents << std::endl;
//
//	// Now generate texture
//	texture.Generate(width, height, image);	
//
//	// And finally free image data
//	stbi_image_free(image);
//	return texture;
//}