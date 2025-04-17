#ifndef FBOMANAGER_H
#define FBOMANAGER_H

#include<GL/glew.h>

class FBOManager
{
public:
	// 一般这三个参数分别为 GL_COLOR_ATTACHMENT0 GL_TEXTURE_2D textureID是生成的纹理的ID
	FBOManager(GLenum tempAttachment, GLenum tempKind, GLuint textureID);
	
	void updateFBO(GLuint textureID);
	void updateFBO(GLenum, GLuint);
	void updateRBO(GLint, GLint);
	void bindFBO();
	void unbindFBO();
	void bindRBO();
	void unbindRBO();

private:
	bool activeRBO = false;
	GLuint frameBufferID;
	GLuint renderBufferID;
	GLenum attachment;
	GLenum kind;
};

#endif