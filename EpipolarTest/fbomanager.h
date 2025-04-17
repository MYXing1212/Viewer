#ifndef FBOMANAGER_H
#define FBOMANAGER_H

#include<GL/glew.h>

class FBOManager
{
public:
	// һ�������������ֱ�Ϊ GL_COLOR_ATTACHMENT0 GL_TEXTURE_2D textureID�����ɵ������ID
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