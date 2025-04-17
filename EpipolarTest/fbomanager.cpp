#include"fbomanager.h"

FBOManager::FBOManager(GLenum tempAttachment, GLenum tempKind, GLuint textureID)
{
	attachment = tempAttachment;
	kind = tempKind;

	glGenFramebuffers(1, &frameBufferID);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);

	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, kind, textureID, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		break;
	default:
		break;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FBOManager::updateFBO(GLenum tempAttachment, GLuint textureID)
{
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, tempAttachment, kind, textureID, 0);
}

void FBOManager::updateFBO(GLuint textureID)
{
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, kind, textureID, 0);
}

void FBOManager::updateRBO(GLint width, GLint height)
{
	glGenRenderbuffers(1, &renderBufferID);
	glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_RENDERBUFFER, frameBufferID);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferID);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	activeRBO = true;
}

void FBOManager::bindFBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
}

void FBOManager::unbindFBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FBOManager::bindRBO()
{
	glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID);
}

void FBOManager::unbindRBO()
{
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}