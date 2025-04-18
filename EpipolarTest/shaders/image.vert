#version 450 core
layout (location = 0) in vec3 aPos;	
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;


uniform vec2 offset;
uniform float scale;

void main()
{
	gl_Position = vec4(aPos, 1.0);
	TexCoord = offset + aTexCoord*scale;
}