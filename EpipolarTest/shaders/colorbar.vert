#version 450 core
layout (location = 0) in vec2 aPos;		// the position variable has attribute position 0

void main()
{
	gl_Position = vec4(aPos, 0.0f, 1.0);
}