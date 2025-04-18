#version 450 core
layout (location = 0) in vec3 aPos;		// the position variable has attribute position 0
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0);
	TexCoords = aTexCoords;
}