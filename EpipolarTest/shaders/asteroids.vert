#version 450 core
layout (location = 0) in vec3 aPos;		// the position variable has attribute position 0
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in mat4 aInstanceMatrix;

out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
	gl_Position = projection * view * aInstanceMatrix * vec4(aPos, 1.0);
	TexCoords = aTexCoords;
}