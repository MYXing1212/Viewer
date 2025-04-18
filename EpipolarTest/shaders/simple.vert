#version 450 core
layout (location = 0) in vec3 aPos;		// the position variable has attribute position 0
layout (location = 1) in vec2 aTexCoords;

out VS_OUT {
	vec2 texCoords;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
	vs_out.texCoords = aTexCoords;
	gl_Position = projection * view * model * vec4(aPos, 1.0);
}