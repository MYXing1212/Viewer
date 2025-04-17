#version 450 core
layout (location = 0) in vec4 vertex;

out vec2 TexCoords;
out vec4 ParticleColor;

uniform vec2 offset;
uniform vec4 color;
uniform mat4 projection;

void main()
{
	float scale = 10.0f;
	ParticleColor = color;
	TexCoords = vertex.zw;
	//gl_Position = vec4(vertex.x, vertex.y, 0.0f, 1.0f);
	gl_Position = projection * vec4((vertex.xy * scale) + offset, 0.0f, 1.0f);
}