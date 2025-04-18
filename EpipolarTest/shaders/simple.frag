#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D texture_diffuse1;

void main()
{
	FragColor = texture(texture_diffuse1, TexCoords) + vec4(0.2);
}