#version 450 core
out vec4 FragColor;

uniform vec3 color;

void main()
{
	FragColor = vec4(color, 1.0);
	//FragColor = vec4(1.0, 1.0, 0.0, 1.0);
}