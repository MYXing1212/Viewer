#version 450 core
out vec4 FragColor;

in vec4 color;
in vec3 position;

void main()
{
	FragColor = color;
	//FragColor = vec4(1.0, 1.0, 0.0, 1.0);
}