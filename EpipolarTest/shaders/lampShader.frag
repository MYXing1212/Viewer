#version 450 core
out vec4 FragColor;

uniform vec3 aColor;

void main()
{
	FragColor = vec4(aColor, 1.0);		// Set all 4 vector valus to 1.0
}
