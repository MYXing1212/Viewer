#version 450 core
layout (location = 0) in vec3 aPos;	

void main()
{
	//gl_Position = projection * vec4(aPos, 1.0);
	//gl_Position = projection * model* vec4(aPos, 1.0);
	gl_Position = vec4(aPos, 1.0);
	//color = aColor;
	//gl_PointSize = 3.0f;
}