#version 450 core
layout (location = 0) in vec3 aPos;		// the position variable has attribute position 0

uniform mat4 transform;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	//gl_Position = transform * vec4(aPos, 1.0);
	gl_Position = projection * view * model * vec4(aPos, 1.0);
	gl_PointSize = gl_Position.z * 3.0f;
}