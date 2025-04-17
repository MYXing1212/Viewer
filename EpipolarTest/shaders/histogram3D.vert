#version 450 core
layout (location = 0) in vec3 aPos;		// the position variable has attribute position 0

out vec4 Color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float topColor;
uniform float height;
uniform vec2 pos;


// 获取伪彩色 输入scalar 为0~1 float
vec3 getPseudoColor(float scalar)
{
	return vec3(scalar*1.0, scalar * 1.0, -scalar * 0.5 + 0.5);
}


void main()
{
	//gl_Position = projection * vec4(aPos, 1.0);
	//gl_Position = projection * model* vec4(aPos, 1.0);
	gl_Position = projection * view * model * vec4(aPos.x + pos.x, aPos.y + pos.y, aPos.z * height, 1.0);

	if(aPos.z > 0)
		Color = vec4(getPseudoColor(topColor), 1.0);
	else 
		Color = vec4(getPseudoColor(0.0), 1.0);
}