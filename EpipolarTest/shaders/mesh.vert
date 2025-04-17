#version 450 core
layout (location = 0) in vec3 aPos;	

out vec3 position;
out vec4 color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float minValue;
uniform float maxValue;
uniform vec3 axisScale = vec3(2.0f, 2.0f, 100.0f);

uniform vec2 colorRange;

// 获取伪彩色 输入scalar 为0~1 float
vec3 getPseudoColor(float scalar)
{
	//scalar = (scalar + 1.0) / 2.0;
	vec3 color;
	if (scalar < 0.5f)
		color.x = 0.0f;
	else if (scalar < 0.75f)
		color.x = 4.0f * (scalar - 0.5f);
	else
		color.x = 1.0f;

	if (scalar < 0.25f)
		color.y = 4.0f * scalar;
	else if (scalar < 0.75f)
		color.y = 1.0f;
	else
		color.y = 1.0f - 4.0f * (scalar - 0.75f);

	if (scalar < 0.25f)
		color.z = 1.0f;
	else if (scalar < 0.5f)
		color.z = 1.0f - 4.0f * (scalar - 0.25f);
	else
		color.z = 0.0f;

	color = clamp(color, vec3(0.0), vec3(1.0));
	return color;
}

void main()
{
	float tmp =1.0;
	if(minValue != maxValue)
		tmp = (aPos.z - minValue) / (maxValue - minValue);

	position.x = aPos.x * axisScale[0];
	position.y = aPos.y * axisScale[1];
	position.z =  tmp * axisScale[2];

	gl_Position = projection * view * model * vec4(position, 1.0);
	color = vec4(getPseudoColor(tmp * (colorRange[1] - colorRange[0]) + colorRange[0]), 1.0f);
}