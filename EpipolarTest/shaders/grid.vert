#version 450 core
layout (location = 0) in vec3 aPos;	

out vec3 position;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float minValue;
uniform float maxValue;
uniform vec3 axisScale = vec3(2.0f, 2.0f, 100.0f);


void main()
{
	float tmp =1.0;
	if(minValue != maxValue)
		tmp = (aPos.z - minValue) / (maxValue - minValue);

	position.x = aPos.x * axisScale[0];
	position.y = aPos.y * axisScale[1];
	position.z =  tmp * axisScale[2];

	gl_Position = projection * view * model * vec4(position, 1.0);
}