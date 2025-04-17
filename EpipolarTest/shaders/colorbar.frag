#version 450 core
out vec4 FragColor;

uniform float view_width;
uniform float view_height;

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
	FragColor = vec4(getPseudoColor((view_height * 0.9 - gl_FragCoord.y) / (view_height * 0.8)), 1.0f);
}