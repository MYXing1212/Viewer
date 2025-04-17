#version 450 core

out vec4 FragColor;

in vec2 TexCoord;

uniform bool normalizeValue = false;
uniform vec2 normalizeRange = vec2(0);

uniform sampler2D texture1;

uniform bool render8bitImage = true;

uniform bool usePseudoColor = false;

// ��ȡα��ɫ ����scalar Ϊ0~1 float
vec3 getPseudoColor(float scalar)
{
	vec3 color;
	if (scalar < 0.5f)
		color.r = 0.0f;
	else if (scalar < 0.75f)
		color.r = 4.0f * (scalar - 0.5f);
	else
		color.r = 1.0f;

	if (scalar < 0.25f)
		color.g = 4.0f * scalar;
	else if (scalar < 0.75f)
		color.g = 1.0f;
	else
		color.g = 1.0f - 4.0f * (scalar - 0.75f);

	if (scalar < 0.25f)
		color.b = 1.0f;
	else if (scalar < 0.5f)
		color.b = 1.0f - 4.0f * (scalar - 0.25f);
	else
		color.b = 0.0f;

	color = max(vec3(0.0f), color);
	color = min(vec3(1.0f), color);

	return color;
}

void main()
{
	if(render8bitImage)
		FragColor = vec4(texture(texture1, TexCoord).rgb, 1.0);
	else
	{
		float value = texture(texture1, TexCoord).r;
		if(normalizeValue)
		{
			if(normalizeRange == vec2(0))
				value = value / 255.0;
			else 
				value = (value - normalizeRange[0]) / (normalizeRange[1] - normalizeRange[0]);
		}

		if(usePseudoColor)
			FragColor = vec4(getPseudoColor(value), 1.0);
		else 
			//FragColor = vec4(0.5f, 0.5f, 0.0f, 1.0);
			FragColor = vec4(value, value, value, 1.0);
	}
}