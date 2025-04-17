#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D textureLeft;
uniform sampler2D textureRight;
uniform float inverseWidth;
uniform float inverseHeight;

float std(float x, float y)
{
	float sum = 0.0;
	float count = 0.0;
	for(float j = -7.0 * inverseHeight; j < 8.0 * inverseHeight; j+= inverseHeight)
	{
		for(float i = -7.0 * inverseWidth; i < 8.0 * inverseWidth; i+= inverseWidth)
		{
			sum = sum + texture2D(textureLeft, vec2(x + i, y + j)).r;
			count = count + 1.0;
		}
	}
	float meanValue = sum / count;
	if(meanValue < 0.1 || meanValue > 0.99)
		return 0.0;
	else 
	{
		sum = 0.0;
		for(float j = -7.0*inverseHeight; j<8.0*inverseHeight;j+=inverseHeight)
		{
			for(float i = -7.0*inverseWidth; i<8.0*inverseWidth;i+=inverseWidth)
			{
				sum = sum +(meanValue - texture2D(textureLeft, vec2(x+i, y+j)).r) * (meanValue - texture2D(textureLeft, vec2(x+i, y+j)).r);
			}
		}
		return sqrt(sum / count);
	}
}

float sad(float x1, float x2, float y)
{
	float sum = 0.0;

	for(float j = -7.0 * inverseHeight; j < 8.0 * inverseHeight; j+= inverseHeight)
	{
		for(float i = -7.0 * inverseWidth; i < 8.0 * inverseWidth; i+= inverseWidth)
		{
			sum += abs(texture2D(textureLeft, vec2(x1 + i, y + j)).r - texture2D(textureRight, vec2(x2 + i, y+j))).r;
		}
	}
	return sum;
}

void main()
{
	float disparity = 0.0;
	vec2 texCoord = TexCoords.st;
	if(std(texCoord.s, texCoord.t) > 0.01)
	{
		float sadValue = 0.0;
		float minSadValue = 999999999.0;
		float x = 0.0;
		float matchRight = 0.0;
		for(x = texCoord.s + 30.0 * inverseWidth; x < texCoord.s + 128.0 * inverseWidth; x += inverseWidth)
		{
			sadValue = sad(texCoord.s, x, texCoord.t);
			if(sadValue < minSadValue && sadValue < 255.0 * 0.2)
			{
				minSadValue = sadValue;
				matchRight = x;
			}
		}
		if(minSadValue < 255.0)
		{
			float windowValues[3];
			windowValues[0] = sad(texCoord.s, matchRight - inverseWidth, texCoord.t);
			windowValues[1] = minSadValue;
			windowValues[2] = sad(texCoord.s, matchRight + inverseWidth, texCoord.t);
			float matchRightSubPixel = matchRight + inverseWidth * (0.5 * (windowValues[0] - windowValues[2]) / (windowValues[0] - 2.0 * windowValues[1] + windowValues[2]));
			disparity = texCoord.s - matchRightSubPixel;
		}
	}
	FragColor = vec4(disparity, 0.0, 0.0, 1.0);
}