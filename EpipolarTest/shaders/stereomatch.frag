#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D textureLeft;
uniform sampler2D textureRight;
uniform float inverseWidth;
uniform float inverseHeight;

void main()
{
	float disparity = 0.0;
	vec2 texCoord = TexCoords.st;
	float tempPhaseRight1 = 0.0;
	float tempPhaseRight2 = 0.0;
	float tempPhaseLeft = texture2D(textureLeft, texCoord.st).r;
	if(tempPhaseLeft > 0.0)
	{
		float idxX = texCoord.s;
		for(float x = 0.0; x < 1.0; x+= inverseWidth)
		{
			tempPhaseRight1 = texture2D(textureRight, vec2(x, texCoord.t)).r;
			tempPhaseRight2 = texture2D(textureRight, vec2(x + inverseWidth, texCoord.t)).r;
			if(tempPhaseRight1 > 0.0 && tempPhaseRight2 > 0.0)
			{
				if(tempPhaseLeft >= tempPhaseRight1 && tempPhaseLeft <= tempPhaseRight2 && abs(tempPhaseLeft - tempPhaseRight1) < 0.5)
				{
					float weight1 = tempPhaseLeft - tempPhaseRight1;
					float weight2 = tempPhaseRight2 - tempPhaseLeft;
					float matchLeftX = idxX;
					float matchRightX = (x * weight2 + (x + inverseWidth) * weight1) / (weight1 + weight2);
					disparity = matchLeftX - matchRightX;
					break;
				}
			}
		}
	}
	FragColor = vec4(disparity, 0.0, 0.0, 1.0);
}
