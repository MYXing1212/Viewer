#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

//// ÂË²¨
uniform sampler2D texturebeat;
// uniform vec2 invAOResolutionx;
uniform float inverseWidth;
float blurFalloff = 0.0;
float sharpness = 5.0;
float blurRadius = 15.0;
/////

float blurFunction(vec2 uv, float r, float centerC, float centerD, inout float wTotal)
{
	float c = texture2D(texturebeat, uv).x;
	float d = c;

	float ddiff = centerD - d;
	float w = exp(-r * r*blurFalloff - ddiff * ddiff * sharpness);
	// float w = exp(-ddiff * ddiff * sharpness);
	wTotal += w;

	return w * c;
}

void main()
{
	//////xÂË²¨  ///////////////
	float b = 0;
	float wTotal = 0;
	float centerC = texture2D(texturebeat, TexCoords.st).x;
	float centerD = centerC;

	for(float r =  -blurRadius; r<=blurRadius; ++r)
	{
		vec2 uv = TexCoords.st + vec2(r * inverseWidth, 0);
		b += blurFunction(uv, r, centerC, centerD, wTotal);
	}

	FragColor = vec4(b / wTotal);
	// gl_FragColor = vec4(centerC);
}
////////////////////////////
