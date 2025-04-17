#version 450 core

uniform sampler2D textureDepthBuffer;
uniform sampler2D textureSSAO;
uniform vec2 invAOResolution;
uniform float nearClip;
uniform float farClip;
float blurFalloff = 0.0;
float sharpness = 100.0;
float blurRadius = 25.0;

float linearizeDepth(vec2 uv)
{
	float z = texture2D(textureDepthBuffer, uv).x;
	return farClip * nearClip / (z * (farClip - nearClip) - farClip);
}

float blurFunction(vec2 uv, float r, float centerC, float centerD, inout float wTotal)
{
	float c = texture2D(textureSSAO, uv).x;
	float d = linearizeDepth(uv);
	
	float ddiff = centerD - d;
	float w = exp(-r*r*blurFalloff);	// -ddiff * ddiff * sharpness);
	wTotal += w;

	return w*c;
}

void main()
{
	float b = 0;
	float wTotal = 0;
	float centerC = texture2D(textureSSAO, gl_TexCoord[0].st).x;
	float centerD = linearizeDepth(gl_TexCoord[0].st);

	for(float r = -blurRadius; r<= blurRadius; ++r)
	{
		vec2 uv = gl_TexCoord[0].st + vec2(0, r*invAOResolution.y);
		b += blurFunction(uv, r, centerC, centerD, wTotal);
	}

	gl_FragColor = vec4(b / wTotal);
}