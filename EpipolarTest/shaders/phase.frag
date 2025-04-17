#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texturebeatfx;

const float pi = 3.1415926;
const float pi2 = 2.0 * pi;
const float invPi2 = 1.0 / pi2;
const float threshold = 5.0;
const float lambda1 = 31.0;
const float lambda2 = 32.0;

//// 滤波
// uniform vec2 invAOResolutionx;
uniform float inverseHeight;
float blurFalloff = 0.0;
float sharpness = 5.0;
float blurRadius = 15.0;
/////

float blurFunction(vec2 uv, float r, float centerC, float centerD, inout float wTotal)
{
	float c = texture2D(texturebeatfx, uv).x;
	float d =c;

	float ddiff = centerD - d;
	float w = exp(-r * r*blurFalloff - ddiff * ddiff * sharpness);
	// float w  =exp(-ddiff * ddiff * sharpness);
	wTotal += w;

	return w*c;
}

float INT(float x)
{
	if(x > -1 && x < 0)
		return -1.0f;
	else
		return 0;
}

float round(float x)
{
	if(x - floor(x) < 0.5)
		return  floor(x);
	else 
		return floor(x)+1;
}

void main()
{
	float num1inb = lambda2 / (lambda2 - lambda1);

	vec3 src1 = texture2D(texture1, TexCoords).rgb;
	vec3 src2 = texture2D(texture2, TexCoords).rgb;
	
	float phase1 = atan(1.732 * (src1.z - src1.y), 2.0 * src1.x - src1.y - src1.z);
	float phase2 = atan(1.732 * (src2.z - src2.y), 2.0 * src2.x - src2.y - src2.z);
	
	float b = 0;
	float wTotal = 0;
	float centerC = texture2D(texturebeatfx, TexCoords).x;
	float centerD = centerC;

	for(float r = -blurRadius; r <= blurRadius; ++r)
	{
		vec2 uv = TexCoords.st + vec2(0, r*inverseHeight);
		b += blurFunction(uv, r, centerC, centerD, wTotal);
	}

	float beat = b / wTotal;
	// beat = centerC;

//////////////////////////////////////////////
//	beat = phase1 - phase2;
//	if (beat < 0.0)
//		beat += 2.0 * pi;
//////////////////////////////////////////////

	float beatM = num1inb * beat;
	//-----------------------------------------------------------------------------------
	//	float phase = beatM;
	//	float phase = phase1 + 2.0 * pi * (beatM - phase1) / 2.0 / pi;
	//	float phase = phase1 + 2.0 * pi * round((beatM - phase1) / 2.0 / pi);
	//	float phase = phase1 + 2.0 * pi * floor((beatM - phase1) / 2.0 / pi);
	//	float phase = phase1 + 2.0 * pi * floor((beatM - phase1) / 2.0 / pi + 0.5);
	//	float phase = phase1 + 2.0 * pi * floor(beatM / 2.0 / pi);
	//-----------------------------------------------------------------------------------

	// float phase = phase1 - mod(beatM, 2*pi);
	// if(phase < -pi)
	//	phase += 2.0 * pi;
	// phase += beatM;

	//float phi12 = phase1 - phase2 - 2*pi*INT((phase1 - phase2) / 2 / pi);

	// Kim
	//float phase = phase1 - mod(beatM, 2*pi) + beatM;
	//float phase = beat;
	//float phase = phase1 + 2.0 * pi * floor(beatM / 2.0 / pi);
	float f1 = 1.0f / lambda1;
	float f2 = 1.0f / lambda2;
	float f12 = f1 - f2;
	float phase = phase1 + 2.0 * pi * round((beatM - phase1) / 2.0 / pi);
	//phase = beat;
	//float phase = 0;
	//if(phase2 > phase1)
	//	phase = phase1 + 2.0 * pi * (round(num1inb * (floor(beat / 2.0 / pi) + 1 + (phase1 - phase2) / 2.0 / pi) + 100)-100);
	//else
	//	phase = phase1 + 2.0 * pi * (round(num1inb * (floor(beat / 2.0 / pi) + (phase1 - phase2) / 2.0 / pi) + 100)-100);
	

	//phase = round(num1inb * (floor(beat / 2.0 / pi) + (phase1 - phase2) / 2.0 / pi) + 100);
	//float phase = phase2;	
	
	//if(beatM - phase > pi)
	//	phase += 2 * pi;
	//if(phase - beatM > pi)
	//	phase -= 2*pi;

	// 三频如何计算条制度
	float term1 = 0.866 * (src1.x - src1.y);
	float term2 = src1.z - 0.5 * (src1.x + src1.y);
	float modulation = term1 * term1 + term2 * term2;
	if(modulation * 255 * 255 < 4.0 * threshold)
		phase = -1.0;

	term1 = 0.866 * (src2.x - src2.y);
	term2 = src2.z - 0.5 * (src2.x + src2.y);
	modulation = term1 * term1 + term2 * term2;
	
	if(modulation * 255 *255 < 4.0 * threshold)
		phase = -1.0;
		
//-----------------------------------------------------------------------------------

	FragColor = vec4(phase, 0.0, 0.0, 0.0);
	
//-----------------------------------------------------------------------------------
	//fragColor = vec4(phase, 0.0, 0.0, 0.0);

//-----------------------------------------------------------------------------------
//	fragColor = vec4(beatM / 2.0 / pi, 0.0, 0.0, 0.0);
//	fragColor = vec4((beatM - phase1) / 2.0 / pi * 29.0, 0.0, 0.0, 0.0);
//	fragColor = vec4(round((beatM - phase1) / 2.0 / pi) * 29.0, 0.0, 0.0, 0.0);
//	fragColor = vec4(floor((beatM - phase1) / 2.0 / pi) * 29.0, 0.0, 0.0, 0.0);
//	fragColor = vec4(floor((beatM - phase1) / 2.0 / pi + 0.5) * 29.0, 0.0, 0.0, 0.0);
//	fragColor = vec4(floor(beatM / 2.0 / pi) * 29.0, 0.0, 0.0, 0.0);
//-----------------------------------------------------------------------------------
	//copy.frag还要除2pi
}
