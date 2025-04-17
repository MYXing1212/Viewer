#version 450 core
out vec4 FragColor;

in vec2 position;

uniform vec2 fringeAB = vec2(110.0f,120.0f);
uniform vec2 resolution = vec2(1920, 1080);

uniform bool useROI = false;
uniform vec2 anchor = vec2(0.0f, 0.0f);
uniform float filterSize = 3.0f;    // 3*3

uniform vec3 pointsColor;
//uniform sampler1D lut;

uniform bool prjIlluminance = false;
uniform float illuminanceValue = 0.0f;
//uniform vec4 freqPhase;

// data[0] x方向条纹宽度
// data[1] y方向条纹宽度
// data[2] phi/2pi
uniform vec3 data;  

float PI = 3.1415926535f;
float PI_2 = PI * 2.0f;

uniform float lut[256];
float findVal(float g)
{
	for(unsigned int i = 0;i<255;i++)
	{
		float i_float = float(i);
		if(lut[i] < g && lut[i+1] >= g)
		{
			if(lut[i+1]-lut[i] > 0.00001)
				return (i_float+(g-lut[i])/(lut[i+1]-lut[i]));
			else 
				return (i_float);
		}
	}
	return 0.0f;

	//if(g>100.0)
	//	return 255.0f;
	//else
	//	return 0;
}

//uniform sampler1D lut;
//float findVal(float g)
//{
//	for(unsigned int i = 0;i<255;i++)
//	{
//		float i_float = float(i);

//		float v0 = texture(lut, i_float/255.0).r;
//		float v1 = texture(lut, (i_float+1.0)/255.0).r;
//		if(v0 < g && v1 >= g)
//		{
//			//return 255.0f;
			
//			if(v1-v0 > 0.00001)
//				return (i_float+(g-v0)/(v1-v0));
//			else 
//				return (i_float);
//		}
//	}
//	return 0.0f;
//	//if(g>100.0)
//	//	return 255.0f;
//	//else
//	//	return 0;
//}

//uniform sampler2D lut;
//float findVal(float g)
//{
//	//for(int i = 0;i<255;i++)
//	//{
//	//	float i_float = float(i);

//	//	float v0 = texture2D(lut, vec2(0.0, i_float/255.0, 0.0)).r;
//	//	float v1 = texture2D(lut, vec2(0.0, (i_float+1.0)/255.0, 0.0)).r;

//	//	if(v0 < g && v1 >= g)
//	//	{
//	//		//return 255.0f;
			
//	//		if(v1-v0 > 0.00001)
//	//			return (i_float+(g-v0)/(v1-v0));
//	//		else 
//	//			return (i_float);
//	//	}
//	//}

//	//return  i_float;
//	return  texture2D(lut, vec2(100.0, 0.0)).r;
//	//return  texture2D(lut, vec2(0.5, 0.0)).r;
//	//return  texture2D(lut, vec2(0.0, 100.0)).r;
//	//return  texture2D(lut, vec2(100.0, 0.0)).r * 255.0;

//	//return 0.0f;
//	//if(g>100.0)
//	//	return 255.0f;
//	//else
//	//	return 0;
//}

void main()
{
	if(prjIlluminance == false)
	{
		float a = 0;
		if(useROI)
		{
			float sz = 	(filterSize - 1.0f)/2.0f;
			if((gl_FragCoord.x - 0.5f) < (anchor.x - sz)
				|| (gl_FragCoord.x - 0.5f)  > (anchor.x + sz)
				|| (gl_FragCoord.y - 0.5f) < (anchor.y - sz)
				|| (gl_FragCoord.y - 0.5f) > (anchor.y + sz)
				)
				FragColor = vec4(vec3(0.0f), 1.0f);
			else 
			{
				float sx = PI_2 * data[0] / filterSize;
				float sy = PI_2 * data[1] / filterSize;
				a = fringeAB.x * cos(sx * (gl_FragCoord.x - 0.5f - anchor.x + sz) 
					+ sy * (gl_FragCoord.y - 0.5f - anchor.y + sz) + data[2]) + fringeAB.y;
				FragColor = vec4(pointsColor * a / 255.0f, 1.0f);
				//FragColor = vec4(pointsColor * findVal(a) / 255.0f, 1.0f);
				//FragColor = vec4(vec3(1.0f), 1.0f);
			}	
			//FragColor = vec4(1.0f);
		}
		else
		{			
			//float A = 96.550477;
			//float B = 143.449523;

			if(data[0] > 0)
			{
				float x = (gl_FragCoord.x - 0.5f +1.0f) / data[0] * PI_2;
				a = fringeAB.x * cos(x + data[2] * 2.0f * PI) + fringeAB.y; 
			}

			if(data[1] > 0)
			{
				float y = (gl_FragCoord.y- 0.5f + 1.0f) / data[1] * PI_2;
				a = fringeAB.x * cos(y + data[2] * 2.0f * PI) + fringeAB.y; 
			}

			//FragColor = vec4(pointsColor * a / 255.0f, 1.0f);
			FragColor = vec4(pointsColor * findVal(a) / 255.0f, 1.0f);
		}		
	}
	else
	{
		FragColor = vec4(pointsColor * illuminanceValue, 1.0f);
	}
	

	//if(data[0] > 0)
	//{
	//	float x = (gl_FragCoord.x-0.5f + 1.0f) / data[0] * PI_2;
	//	a = 110.0f * cos(x + data[2] * PI) + 120.0f; 
	//}

	//if(data[1] > 0)
	//{
	//	float y = (gl_FragCoord.y-0.5f + 1.0f) / data[1] * PI_2;
	//	a = 110.0f * cos(y + data[2] * PI) + 120.0f; 
	//}

	//float a = 120+110*cos(2*3.1415926535*(data[0] * (gl_FragCoord.x-0.5) / Mlow 
	//	+ data[1] * (gl_FragCoord.y-0.5) / Nlow) + data[2]);

	//FragColor = vec4(vec3(0), 1.0f);
	
	//FragColor = vec4(1.0, 1.0, 0, 1.0);

	//if(cos(freqPhase.x * gl_FragCoord.x + freqPhase.y * gl_FragCoord.y + freqPhase.z)>0)
	//	FragColor = vec4(pointsColor, 1.0f);
	//else 
	//	FragColor = vec4(0.0f);
}
