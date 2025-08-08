#version 450 core

in vec2 fragTexCoord;
out vec4 color;

uniform sampler2D imageTexture;
uniform int imgType;

uniform vec2 normalizeRange = vec2(0.0, 1.0);

void main() 
{
    if (imgType == 0 || imgType == 5) // CV_8UC1 / CV_32FC1
    {
        float gray = texture(imageTexture, fragTexCoord).r;
        if(imgType == 5) // CV_32FC1
        {
            gray = (gray - normalizeRange.x) / (normalizeRange.y - normalizeRange.x);
        }
        color = vec4(gray, gray, gray, 1.0);
    }
    else if(imgType == 16 || imgType == 21) // CV_8UC3 / CV_32FC3
    {
        vec3 val = texture(imageTexture, fragTexCoord).rgb;
        color = vec4(val.b, val.g, val.r, 1.0);
    }
}

//out vec4 FragColor;
//
//in vec2 TexCoord;
//
//uniform bool normalizeValue = false;
//uniform vec2 normalizeRange = vec2(0);
//
//uniform sampler2D texture1;
//
//uniform bool render8bitImage = true;
//
//uniform bool usePseudoColor = false;
//
//// 获取伪彩色 输入scalar 为0~1 float
//vec3 getPseudoColor(float scalar)
//{
//	vec3 color;
//	if (scalar < 0.5f)
//		color.r = 0.0f;
//	else if (scalar < 0.75f)
//		color.r = 4.0f * (scalar - 0.5f);
//	else
//		color.r = 1.0f;
//
//	if (scalar < 0.25f)
//		color.g = 4.0f * scalar;
//	else if (scalar < 0.75f)
//		color.g = 1.0f;
//	else
//		color.g = 1.0f - 4.0f * (scalar - 0.75f);
//
//	if (scalar < 0.25f)
//		color.b = 1.0f;
//	else if (scalar < 0.5f)
//		color.b = 1.0f - 4.0f * (scalar - 0.25f);
//	else
//		color.b = 0.0f;
//
//	color = max(vec3(0.0f), color);
//	color = min(vec3(1.0f), color);
//
//	return color;
//}
//
//void main()
//{
//	if(render8bitImage)
//		FragColor = vec4(texture(texture1, TexCoord).rgb, 1.0);
//	else
//	{
//		float value = texture(texture1, TexCoord).r;
//		if(normalizeValue)
//		{
//			if(normalizeRange == vec2(0))
//				value = value / 255.0;
//			else 
//				value = (value - normalizeRange[0]) / (normalizeRange[1] - normalizeRange[0]);
//		}
//
//		if(usePseudoColor)
//			FragColor = vec4(getPseudoColor(value), 1.0);
//		else 
//			//FragColor = vec4(0.5f, 0.5f, 0.0f, 1.0);
//			FragColor = vec4(value, value, value, 1.0);
//	}
//}