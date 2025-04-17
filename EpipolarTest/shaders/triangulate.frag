#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D textureDisparity;

uniform float width;
uniform float height;
uniform vec2 ccLeftRectify;
uniform vec2 ccRightRectify;
uniform vec2 fc;
uniform float tRectify;
uniform float backgroundDisparity;
uniform float zMin;
uniform float zMax;

void main()
{
	vec2 texCoord = TexCoords.st;
	vec3 point = vec3(0.0);
	float disparity = texture2D(textureDisparity, texCoord.st).r  * width;
	if(abs(disparity) > 0.0)
	// if(disparity != backgroundDisparity)
	{
		float matchY = (1.0 - texCoord.t) * height - 0.5 - ccLeftRectify.y;
		float invDisparity = 1.0 / (ccLeftRectify.x - ccRightRectify.x - disparity);
		point.x = tRectify * (texCoord.s * width - 0.5 - ccLeftRectify.x) * invDisparity;
		point.y = tRectify * fc.x * matchY * invDisparity / fc.y;
		point.z = tRectify * fc.x * invDisparity;
		if(point.z < zMin || point.z > zMax)
			point = vec3(0.0);
	}
	FragColor = vec4(point, 1.0);
}