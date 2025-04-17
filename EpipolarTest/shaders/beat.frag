#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;

const float pi = 3.1415926;
const float pi2 = 2.0 * pi;
const float invPi2 = 1.0 / pi2;

const float threshold = 5.0;
const float lambda1 = 31.0;
const float lambda2 = 32.0;

void main()
{
	float num1inb = lambda2 / (lambda2 - lambda1);
	vec3 src1 = texture2D(texture1, TexCoords.st).rgb;
	vec3 src2 = texture2D(texture2, TexCoords.st).rgb;
	float phase1 = atan(1.732 * (src1.z - src1.y), 2.0 * src1.x - src1.y - src1.z);	
	float phase2 = atan(1.732 * (src2.z - src2.y), 2.0 * src2.x - src2.y - src2.z);
	float beat  = phase1 - phase2;
	if(beat < 0.0)
		beat += 2.0 * pi;

	FragColor = vec4(beat, 0.0, 0.0, 0.0);
}