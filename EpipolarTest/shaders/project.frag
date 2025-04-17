#version 450 core
varying vec4 modelPosition;
varying vec3 previousNormal;
uniform float minDepth;

vec2 encode(vec3 n)
{
	float f = sqrt(8.0 * n.z + 8.0);
	return n.xy / f + vec2(0.5);
}

void main()
{
	if(modelPosition.z < minDepth)
		discard;
	gl_FragColor = vec4(encode(previousNormal), modelPosition.z, 1.0);
}