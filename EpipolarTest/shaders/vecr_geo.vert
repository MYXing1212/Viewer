#version 450 core
layout (location = 0) in vec3 aPos;			// the position variable has attribute position 0



uniform vec2 startPt;
uniform vec2 endPt;
uniform vec2 grayRange;	// 先小后大
uniform sampler2D graySeqTexture;

void main()
{
    float rval = 0;
	vec2 dir = endPt - startPt;
	vec2 coord = startPt + dir * aPos.x;
	rval = texture(graySeqTexture, coord).r;
	float val = ( rval - grayRange[0]) / (grayRange[1] - grayRange[0]);	
	gl_Position = vec4(aPos.x * 2.0f - 1.0f, val*2.0f-1.0f, aPos.z, 1.0);
}