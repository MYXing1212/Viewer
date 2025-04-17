#version 450 core
layout (location = 0) in vec3 aPos;			// the position variable has attribute position 0

out vec3 position;
out vec3 posInViewSpace;
out vec3 vertColor;

uniform float pointSize = 10.0f;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform bool render2D = false;
uniform bool renderGraySeq = false;

uniform vec2 startPt;
uniform vec2 endPt;
uniform vec2 grayRange;	// 先小后大
uniform sampler2D graySeqTexture;

void main()
{
    float rval = 0;
	if(!render2D)
	{	//gl_Position = vec4(0.0f, 0.0f, 0.0f, 1.0f);
		gl_Position = projection * view * model * vec4(aPos, 1.0);
		//gl_Position.z = 0.0;
	}
	else 
	{
		if(renderGraySeq)
		{
			vec2 dir = endPt - startPt;
			vec2 coord = startPt + dir * aPos.x;
			rval = texture(graySeqTexture, coord).r;
			float val = ( rval - grayRange[0]) / (grayRange[1] - grayRange[0]);	
			gl_Position = vec4(aPos.x * 2.0f - 1.0f, val*2.0f-1.0f, aPos.z, 1.0);
		}
		else
		{
			gl_Position = vec4(aPos, 1.0);
		}
	}

	gl_PointSize = pointSize;
	position = aPos;
	vec4 tmpPos = model * vec4(aPos, 1.0);
	posInViewSpace = tmpPos.xyz;

	// 调试时使用
	vertColor = vec3(rval, rval, rval);
}