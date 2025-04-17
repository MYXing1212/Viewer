#version 450 core

uniform sampler2D texturePhase;

void main()
{
	vec3 color = vec3(texture2D(texturePhase, gl_TexCoord[0].st).r * 0.0045);
	// vec3 color = vec3(texture2D(texturePhase, gl_TexCoord[0].st).r / 6.289);
	gl_FragColor = vec4(color, 1.0);
}