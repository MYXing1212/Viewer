#version 450 core
out vec4 FragColor;
layout (depth_greater) out float gl_FragDepth;

void main()
{
	//FragColor = mix(texture(texture1, TexCoord), texture(texture2, vec2(1.0 - TexCoord.x,  TexCoord.y)), mixScale);
	if(gl_FragCoord.x < 640)
		FragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	else 
		FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);

	gl_FragDepth = gl_FragCoord.z + 0.03f;
}
