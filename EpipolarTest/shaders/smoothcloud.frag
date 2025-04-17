#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D textureCloud;
uniform float inverseWidth;
uniform float inverseHeight;

float boxFilter(vec2 uv)
{
	float count = 0.0;
	float centerZ = texture2D(textureCloud, uv).z;
	float sumZ = 0.0;
	float z = 0.0;
	for(float j = -2.0; j<3.0;j+=1.0)
	{
		for(float i = -2.0; i< 3.0; i+= 1.0)
		{
			z = texture2D(textureCloud, uv + vec2(i * inverseWidth, j * inverseHeight)).z;
			if(abs(z - centerZ) < 5.0 && z>0.0 && z < 1000000.0)
			{
				sumZ += z;
				count = count + 1.0;
			}
		}
	}
	return sumZ / count;
}

void main()
{
	vec3 point = texture2D(textureCloud, TexCoords.st).xyz;
	float z = point.z;
	if(z > 0.0)
		z = boxFilter(TexCoords.st);
	FragColor = vec4(point.x, point.y, z, 0.0);
}