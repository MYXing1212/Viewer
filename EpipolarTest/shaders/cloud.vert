#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform sampler2D textureCloud;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform vec3 color = vec3(1.0f, 0.0f, 0.0f);
vec3 eyeNormal = vec3(0.0f, 1.0f, 0.0f);
vec4 eyePosition = vec4(0, 0, 3.0f, 1.0f);
vec3 lightDirection = vec3(0, 0, 1.0f);

vec3 albedo;

uniform vec3 lightPos = vec3(1.2f, 1.0f, 2.0f);

uniform bool isLighting;
uniform float inverseWidth;
uniform float inverseHeight;

out vec2 TexCoord;

vec3 points[8];

void main()
{
	vec2 texCoord = aPos + 0.5 * vec2(inverseWidth, inverseHeight);
	vec3 point = texture2D(textureCloud, texCoord).xyz;

	points[0] = texture2D(textureCloud, texCoord + vec2(-inverseWidth, 0.0)).xyz;
	points[1] = texture2D(textureCloud, texCoord + vec2(inverseWidth, 0.0)).xyz;
	points[2] = texture2D(textureCloud, texCoord + vec2(0.0, -inverseHeight)).xyz;
	points[3] = texture2D(textureCloud, texCoord + vec2(0.0, inverseHeight)).xyz;
	points[4] = texture2D(textureCloud, texCoord + vec2(-inverseWidth, -inverseHeight)).xyz;
	points[5] = texture2D(textureCloud, texCoord + vec2(-inverseWidth, inverseHeight)).xyz;
	points[6] = texture2D(textureCloud, texCoord + vec2(inverseWidth, -inverseHeight)).xyz;
	points[7] = texture2D(textureCloud, texCoord + vec2(inverseWidth, inverseHeight)).xyz;

	for(int i = 0;i<8;++i)
	{
		if(abs(points[i].z - point.z) > 5.0)
		{
			point.z = 0.0;
			break;
		}
	}

	if(point.z == 0.0)
		point.z = 10000000.0;

	// 算法线是平滑之前的
	vec3 normal = normalize(cross(points[0] - points[1], points[3] - points[2]));

	//albedo = gl_Color.rgb;
	albedo = color;
	//eyeNormal = normalize(vec3(0, 0, 1) * normal);
	//eyeNormal = normalize(gl_NormalMatrix * normal);
	if(isLighting)
	{
		if(eyeNormal.z < 0.0)
		{
			eyeNormal = -eyeNormal;
		}
		eyePosition = view * model * vec4(point, 1.0);
		//lightDirection = normalize(lightPos.position.xyz);
	}
	gl_Position = projection * view * model * vec4(point, 1.0);
	TexCoord = aTexCoord;
}