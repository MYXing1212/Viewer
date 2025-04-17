#version 450 core

in vec2	TexCoord;
out vec4 FragColor;

uniform vec3 eyeNormal;
uniform vec4 eyePosition;
uniform vec3 lightDirection;
uniform vec3 albedo;
uniform bool isLighting;

uniform float shininess = 1.0f;
uniform vec3 diffuse = vec3(0.8, 0.7, 0.7);

void main()
{
	if(isLighting)
	{
		//vec3 lightColor = gl_LightSource[0].diffuse.rgb;
		vec3 lightColor = diffuse;
		float nDotL = dot(eyeNormal, lightDirection);
		float smoothness = shininess;
		//float specularIntensity = pow(max(0.0, dot(eyeNormal, vec3(gl_LightSource[0].halfVector))), smoothness);
		float specularIntensity = 0.0f;
		vec3 specular = lightColor * specularIntensity;
		vec3 diffuse = lightColor * max(0.0, nDotL);
		FragColor = vec4(specular + diffuse * albedo, 1.0);
	}
	else 
		FragColor = vec4(albedo, 1.0);
}
