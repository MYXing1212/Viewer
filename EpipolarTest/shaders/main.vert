#version 450 core

varying vec3 eyeNormal;
varying vec4 eyePosition;
varying vec3 lightDirection;
varying vec3 albedo;
uniform bool isLighting;

void main()
{
	albedo = gl_Color.rgb;
	gl_Position = ftransform();
	eyeNormal = normalize((gl_NormalMatrix * gl_Normal).xyz);
	if(isLighting)
	{
		if(eyeNormal.z < 0.0)
		{
			//albedo = gl_BackMaterial.diffuse.rgb;
			eyeNormal = -eyeNormal;
		}
		eyePosition = gl_ModelViewMatrix * gl_Vertex;
		lightDirection = normalize(gl_LightSource[0].position.xyz);
	}
}