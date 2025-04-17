#version 450 core
varying vec3 eyeNormal;
varying vec4 eyePosition;
varying vec3 lightDirection;
varying vec3 albedo;
uniform bool isLighting;

void main()
{
	if(isLighting)
	{
		vec3 lightColor = gl_LightSource[0].diffuse.rgb;
		float nDotL = dot(eyeNormal, lightDirection);
		float smoothness = gl_FrontMaterial.shininess;
		float specularIntensity = pow(max(0.0, dot(eyeNormal, vec3(gl_LightSource[0].halfVector))), gl_FrontMaterial.shininess);
		vec3 specular = lightColor * specularIntensity;
		vec3 diffuse = lightColor * max(0.0, nDotL);
		gl_FragColor = vec4(specular + diffuse * albedo, 1.0);
	}
	else 
		gl_FragColor = vec4(albedo, 1.0);
}