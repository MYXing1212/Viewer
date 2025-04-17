#version 450 core

varying vec4 modelPosition;
varying vec3 previousNormal;
uniform mat4 previousMatrix;
uniform mat4 modelMatrix;
uniform float mirrorY;

void main()
{
	vec4 eyePosition = gl_ModelViewMatrix * gl_Vertex;
	eyePosition.y *= mirrorY;
    gl_Position = gl_ProjectionMatrix * eyePosition;
	modelPosition = modelMatrix * gl_Vertex;
	previousNormal = mat3(previousMatrix) * normalize((gl_NormalMatrix * gl_Normal).xyz);
}