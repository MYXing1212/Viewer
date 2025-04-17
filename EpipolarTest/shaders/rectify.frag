#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D textureSrc;
uniform sampler2D textureRectifyX;
uniform sampler2D textureRectifyY;
uniform float inverseWidth;
uniform float inverseHeight;

void main()
{
	FragColor = texture2D(textureSrc, vec2(inverseWidth * texture2D(textureRectifyX, TexCoords.st).r, 
		inverseHeight * texture2D(textureRectifyY, TexCoords.st).r));
}
