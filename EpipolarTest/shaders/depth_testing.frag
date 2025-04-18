#version 450 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture1;

float near = 0.1f;
float far = 100.0f;

float LinearizeDepth(float depth)
{
	float z = depth * 2.0 - 1.0 ;	// back to NDC
	return (2.0 * near * far) / (far + near - z *(far -near));
}

void main()
{
	//float depth = LinearizeDepth(gl_FragCoord.z) / 10.0f;		// divide by far for demonstration
	//FragColor = vec4(vec3(depth), 1.0);
	FragColor = texture(texture1, TexCoords);
}
