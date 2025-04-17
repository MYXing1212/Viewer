#version 450 core
layout (location = 0) in vec3 aPos;		
layout (location = 1) in vec3 aNorm;	
layout (location = 2) in vec4 aColor;

out vec3 position;
out vec3 Normal;
out vec4 Color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float pointSize = 1.0f;

//layout (rgba32f, binding = 0) uniform image2D input_image;
//layout (rgba32f, binding = 1) uniform image2D output_image;

void main()
{
	//gl_Position = projection * vec4(aPos, 1.0);
	//gl_Position = projection * model* vec4(aPos, 1.0);
	gl_Position = projection * view * model * vec4(aPos, 1.0);
	Color = aColor;
	Normal = aNorm;
	position = aPos;

	gl_PointSize = pointSize;
	//gl_PointSize = 3.0f;

	
	//data = data + 2.0;
	//imageStore(output_image, ivec2(0, 0), vec4(5.0));
}