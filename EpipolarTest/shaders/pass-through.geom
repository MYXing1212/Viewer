#version 450 core
layout (points) in;
layout (triangle_strip, max_vertices = 5) out;

out vec3 fColor;

in VS_OUT {
	vec3 color;
} gs_in[];

void build_house(vec4 position)
{
	fColor = gs_in[0].color;	// gs_in[0] since there's only one input vertex
	gl_Position = position + vec4(-0.2f, -0.2, 0.0, 0.0);	// 1. bottom-left
	EmitVertex();
	gl_Position = position + vec4(0.2, -0.2, 0.0, 0.0);		// 2:bottom-right
	EmitVertex();
	gl_Position = position + vec4(-0.2, 0.2, 0.0, 0.0);		// 3: top-left
	EmitVertex();
	gl_Position = position + vec4(0.2, 0.2, 0.0, 0.0);		// 4 : top-right
	EmitVertex();
	gl_Position = position + vec4(0.0, 0.4, 0.0, 0.0);		// top
	fColor = vec3(1.0f);
	EmitVertex();
	EndPrimitive();
}

void main() {
	build_house(gl_in[0].gl_Position);
}