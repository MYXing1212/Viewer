#version 450 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec2 fragTexCoord;

void main() {
    gl_Position = projection * view * model * vec4(position, 0.0, 1.0);
    gl_Position.y = - gl_Position.y; // Flip Y coordinate for OpenGL
    fragTexCoord = texCoord;
}