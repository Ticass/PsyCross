#version 450

layout(location=0) out vec2 vUv;

void main() {
    vec2 p = vec2(float((gl_VertexIndex & 1) << 2) - 1.0,
                  float((gl_VertexIndex & 2) << 1) - 1.0);
    vUv = (p + 1.0) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
