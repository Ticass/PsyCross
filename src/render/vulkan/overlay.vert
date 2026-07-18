#version 450

layout(push_constant) uniform OverlayPush {
    vec4 rect;
    vec4 uvRect;
} pc;

layout(location=0) out vec2 vUv;

void main() {
    const vec2 corners[6] = vec2[6](
        vec2(0,0), vec2(0,1), vec2(1,0),
        vec2(1,0), vec2(0,1), vec2(1,1));
    vec2 corner = corners[gl_VertexIndex];
    gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, corner), 0.0, 1.0);
    vUv = mix(pc.uvRect.xy, pc.uvRect.zw, corner);
}
