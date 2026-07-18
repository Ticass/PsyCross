#version 450

layout(location=7) in vec3 inNormal;
layout(location=8) in vec3 inViewPos;

layout(set=0,binding=0) uniform RendererUniforms {
    mat4 projection;
    mat4 projection3D;
    vec4 fogColorStrength;
    vec4 textureInfo;
    vec4 hiresInfo;
    vec4 renderInfo;
    vec4 lightPosRange;
    vec4 lightDirStyle;
    vec4 lightColor;
    vec4 effectInfo;
    mat4 shadowMatrix;
    vec4 shadowParams;
    vec4 shadowClip;
    vec4 framebufferRects[8];
} ubo;

void main() {
    if (inNormal.y < 0.5 || inViewPos.z <= 0.0 || inNormal.x > 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    gl_Position = ubo.shadowMatrix * vec4(inViewPos, 1.0);
}
