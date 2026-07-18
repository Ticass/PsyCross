#version 450

layout(location=0) in ivec2 inXY;
layout(location=1) in ivec2 inPageClut;
layout(location=2) in float inZ;
layout(location=3) in uvec4 inUvBrightDither;
layout(location=4) in vec4 inColor;
layout(location=5) in ivec4 inExtra;
layout(location=6) in vec3 inPgxp;
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

layout(location=0) out vec4 vTexcoord;
layout(location=1) out vec4 vColor;
layout(location=2) out vec4 vPageClut;
layout(location=3) out float vFog;
layout(location=4) out float vIs3D;
layout(location=5) out vec3 vViewPos;
layout(location=6) out vec4 vShadowViewPos;

void main() {
    vTexcoord = vec4(inUvBrightDither);
    vTexcoord.xy += vec2(inExtra.xy) * 0.5;
    vColor = inColor;
    vColor.rgb *= float(inUvBrightDither.z);

    float page = float(inPageClut.x);
    float clut = float(inPageClut.y);
    vPageClut = vec4(fract(page / 16.0) * 1024.0,
                     floor(page / 16.0) * 256.0,
                     fract(clut / 64.0),
                     floor(clut / 64.0) / 512.0);
    vPageClut.xy += vec2(0.00025);
    vPageClut.zw += vec2(0.00025);

    bool pgxp = ubo.renderInfo.x > 0.5 && inPgxp.z > 0.0;
    if (pgxp) {
        vec4 base = ubo.projection * vec4(inPgxp.xy, inZ, 1.0);
        float w = inPgxp.z;
        if (ubo.renderInfo.y > 0.0) w = min(w, ubo.renderInfo.y);
        gl_Position = base * w;
    } else {
        gl_Position = ubo.projection * vec4(vec2(inXY), inZ, 1.0);
    }

    vIs3D = (ubo.renderInfo.x > 0.5) ? float(inPgxp.z > 0.0) : 1.0;
    vFog = clamp(float(inExtra.z) / 127.0, 0.0, 1.0);
    vViewPos = inViewPos;
    float shadowInvZ = inViewPos.z > 0.0 ? 1.0 / inViewPos.z : 0.0;
    float shadowClipW = pgxp ? ((ubo.renderInfo.y > 0.0) ? min(inPgxp.z, ubo.renderInfo.y) : inPgxp.z) : 1.0;
    vShadowViewPos = vec4(inViewPos * shadowInvZ, shadowInvZ) * shadowClipW;
}
