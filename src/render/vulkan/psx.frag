#version 450

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
layout(set=0,binding=1) uniform sampler2D sceneTexture;
layout(set=0,binding=2) uniform sampler2D framebufferTexture;
layout(set=0,binding=3) uniform sampler2D shadowTexture;

layout(location=0) in vec4 vTexcoord;
layout(location=1) in vec4 vColor;
layout(location=2) in vec4 vPageClut;
layout(location=3) in float vFog;
layout(location=4) in float vIs3D;
layout(location=5) in vec3 vViewPos;
layout(location=6) in vec4 vShadowViewPos;
layout(location=0) out vec4 outColor;

const vec2 VRAM_TEXEL = vec2(1.0 / 1024.0, 1.0 / 512.0);

vec2 readVRAM(vec2 uv) {
    return floor(texture(sceneTexture, uv).rg * 255.0 + 0.5) / 255.0;
}

float packRG(vec2 rg) {
    return (rg.y * 256.0 + rg.x) * 255.001;
}

vec4 decode1555(float packed) {
    vec4 value = fract(floor(packed / vec4(1.0, 32.0, 1024.0, 32768.0)) / 32.0);
    return vec4(value.rgb, packed == 0.0 ? 0.0 : 1.0 - value.a * 16.0);
}

float shadowLinearDepth(float windowDepth) {
    float n = ubo.shadowClip.x, f = ubo.shadowClip.y;
    return (n * f) / max(f - windowDepth * (f - n), 1e-6);
}

float samplePSX(vec2 tc, int format) {
    if (format == 0) {
        vec2 comp = readVRAM((tc * vec2(0.25, 1.0) + vPageClut.xy) * VRAM_TEXEL);
        int nibble = int(fract(tc.x / 4.0 + 0.0001) * 4.0);
        float value = comp[nibble / 2] * (255.001 / 16.0);
        float lo = fract(value) * 16.0;
        float hi = floor(value);
        float index = ((nibble & 1) == 0) ? lo : hi;
        vec2 clutUv = vPageClut.zw + vec2(index * VRAM_TEXEL.x, 0.0);
        return packRG(readVRAM(clutUv));
    }
    if (format == 1) {
        vec2 comp = readVRAM((tc * vec2(0.5, 1.0) + vPageClut.xy) * VRAM_TEXEL);
        float index = comp[int(mod(tc.x, 2.0))] * 255.001;
        return packRG(readVRAM(vPageClut.zw + vec2(index * VRAM_TEXEL.x, 0.0)));
    }
    return packRG(readVRAM((tc + vPageClut.xy) * VRAM_TEXEL));
}

vec4 sampleNative(vec2 tc, int format) {
    if (ubo.renderInfo.z > 0.5 && vIs3D > 0.5) {
        vec2 f = fract(tc);
        vec2 p = floor(tc);
        float a = samplePSX(p, format);
        float b = samplePSX(p + vec2(1,0), format);
        float c = samplePSX(p + vec2(0,1), format);
        float d = samplePSX(p + vec2(1,1), format);
        float coverage = mix(mix(float(a > 0.0), float(b > 0.0), f.x),
                             mix(float(c > 0.0), float(d > 0.0), f.x), f.y);
        if (coverage < 0.5) discard;
        return mix(mix(decode1555(a), decode1555(b), f.x),
                   mix(decode1555(c), decode1555(d), f.x), f.y);
    }
    float packed = samplePSX(tc, format);
    if (packed == 0.0) discard;
    return decode1555(packed);
}

void main() {
    int format = int(ubo.hiresInfo.w + 0.5);
    vec4 color;
    vec2 framebufferCoord = vTexcoord.xy + vPageClut.xy;
    bool framebufferHit = false;
    for (int i = 0; i < 8; ++i) {
        vec4 r = ubo.framebufferRects[i];
        framebufferHit = framebufferHit || (r.z > 0.0 &&
                         all(greaterThanEqual(framebufferCoord, r.xy)) &&
                         all(lessThan(framebufferCoord, r.xy + r.zw)));
    }
    if (format == 4) {
        color = texture(sceneTexture, vTexcoord.xy / 255.0);
    } else if (format == 3) {
        vec2 nativeUv = vTexcoord.xy + ubo.textureInfo.zw;
        vec2 cell = floor(nativeUv);
        vec2 halfTexel = ubo.hiresInfo.xy;
        vec2 tc = (cell + clamp(nativeUv - cell, halfTexel, vec2(1.0) - halfTexel)) * ubo.textureInfo.xy;
        color = texture(sceneTexture, tc);
        if (color.a < 0.5) discard;
    } else if (format == 2 && framebufferHit) {
        color = texture(framebufferTexture,
                        framebufferCoord / vec2(1024.0, 512.0));
    } else {
        color = sampleNative(vTexcoord.xy, format);
    }

    vec3 albedo = color.rgb;
    color *= vColor;
    if (ubo.effectInfo.x > 0.5 && vViewPos.z > 0.0) {
        vec3 lightDir = normalize(ubo.lightDirStyle.xyz);
        bool classicStyle = ubo.effectInfo.y > 0.5;
        vec3 lightOrigin = classicStyle ? ubo.lightPosRange.xyz - lightDir * 39.0
                                        : ubo.lightPosRange.xyz;
        vec3 toLight = lightOrigin - vViewPos;
        float distanceToLight = length(toLight);
        vec3 L = toLight / max(distanceToLight, 0.0001);
        float cone = smoothstep(ubo.lightColor.w, ubo.renderInfo.w, dot(-L, lightDir));
        float ndl = 1.0;
        float attenuation;
        color.rgb *= classicStyle ? 0.49 : 0.15;
        if (classicStyle) {
            cone = cone * (2.0 - cone);
            float attenuationDistance = distanceToLight * 2.0;
            float inverseDistance = 1.0 / max(attenuationDistance, 1.0);
            attenuation = max(0.0, 134217728.0 * inverseDistance * inverseDistance - 16.0);
            attenuation += min(48.0, 32768.0 * inverseDistance);
            attenuation = clamp(attenuation / 255.0, 0.0, 1.0);
            attenuation *= 1.0 - smoothstep(ubo.lightPosRange.w * 0.9,
                                             ubo.lightPosRange.w,
                                             attenuationDistance);
        } else {
            vec3 faceNormal = cross(dFdx(vViewPos), dFdy(vViewPos));
            float normalLength = length(faceNormal);
            vec3 N = normalLength > 1e-9 ? faceNormal / normalLength : vec3(0.0, 0.0, -1.0);
            if (dot(N, vViewPos) > 0.0) N = -N;
            ndl = 0.15 + 0.85 * max(dot(N, L), 0.0);
            attenuation = clamp(1.0 - distanceToLight / max(ubo.lightPosRange.w, 1.0), 0.0, 1.0);
        }
        float shadow = 1.0;
        if (ubo.effectInfo.w > 0.5) {
            vec3 shadowViewPos = vShadowViewPos.xyz / max(vShadowViewPos.w, 1e-9);
            vec3 receiver = shadowViewPos + L * (ubo.shadowParams.z * distanceToLight);
            vec4 lightPosition = ubo.shadowMatrix * vec4(receiver, 1.0);
            if (lightPosition.w > 0.0) {
                vec3 lightUv;
                lightUv.xy = lightPosition.xy / lightPosition.w * 0.5 + 0.5;
                lightUv.z = lightPosition.z / lightPosition.w;
                if (lightUv.x > 0.0 && lightUv.x < 1.0 && lightUv.y > 0.0 && lightUv.y < 1.0 && lightUv.z < 1.0) {
                    vec3 uvDx = dFdx(lightUv), uvDy = dFdy(lightUv);
                    float det = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
                    float dzdu = 0.0, dzdv = 0.0;
                    if (abs(det) > 1e-9) {
                        dzdu = (uvDx.z * uvDy.y - uvDy.z * uvDx.y) / det;
                        dzdv = (uvDy.z * uvDx.x - uvDx.z * uvDy.x) / det;
                    }
                    float receiverLinear = ubo.shadowClip.w > 0.0 ? shadowLinearDepth(lightUv.z) : 0.0;
                    float occlusion = 0.0;
                    vec2 texelPosition = lightUv.xy / ubo.shadowClip.zz - vec2(0.5);
                    vec2 texelBase = floor(texelPosition), texelFraction = fract(texelPosition);
                    for (int sy = -1; sy <= 2; ++sy) {
                        float wy = sy == -1 ? 1.0 - texelFraction.y : (sy == 2 ? texelFraction.y : 1.0);
                        for (int sx = -1; sx <= 2; ++sx) {
                            float wx = sx == -1 ? 1.0 - texelFraction.x : (sx == 2 ? texelFraction.x : 1.0);
                            float weight = wx * wy;
                            vec2 sampleUv = (texelBase + vec2(float(sx), float(sy)) + vec2(0.5)) * ubo.shadowClip.zz;
                            float storedDepth = texture(shadowTexture, sampleUv).r;
                            float receiverDepth = lightUv.z + dzdu * (sampleUv.x - lightUv.x) + dzdv * (sampleUv.y - lightUv.y);
                            if (receiverDepth - ubo.shadowParams.y > storedDepth) {
                                if (ubo.shadowClip.w > 0.0) {
                                    float gap = receiverLinear - shadowLinearDepth(storedDepth);
                                    occlusion += weight * (1.0 - clamp(gap / ubo.shadowClip.w, 0.0, 1.0));
                                } else occlusion += weight;
                            }
                        }
                    }
                    shadow = 1.0 - ubo.shadowParams.w * (occlusion / 9.0);
                }
            }
        }
        color.rgb += albedo * ubo.lightColor.rgb * cone * attenuation * ndl * shadow;
    }
    float fog = clamp(vFog * ubo.fogColorStrength.a, 0.0, 1.0);
    if (ubo.effectInfo.z > 0.5) color.rgb *= 1.0 - fog;
    else color.rgb = mix(color.rgb, ubo.fogColorStrength.rgb, fog);

    if (ubo.hiresInfo.z > 0.5 && vIs3D > 0.5) {
        const mat4 dither = mat4(-4,0,-3,1, 2,-2,3,-1, -3,1,-4,0, 3,-1,2,-2) / 255.0;
        ivec2 dc = ivec2(fract(gl_FragCoord.xy / 8.0) * 4.0);
        if (ubo.effectInfo.z < 0.5) color.rgb += vec3(dither[dc.x][dc.y]);
        color.rgb = floor(color.rgb * 32.0 + 0.5) / 32.0;
    }
    outColor = color;
}
