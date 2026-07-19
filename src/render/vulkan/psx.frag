#version 460
#ifdef RTGI
#extension GL_EXT_ray_query : require
#endif

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
#ifdef RTGI
layout(set=1,binding=0) uniform accelerationStructureEXT sceneAccelerationStructure;
struct RayMaterial {
    vec4 uv01;
    vec4 uv2Format;
    vec4 pageClut;
    vec4 color0;
    vec4 color1;
    vec4 color2;
};
layout(set=1,binding=1,std430) readonly buffer RayMaterials {
    RayMaterial materials[];
} rayScene;
#endif

layout(location=0) in vec4 vTexcoord;
layout(location=1) in vec4 vColor;
layout(location=2) in vec4 vPageClut;
layout(location=3) in float vFog;
layout(location=4) in float vIs3D;
layout(location=5) in vec3 vViewPos;
layout(location=6) in vec4 vShadowViewPos;
layout(location=0) out vec4 outColor;

const vec2 VRAM_TEXEL = vec2(1.0 / 1024.0, 1.0 / 512.0);

vec3 surfaceNormal(vec3 position) {
    vec3 faceNormal = cross(dFdx(position), dFdy(position));
    float normalLength = length(faceNormal);
    vec3 normal = normalLength > 1e-9 ? faceNormal / normalLength : vec3(0.0, 0.0, -1.0);
    return dot(normal, position) > 0.0 ? -normal : normal;
}

#ifdef RTGI
float rayVisibility(vec3 origin, vec3 direction, float maximumDistance) {
    if (maximumDistance <= 8.0) return 1.0;
    rayQueryEXT query;
    rayQueryInitializeEXT(query, sceneAccelerationStructure,
                          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                          0xff, origin, 4.0, direction, maximumDistance);
    while (rayQueryProceedEXT(query)) {}
    return rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

float screenNoise(vec2 position) {
    return fract(sin(dot(position, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 cosineHemisphere(vec3 normal, vec2 seed) {
    float angle = 6.28318530718 * screenNoise(seed);
    float radius = sqrt(screenNoise(seed.yx + vec2(31.17, 9.73)));
    vec3 tangent = normalize(abs(normal.z) < 0.999 ? cross(normal, vec3(0.0, 0.0, 1.0))
                                                   : cross(normal, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * (cos(angle) * radius) +
                     bitangent * (sin(angle) * radius) +
                     normal * sqrt(max(0.0, 1.0 - radius * radius)));
}
#endif

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

float samplePSXAt(vec2 tc, int format, vec4 pageClut) {
    if (format == 0) {
        vec2 comp = readVRAM((tc * vec2(0.25, 1.0) + pageClut.xy) * VRAM_TEXEL);
        int nibble = int(fract(tc.x / 4.0 + 0.0001) * 4.0);
        float value = comp[nibble / 2] * (255.001 / 16.0);
        float lo = fract(value) * 16.0;
        float hi = floor(value);
        float index = ((nibble & 1) == 0) ? lo : hi;
        vec2 clutUv = pageClut.zw + vec2(index * VRAM_TEXEL.x, 0.0);
        return packRG(readVRAM(clutUv));
    }
    if (format == 1) {
        vec2 comp = readVRAM((tc * vec2(0.5, 1.0) + pageClut.xy) * VRAM_TEXEL);
        float index = comp[int(mod(tc.x, 2.0))] * 255.001;
        return packRG(readVRAM(pageClut.zw + vec2(index * VRAM_TEXEL.x, 0.0)));
    }
    return packRG(readVRAM((tc + pageClut.xy) * VRAM_TEXEL));
}

float samplePSX(vec2 tc, int format) {
    return samplePSXAt(tc, format, vPageClut);
}

vec4 sampleNative(vec2 tc, int format) {
    if ((ubo.renderInfo.z > 0.5 && vIs3D > 0.5) || ubo.renderInfo.z > 1.5) {
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

#ifdef RTGI
bool traceReflection(vec3 origin, vec3 direction, float maximumDistance, out vec3 radiance) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, sceneAccelerationStructure, gl_RayFlagsOpaqueEXT,
                          0xff, origin, 6.0, direction, maximumDistance);
    while (rayQueryProceedEXT(query)) {}
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT)
        return false;

    uint primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
    vec2 bary12 = rayQueryGetIntersectionBarycentricsEXT(query, true);
    vec3 bary = vec3(1.0 - bary12.x - bary12.y, bary12.x, bary12.y);
    RayMaterial material = rayScene.materials[primitive];
    vec2 uv = material.uv01.xy * bary.x + material.uv01.zw * bary.y +
              material.uv2Format.xy * bary.z;
    vec3 vertexColor = material.color0.rgb * bary.x + material.color1.rgb * bary.y +
                       material.color2.rgb * bary.z;
    int format = int(material.uv2Format.z + 0.5);
    vec4 texel = format <= 2 ? decode1555(samplePSXAt(uv, format, material.pageClut))
                             : vec4(1.0);
    radiance = texel.rgb * vertexColor;
    return true;
}

float inferredMetalness(vec3 albedo) {
    float high = max(albedo.r, max(albedo.g, albedo.b));
    float low = min(albedo.r, min(albedo.g, albedo.b));
    float saturation = (high - low) / max(high, 0.08);
    float luminance = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    /* Silent Hill has no material channel. Neutral, mid-value painted surfaces
     * are the safest proxy for steel; dark cloth and bright plaster stay rough. */
    return (1.0 - smoothstep(0.12, 0.38, saturation)) *
           smoothstep(0.16, 0.42, luminance) * (1.0 - smoothstep(0.82, 0.98, luminance));
}
#endif

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
#ifdef RTGI
    if (vIs3D > 0.5 && vViewPos.z > 0.0) {
        vec3 normal = surfaceNormal(vViewPos);
        vec3 rayOrigin = vViewPos + normal * 6.0;
        vec3 bounceDirection = cosineHemisphere(normal, gl_FragCoord.xy);
        float bounceVisibility = rayVisibility(rayOrigin, bounceDirection, 520.0);
        /* One hardware visibility ray supplies contact AO and an environment
         * bounce. The deliberately restrained energy preserves the PSX grade. */
        color.rgb *= mix(0.68, 1.0, bounceVisibility);
        vec3 ambientBounce = max(ubo.fogColorStrength.rgb, vec3(0.035));
        color.rgb += albedo * ambientBounce * (0.12 * bounceVisibility);

        float metalness = inferredMetalness(albedo);
        if (metalness > 0.03) {
            vec3 viewDirection = normalize(-vViewPos);
            vec3 reflectionDirection = normalize(reflect(-viewDirection, normal));
            vec3 reflectedColor;
            bool reflectionHit = traceReflection(rayOrigin, reflectionDirection, 2400.0, reflectedColor);
            if (!reflectionHit)
                reflectedColor = max(ubo.fogColorStrength.rgb, vec3(0.025));
            float fresnel = 0.08 + 0.92 * pow(1.0 - max(dot(normal, viewDirection), 0.0), 5.0);
            float reflectionWeight = metalness * mix(0.18, 0.58, fresnel);
            color.rgb = mix(color.rgb, reflectedColor * 1.35, reflectionWeight);
        }
    }
#endif
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
            vec3 N = surfaceNormal(vViewPos);
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
#ifdef RTGI
        vec3 rtNormal = surfaceNormal(vViewPos);
        shadow *= rayVisibility(vViewPos + rtNormal * 6.0, L, max(distanceToLight - 10.0, 0.0));
#endif
        color.rgb += albedo * ubo.lightColor.rgb * cone * attenuation * ndl * shadow;
#ifdef RTGI
        /* A visible flashlight response on painted metal. The visibility term
         * is the same hardware ray used by the diffuse beam, so highlights are
         * correctly removed when an object blocks the light. */
        float metalness = inferredMetalness(albedo);
        if (metalness > 0.03) {
            vec3 N = surfaceNormal(vViewPos);
            vec3 V = normalize(-vViewPos);
            vec3 H = normalize(L + V);
            float specularPower = mix(28.0, 112.0, metalness);
            float specular = pow(max(dot(N, H), 0.0), specularPower);
            float fresnel = 0.12 + 0.88 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
            color.rgb += ubo.lightColor.rgb * cone * attenuation * shadow *
                         specular * metalness * mix(0.55, 1.5, fresnel);
        }
#endif
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
