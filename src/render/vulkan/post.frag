#version 450

layout(binding=0) uniform sampler2D sourceTexture;

layout(push_constant) uniform PostConstants {
    int mode;
    int tonemapMode;
    vec2 texelSize;
    float time;
    float postIntensity;
    float tonemapIntensity;
    float padding;
} pc;

layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 colorGrade(vec3 c) {
    c = (c - 0.5) * 1.12 + 0.5;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, 1.15);
    return c * vec3(1.06, 1.0, 0.94);
}

vec2 curve(vec2 uv) {
    uv = uv * 2.0 - 1.0;
    vec2 o = abs(uv.yx) / vec2(6.0, 5.0);
    uv += uv * o * o;
    return uv * 0.5 + 0.5;
}

vec3 applyTonemap(vec3 c) {
    if (pc.tonemapMode == 1) return c / (c + vec3(1.0));
    if (pc.tonemapMode == 2) {
        c *= 0.6;
        return clamp((c * (2.51 * c + 0.03)) /
                     (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    }
    if (pc.tonemapMode == 3) {
        vec3 x = max(vec3(0.0), c - 0.004);
        return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
    }
    return c;
}

void main() {
    vec2 uv = vUv;
    vec3 original = texture(sourceTexture, vUv).rgb;
    vec3 color = original;

    if (pc.mode == 1) {
        uv = curve(uv);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            outColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        color = texture(sourceTexture, uv).rgb;
        color *= 0.75 + 0.25 * abs(sin(uv.y * 240.0 * 3.14159));
        int m = int(mod(gl_FragCoord.x, 3.0));
        vec3 mask = m == 0 ? vec3(1.0, 0.72, 0.72) :
                    (m == 1 ? vec3(0.72, 1.0, 0.72) : vec3(0.72, 0.72, 1.0));
        color *= mask * 1.25;
        vec2 d = uv - 0.5;
        color *= clamp(1.0 - dot(d, d) * 1.1, 0.0, 1.0);
    } else if (pc.mode == 2) {
        color *= 0.7 + 0.3 * abs(sin(uv.y * 240.0 * 3.14159));
    } else if (pc.mode == 3) {
        vec2 d = uv - 0.5;
        color *= clamp(1.0 - dot(d, d) * 1.3, 0.0, 1.0);
    } else if (pc.mode == 4) {
        color = colorGrade(color);
    } else if (pc.mode == 5) {
        color += (hash(floor(uv / pc.texelSize) + pc.time) - 0.5) * 0.10;
    } else if (pc.mode == 6) {
        vec3 center = color;
        vec3 blur = (texture(sourceTexture, uv + vec2(pc.texelSize.x, 0.0)).rgb +
                     texture(sourceTexture, uv - vec2(pc.texelSize.x, 0.0)).rgb +
                     texture(sourceTexture, uv + vec2(0.0, pc.texelSize.y)).rgb +
                     texture(sourceTexture, uv - vec2(0.0, pc.texelSize.y)).rgb) * 0.25;
        color = center + (center - blur) * 0.85;
    } else if (pc.mode == 7) {
        vec2 grid = vec2(320.0, 240.0);
        vec2 quantizedUv = (floor(uv * grid) + 0.5) / grid;
        color = texture(sourceTexture, quantizedUv).rgb;
        const mat4 dither = mat4(-4,0,-3,1, 2,-2,3,-1,
                                  -3,1,-4,0, 3,-1,2,-2) / 255.0;
        ivec2 dc = ivec2(gl_FragCoord.xy) & 3;
        color += vec3(dither[dc.x][dc.y]);
        color = floor(color * 32.0 + 0.5) / 32.0;
    } else if (pc.mode == 8) {
        color = colorGrade(color);
        vec2 d = uv - 0.5;
        color *= clamp(1.0 - dot(d, d) * 0.9, 0.0, 1.0);
        color += (hash(floor(uv / pc.texelSize) + pc.time) - 0.5) * 0.045;
    }

    color = mix(original, color, pc.postIntensity);
    color = mix(color, applyTonemap(color), pc.tonemapIntensity);
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
