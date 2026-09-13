#include "common.slang"

// Engine shader convention is supplied by slangtypes:
//   set=0,binding=0 : Texture2D textures[];
//   set=0,binding=1 : SamplerState samplers[];
//   set=0,binding=2 : RWTexture2D<float4> storage_images[];
//   set=0,binding=3 : ConstantBuffer<GlobalData> global_data;
//
// The pond buffer is intentionally NOT a descriptor. The CPU allocates one
// persistent BufferSlice per frame-in-flight and passes its device address
// through a push constant. This keeps the render path descriptor-free.

#define TAU 6.28318530718
#define KOI_FISH_COUNT   6
#define KOI_RIPPLE_COUNT 24
#define KOI_FOOD_COUNT   8
#define KOI_PAD_COUNT    19
#define KOI_FLOWER_COUNT 5
#define KOI_DUCK_COUNT   5
#define KOI_PETAL_COUNT  6
#define KOI_YFLOW_COUNT  3

struct KoiPondGpu
{
    float4 fish[KOI_FISH_COUNT];
    float4 fish_meta[KOI_FISH_COUNT];
    float4 ripple[KOI_RIPPLE_COUNT];
    float4 food[KOI_FOOD_COUNT];
    float4 game_misc;
};

struct KoiPondPush { uint64_t pond_address;};

[[vk::push_constant]]
KoiPondPush push;
// -----------------------------------------------------------------------------
// Static scene data copied from the source.
// These never change at runtime, so do not waste a descriptor on them.
// -----------------------------------------------------------------------------
static const float4 PADS[KOI_PAD_COUNT] =
{
    float4(0.10,0.940,0.150,0.6), float4(0.30,0.965,0.115,2.6),
    float4(0.02,0.760,0.115,1.3), float4(0.24,0.800,0.085,4.4),
    float4(0.06,0.420,0.125,5.4), float4(0.20,0.280,0.095,0.9),
    float4(0.15,0.070,0.145,3.3), float4(0.45,0.045,0.105,1.0),
    float4(0.75,0.040,0.085,2.2), float4(2.62,0.945,0.160,5.9),
    float4(2.88,0.895,0.125,3.7), float4(2.96,0.700,0.100,0.4),
    float4(2.45,0.830,0.080,1.8), float4(2.94,0.450,0.115,2.9),
    float4(2.80,0.095,0.135,4.9), float4(2.50,0.045,0.090,0.2),
    float4(2.06,0.600,0.095,3.1), float4(2.24,0.360,0.070,5.5),
    float4(1.72,0.800,0.060,2.0)
};

static const float4 FLOWERS[KOI_FLOWER_COUNT] =
{
    float4(0.16,0.865,0.105,0.0),
    float4(0.085,0.520,0.095,1.0),
    float4(2.72,0.815,0.110,0.0),
    float4(2.85,0.210,0.125,0.0),
    float4(2.60,0.300,0.070,1.0)
};

static const float2 DUCK[KOI_DUCK_COUNT] =
{
    float2(0.42,0.70), float2(0.62,0.16), float2(2.30,0.90),
    float2(2.55,0.48), float2(0.95,0.90)
};

static const float2 PETALS[KOI_PETAL_COUNT] =
{
    float2(0.85,0.30), float2(1.35,0.56), float2(1.95,0.86),
    float2(2.42,0.56), float2(0.58,0.74), float2(1.60,0.14)
};

static const float2 YFLOW[KOI_YFLOW_COUNT] =
{
    float2(0.52,0.60), float2(2.20,0.74), float2(1.05,0.10)
};

// -----------------------------------------------------------------------------
// Utility math
// -----------------------------------------------------------------------------
float hash11(float p)
{
    p = frac(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return frac(p);
}

float hash21(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float2 hash22(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float vnoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));

    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float fbm(float2 p)
{
    float s = 0.0;
    float a = 0.5;

    for (int i = 0; i < 3; ++i)
    {
        s += a * vnoise(p);
        p = p * 2.03 + float2(17.3, 9.1);
        a *= 0.5;
    }

    return s;
}

float2x2 rot2(float a)
{
    float c = cos(a);
    float s = sin(a);
    return float2x2(c, s, -s, c);
}

float sdEllApprox(float2 p, float2 ab)
{
    float k = length(p / ab);
    return (k - 1.0) * min(ab.x, ab.y);
}

// High-quality signed ellipse distance. The source used an exact-ish
// algebraic solver rather than the cheap radial approximation above.
float sdEllipse(float2 p, float2 ab)
{
    p = abs(p);

    if (p.x > p.y)
    {
        p = p.yx;
        ab = ab.yx;
    }

    float l = ab.y * ab.y - ab.x * ab.x;
    float m = ab.x * p.x / l;
    float m2 = m * m;
    float n = ab.y * p.y / l;
    float n2 = n * n;
    float c = (m2 + n2 - 1.0) / 3.0;
    float c3 = c * c * c;
    float q = c3 + 2.0 * m2 * n2;
    float d = c3 + m2 * n2;
    float g = m + m * n2;
    float co;

    if (d < 0.0)
    {
        float h = acos(q / c3) / 3.0;
        float s = cos(h);
        float tt = sin(h) * sqrt(3.0);
        float rx = sqrt(max(0.0, -c * (s + tt + 2.0) + m2));
        float ry = sqrt(max(0.0, -c * (s - tt + 2.0) + m2));
        co = (ry + sign(l) * rx + abs(g) / max(rx * ry, 1e-6) - m) * 0.5;
    }
    else
    {
        float h = 2.0 * m * n * sqrt(max(d, 0.0));
        float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
        float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
        float rx = -s - u - 4.0 * c + 2.0 * m2;
        float ry = (s - u) * sqrt(3.0);
        float rm = sqrt(rx * rx + ry * ry);
        co = (ry / sqrt(max(rm - rx, 1e-6)) + 2.0 * g / max(rm, 1e-6) - m) * 0.5;
    }

    co = clamp(co, -1.0, 1.0);
    float2 r = ab * float2(co, sqrt(max(0.0, 1.0 - co * co)));
    return length(r - p) * sign(p.y - r.y);
}

// -----------------------------------------------------------------------------
// Caustics
// -----------------------------------------------------------------------------
float caustic(float2 uv, float t)
{
    float2 p = fmod(uv * TAU, TAU) - 250.0;
    float2 i = p;
    float c = 1.0;
    float inten = 0.005;

    for (int n = 0; n < 4; ++n)
    {
        float tt = t * (1.0 - (3.5 / (float(n) + 1.0)));
        i = p + float2(
            cos(tt - i.x) + sin(tt + i.y),
            sin(tt - i.y) + cos(tt + i.x));

        c += 1.0 / length(float2(
            p.x / (sin(i.x + tt) / inten),
            p.y / (cos(i.y + tt) / inten)));
    }

    c *= 0.25;
    c = 1.17 - pow(c, 1.4);
    return pow(abs(c), 7.0);
}

// -----------------------------------------------------------------------------
// Ambient micro-wave field
// Returns the gradient dH/dP rather than making the GPU estimate it by
// finite differences. That gives the surface normal almost for free.
// -----------------------------------------------------------------------------
float2 ambientGrad(float2 p, float t)
{
    float2 g = float2(0.0, 0.0);

    float w1 = fbm(p * 0.9 + float2(t * 0.050, -t * 0.040));
    float w2 = fbm(p * 0.9 + float2(-t * 0.045, t * 0.055) + 7.3);
    float2 pw = p + 0.55 * float2(w1 - 0.5, w2 - 0.5);

    for (int i = 0; i < 7; ++i)
    {
        float fi = float(i);
        float an = fi * 2.39996 + 1.7;
        float2 d = float2(cos(an), sin(an));
        float fr = 7.0 + fi * 4.3;
        float sp = 0.7 + 0.31 * fi;
        float amp = 0.0052 / (1.0 + fi * 0.4);

        g += d * cos(dot(pw, d) * fr + t * sp) * (amp * fr);
    }

    return g;
}

// -----------------------------------------------------------------------------
// Interactive ripples
// ripple = (centerX, centerY, startTime, amplitude)
// -----------------------------------------------------------------------------
float3 rippleField(float2 P, float t)
{
    float2 g = float2(0.0, 0.0);
    float h = 0.0;

   KoiPondGpu* koi_pond =  (KoiPondGpu*)push.pond_address;
    for (int i = 0; i < KOI_RIPPLE_COUNT; ++i)
    {
        float4 R = koi_pond.ripple[i];
        float age = t - R.z;

        if (R.w > 0.0 && age > 0.0 && age < 6.0)
        {
            float2 d = P - R.xy;
            float r = length(d);
            float x = r - age * 0.5;
            float sig2 = 0.002025;

            float env =
                exp(-x * x / (2.0 * sig2)) *
                exp(-age * 0.9) *
                smoothstep(0.0, 0.06, age) /
                (1.0 + r * 2.0);

            float ph = x * 105.0;
            float cosp = cos(ph);
            float sinp = sin(ph);

            // dh is dH/dr. Since the radial basis vector is d/r,
            // grad(H) = (d/r) * dH/dr.
            float dh = R.w *
                (env * (-x / sig2) * cosp - env * 105.0 * sinp);

            float2 dir = d / max(r, 1e-4);
            g += dir * dh;
            h += R.w * env * cosp;
        }
    }

    return float3(g, h);
}

// -----------------------------------------------------------------------------
// Koi SDF
// -----------------------------------------------------------------------------
float fishD(float2 p, float4 F, float size, out float2 localP, out float finD)
{
    float2 q = mul(rot2(-F.z), (p - F.xy)) / size;

    // Gentle body undulation. F.w is the time-varying wag phase supplied by CPU.
    q.y += sin(F.w + q.x * 2.6) * 0.09 *
           (0.55 - 0.45 * clamp(q.x, -1.0, 1.0));

    localP = q;

    float d = sdEllipse(q, float2(0.95, 0.30));

    float2 tA = mul(rot2(-0.42), (q - float2(-1.00, 0.14)));
    float2 tB = mul(rot2( 0.42), (q - float2(-1.00,-0.14)));
    float tail = min(
        sdEllApprox(tA, float2(0.34, 0.085)),
        sdEllApprox(tB, float2(0.34, 0.085)));

    float2 fA = mul(rot2(-0.80), (q - float2(0.08, 0.30)));
    float2 fB = mul(rot2( 0.80), (q - float2(0.08,-0.30)));
    float fins = min(
        sdEllApprox(fA, float2(0.26, 0.075)),
        sdEllApprox(fB, float2(0.26, 0.075)));

    finD = min(tail, fins);
    return min(d, finD);
}

float4 fishColor(float2 lp, float d, float finD, int ftype, float seed)
{
    float2 pc = float2(lp.x * 1.7, lp.y * 2.6);
    float n = fbm(pc + seed * 19.7);

    float3 base;
    float3 patch;
    float th0;
    float th1;

    if (ftype == 0)
    {
        base = float3(0.97,0.95,0.90);
        patch = float3(0.93,0.32,0.13);
        th0 = 0.55; th1 = 0.63;
    }
    else if (ftype == 1)
    {
        base = float3(0.97,0.96,0.92);
        patch = float3(1.00,0.55,0.14);
        th0 = 0.50; th1 = 0.58;
    }
    else if (ftype == 2)
    {
        base = float3(0.93,0.94,0.96);
        patch = float3(0.10,0.11,0.15);
        th0 = 0.52; th1 = 0.60;
    }
    else if (ftype == 3)
    {
        base = float3(0.99,0.76,0.22);
        patch = float3(0.99,0.90,0.50);
        th0 = 0.56; th1 = 0.63;
    }
    else if (ftype == 4)
    {
        base = float3(0.97,0.95,0.91);
        patch = float3(0.83,0.17,0.11);
        th0 = 0.50; th1 = 0.58;
    }
    else
    {
        base = float3(0.89,0.90,0.93);
        patch = float3(0.13,0.14,0.18);
        th0 = 0.55; th1 = 0.62;
    }

    float m = smoothstep(th0, th1, n);
    float3 col = lerp(base, patch, m);

    if (ftype == 3)
        col = lerp(col, float3(0.95,0.55,0.10),
                   smoothstep(0.25,0.0,abs(lp.y)) * 0.35);

    if (ftype == 5)
    {
        float m2 = smoothstep(0.45,0.50,n) * (1.0 - m);
        col = lerp(col, float3(0.45,0.48,0.53), m2 * 0.8);
    }

    col *= lerp(0.78,1.03,smoothstep(0.34,0.0,abs(lp.y)));
    col = lerp(col, 1.0.xxx, smoothstep(0.0,-0.12,d));

    float finFull =
        smoothstep(0.05,-0.03,finD) *
        smoothstep(-0.01,0.03,d);

    col = lerp(
        col,
        lerp(base,1.0.xxx,0.45),
        saturate(finFull * 1.4) * 0.7);

    // Eye placement in normalized local fish space.
    float ed = min(
        length(lp - float2(0.60,0.155)),
        length(lp - float2(0.60,-0.155)));

    col = lerp(col, float3(0.06,0.05,0.06),
               smoothstep(0.055,0.035,ed));

    float a = smoothstep(0.045,-0.025,d);
    a = max(a, finFull * 0.55);

    return float4(col, a);
}

float4 fishLayer(float2 p)
{
    float3 col = float3(0.0,0.0,0.0);
    float alpha = 0.0;

   KoiPondGpu* koi_pond =  (KoiPondGpu*)push.pond_address;
    for (int i = 0; i < KOI_FISH_COUNT; ++i)
    {
        float4 F = koi_pond.fish[i];
        float4 M = koi_pond.fish_meta[i];
        float sz = M.x;

        float2 dq = p - F.xy;

        // Broad-phase reject before evaluating the SDF.
        if (dot(dq, dq) <= 4.2 * sz * sz)
        {
            float2 lp;
            float fd;
            float d = fishD(p, F, sz, lp, fd);

            if (d <= 0.08)
            {
                float4 fc = fishColor(lp, d, fd, int(M.y + 0.5), M.z);
                col = lerp(col, fc.rgb, fc.a);
                alpha = lerp(alpha, 1.0, fc.a);
            }
        }
    }

    return float4(col, alpha);
}

float fishShadow(float2 p)
{
    float sh = 0.0;

   KoiPondGpu* koi_pond =  (KoiPondGpu*)push.pond_address;
    for (int i = 0; i < KOI_FISH_COUNT; ++i)
    {
        float4 F = koi_pond.fish[i];
        float4 M = koi_pond.fish_meta[i];
        float sz = M.x * 1.12;
        float2 c = F.xy + float2(0.05,0.065);
        float2 dq = p - c;

        if (dot(dq,dq) <= 4.5 * sz * sz)
        {
            float2 lp;
            float fd;
            float d = fishD(p, float4(c,F.z,F.w), sz, lp, fd);
            sh += smoothstep(0.45,-0.15,d) * 0.26;
        }
    }

    return min(sh,0.5);
}

// -----------------------------------------------------------------------------
// Lily pads
// -----------------------------------------------------------------------------
float padShadow(float2 p)
{
    float sh = 0.0;

    for (int i = 0; i < KOI_PAD_COUNT; ++i)
    {
        float2 c = PADS[i].xy + float2(0.055,0.07);
        float r = PADS[i].z;
        sh += smoothstep(r + 0.02, r * 0.35, length(p - c));
    }

    return min(sh,1.0);
}

float4 padLayer(float2 P, float t, out float ao)
{
    float3 col = float3(0.0,0.0,0.0);
    float alpha = 0.0;
    ao = 0.0;

    for (int i = 0; i < KOI_PAD_COUNT; ++i)
    {
        float4 pd = PADS[i];
        float2 c = pd.xy +
            float2(
                sin(t * 0.45 + float(i) * 1.93),
                cos(t * 0.38 + float(i) * 2.71)) * 0.004;

        float2 q = P - c;
        float r = pd.z;
        float dC = length(q) - r;

        float ring = smoothstep(0.055,0.005,dC) *
                     smoothstep(-0.02,0.005,dC);
        ao = max(ao, ring);

        if (dC <= 0.06)
        {
            float rot = pd.w + sin(t * 0.20 + float(i)) * 0.02;
            float2 qr = mul(rot2(-rot), q);

            float nh = 0.42;
            float2 nA = float2(sin(nh),-cos(nh));
            float2 nB = float2(sin(nh), cos(nh));

            float d = max(dC, min(dot(qr,nA), dot(qr,nB)));
            float a = smoothstep(0.006,-0.006,d);

            if (a > 0.0)
            {
                float ang = atan2(qr.y,qr.x);
                float rad = length(qr) / r;
                float hv = hash11(float(i) * 7.13);

                float3 g1 = lerp(
                    float3(0.30,0.63,0.22),
                    float3(0.47,0.74,0.21),
                    hv);
                float3 g2 = lerp(
                    float3(0.16,0.45,0.18),
                    float3(0.30,0.58,0.16),
                    hv);

                float3 pc = lerp(g1,g2,smoothstep(0.10,1.0,rad));

                float veins = smoothstep(0.86,0.99,abs(sin(ang * 9.0))) *
                              smoothstep(0.12,0.45,rad);
                pc += veins * float3(0.20,0.30,0.10);

                pc *= 0.94 + 0.06 * sin(ang * 18.0 + hv * 7.0);
                pc *= lerp(1.0,0.72,smoothstep(0.85,1.0,rad));
                pc += float3(0.10,0.12,0.04) * smoothstep(0.35,0.0,rad);
                pc += float3(0.22,0.26,0.16) *
                      pow(max(0.0,1.0-rad),3.0) * 0.35;

                col = lerp(col,pc,a);
                alpha = lerp(alpha,1.0,a);
            }
        }
    }

    return float4(col,alpha);
}

// -----------------------------------------------------------------------------
// Lotus flowers
// -----------------------------------------------------------------------------
float4 flowerLayer(float2 P, float t)
{
    float3 col = float3(0.0,0.0,0.0);
    float alpha = 0.0;

    for (int i = 0; i < KOI_FLOWER_COUNT; ++i)
    {
        float4 f = FLOWERS[i];
        float2 q0 = P - f.xy;
        float sc = f.z;
        float rr = dot(q0,q0);

        if (rr <= sc * sc * 2.6)
        {
            float halo = smoothstep(sc * 1.45, sc * 0.4, sqrt(rr));
            col = lerp(col, float3(0.02,0.20,0.19),
                        halo * 0.22 * (1.0 - alpha));
            alpha = max(alpha, halo * 0.20);

            float2 q = q0 / sc;
            float baseRot =
                hash11(float(i) * 5.77) * TAU +
                sin(t * 0.15 + float(i)) * 0.03;

            float3 tipC;
            float3 baseC;

            if (f.w < 0.5)
            {
                tipC = float3(1.00,0.45,0.66);
                baseC = float3(1.00,0.88,0.93);
            }
            else
            {
                tipC = float3(0.99,0.80,0.88);
                baseC = 1.0.xxx;
            }

            float d = 1e5;
            float2 lp = float2(0.0,0.0);

            for (int k = 0; k < 8; ++k)
            {
                float a = baseRot + float(k) * 0.7853982;
                float2 rq = mul(rot2(-a),q);
                float dd = sdEllApprox(
                    rq - float2(0.50,0.0),
                    float2(0.52,0.30));

                if (dd < d)
                {
                    d = dd;
                    lp = rq;
                }
            }

            float u = clamp(lp.x,0.0,1.0);
            float3 pcol = lerp(
                baseC,
                tipC,
                smoothstep(0.15,0.95,u));
            pcol *= 0.92 + 0.08 * smoothstep(0.0,0.25,abs(lp.y));
            pcol *= lerp(0.85,1.0,smoothstep(0.0,-0.10,d));

            float pa = smoothstep(0.02,-0.02,d);
            col = lerp(col,pcol,pa);
            alpha = lerp(alpha,1.0,pa);

            // Inner petals
            float d2 = 1e5;
            float2 lp2 = float2(0.0,0.0);

            for (int k = 0; k < 5; ++k)
            {
                float a = baseRot + 0.4 + float(k) * 1.2566371;
                float2 rq = mul(rot2(-a),q);
                float dd = sdEllApprox(
                    rq - float2(0.33,0.0),
                    float2(0.35,0.27));

                if (dd < d2)
                {
                    d2 = dd;
                    lp2 = rq;
                }
            }

            float u2 = clamp(lp2.x / 0.68,0.0,1.0);
            float3 pcol2 = lerp(
                baseC,
                tipC,
                smoothstep(0.2,1.0,u2) * 0.7);
            pcol2 *= 0.95 + 0.05 * smoothstep(0.0,0.2,abs(lp2.y));

            float pa2 = smoothstep(0.02,-0.02,d2);
            col = lerp(col,pcol2 * 1.03,pa2);
            alpha = lerp(alpha,1.0,pa2);

            // Small golden flower center.
            float dc = length(q) - 0.20;
            float caa = smoothstep(0.02,-0.02,dc);
            float3 ccol = float3(1.0,0.85,0.30);
            float angc = atan2(q.y,q.x);

            float dots =
                smoothstep(-0.9,0.9,sin(angc * 10.0 + 0.5)) *
                smoothstep(0.10,0.20,length(q)) *
                smoothstep(0.03,0.08,length(q));

            ccol = lerp(ccol,float3(1.0,0.55,0.10),dots);
            col = lerp(col,ccol,caa);
            alpha = lerp(alpha,1.0,caa);
        }
    }

    return float4(col,alpha);
}

// -----------------------------------------------------------------------------
// Food pellets supplied by the CPU simulation.
// xy = pond-space position, z = remaining lifetime, w = active flag.
// -----------------------------------------------------------------------------
float4 foodLayer(float2 P)
{
    float3 color = 0.0.xxx;
    float alpha = 0.0;

   KoiPondGpu* koi_pond =  (KoiPondGpu*)push.pond_address;
    for (int i = 0; i < KOI_FOOD_COUNT; ++i)
    {
        float4 food = koi_pond.food[i];

        if (food.w <= 0.0)
            continue;

        float radius = 0.010;
        float distance = length(P - food.xy) - radius;
        float a = smoothstep(0.004, -0.002, distance);

        float pulse =
            0.82 +
            0.18 * sin(global_data.time * 7.0 + float(i) * 1.71);

        float3 pellet_color =
            float3(1.0, 0.62, 0.12) * pulse;

        color = lerp(color, pellet_color, a);
        alpha = lerp(alpha, 1.0, a);
    }

    return float4(color, alpha);
}

// -----------------------------------------------------------------------------
// Floating duckweed / petals / tiny yellow flowers
// -----------------------------------------------------------------------------
float4 floatLayer(float2 P, float t)
{
    float3 col = float3(0.0,0.0,0.0);
    float alpha = 0.0;

    for (int c = 0; c < KOI_DUCK_COUNT; ++c)
    {
        float2 bc = DUCK[c] +
            0.012 * float2(
                sin(t * 0.10 + float(c) * 2.0),
                cos(t * 0.083 + float(c) * 1.3));

        if (dot(P - bc, P - bc) <= 0.09)
        {
            for (int k = 0; k < 7; ++k)
            {
                float2 h = hash22(float2(float(c) * 7.7, float(k) * 3.3));
                float2 pos = bc + (h - 0.5) * 0.15;
                float r = 0.006 + 0.007 * hash11(float(k) * 1.7 + float(c));
                float d = length(P - pos) - r;
                float a = smoothstep(0.004,-0.004,d);
                float3 dc = lerp(
                    float3(0.35,0.66,0.25),
                    float3(0.62,0.84,0.37),
                    h.x);

                dc *= 0.9 + 0.2 * smoothstep(-r,0.0,d);
                col = lerp(col,dc,a);
                alpha = lerp(alpha,1.0,a);
            }
        }
    }

    for (int k = 0; k < KOI_PETAL_COUNT; ++k)
    {
        float2 pos = PETALS[k] + 0.035 * float2(
            sin(t * 0.07 + float(k) * 2.39),
            cos(t * 0.06 + float(k) * 1.61));

        float2 rq = mul(
            rot2(t * 0.05 + float(k) * 2.0),
            (P - pos) / 0.020);

        float d = sdEllApprox(rq,float2(1.0,0.55));
        float aa = smoothstep(0.15,-0.15,d);

        float3 pc = lerp(
            float3(1.0,0.75,0.85),
            float3(1.0,0.55,0.70),
            smoothstep(-0.5,0.8,rq.x));

        col = lerp(col,pc,aa);
        alpha = lerp(alpha,1.0,aa);
    }

    for (int k = 0; k < KOI_YFLOW_COUNT; ++k)
    {
        float2 pos = YFLOW[k] + 0.02 * float2(
            sin(t * 0.09 + float(k)),
            cos(t * 0.07 + float(k) * 2.2));

        if (dot(P - pos,P - pos) <= 0.004)
        {
            float2 q = (P - pos) / 0.016;
            float d = 1e5;

            for (int j = 0; j < 4; ++j)
            {
                float2 rq = mul(
                    rot2(float(j) * 1.5708 + 0.4),q);

                d = min(
                    d,
                    sdEllApprox(
                        rq - float2(0.55,0.0),
                        float2(0.5,0.30)));
            }

            float aa = smoothstep(0.12,-0.12,d);
            col = lerp(col,float3(0.95,0.85,0.35),aa);
            alpha = lerp(alpha,1.0,aa);

            float dcc = length(q) - 0.28;
            float ac = smoothstep(0.15,-0.15,dcc);
            col = lerp(col,float3(0.85,0.55,0.15),ac);
            alpha = lerp(alpha,1.0,ac);
        }
    }

    return float4(col,alpha);
}

// -----------------------------------------------------------------------------
// Fullscreen triangle
// -----------------------------------------------------------------------------
struct VSOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vs_main(uint id : SV_VertexID)
{
    // One triangle covers the entire viewport. No vertex buffer required.
    static const float2 V[3] =
    {
        float2(-1.0,-1.0),
        float2( 3.0,-1.0),
        float2(-1.0, 3.0)
    };

    VSOutput o;
    o.position = float4(V[id],0.0,1.0);
    return o;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target0
{
    float2 resolution = global_data.screen_params.xy;
    float2 invResolution = global_data.screen_params.zw;
    float2 uv = input.position.xy * invResolution;

    // Vulkan's fragment coordinates are upper-left based; the original
    // WebGL shader was built around gl_FragCoord with a y-up pond space.
   // uv.y = 1.0 - uv.y;

    // Preserve the source's deliberately 3:1 pond-space coordinates.
    float2 P = float2(uv.x * 3.0, uv.y);
    float t = global_data.time;

    // -------------------------------------------------------------------------
    // Water displacement -> analytic surface normal
    // -------------------------------------------------------------------------
    float3 rip = rippleField(P,t);
    float2 slope = rip.xy + ambientGrad(P,t);
    float sl = length(slope);

    if (sl > 0.6)
        slope *= 0.6 / sl;

    float3 N = normalize(float3(-slope.x,-slope.y,1.0));

    // -------------------------------------------------------------------------
    // Depth + fake refraction
    // -------------------------------------------------------------------------
    float dCenter = length((uv - float2(0.5,0.52)) * float2(1.5,2.4));
    float depth = smoothstep(0.78,0.12,dCenter);
    float2 refr = slope * 0.055 * (0.75 + 0.5 * depth);
    float2 Pr = P - refr;

    // -------------------------------------------------------------------------
    // Pond floor
    // -------------------------------------------------------------------------
    float3 floorCol = lerp(
        float3(0.24,0.79,0.75),
        float3(0.05,0.56,0.59),
        smoothstep(0.05,0.6,depth));

    floorCol = lerp(
        floorCol,
        float3(0.015,0.36,0.41),
        smoothstep(0.55,1.0,depth));

    floorCol *= 0.90 + 0.18 * fbm(
        P * 1.35 + float2(t * 0.020,-t * 0.016));

    float edgeD = min(
        min(P.x,3.0-P.x),
        min(P.y,1.0-P.y));

    float blob =
        smoothstep(0.32,0.03,edgeD) *
        smoothstep(
            0.35,0.75,
            fbm(P * 2.7 + float2(0.0,t * 0.02) + 4.7));

    floorCol = lerp(
        floorCol,
        float3(0.03,0.23,0.21),
        blob * 0.55);

    // -------------------------------------------------------------------------
    // Caustics. Keep the actual pattern this time; do not overwrite it with a
    // scalar brightness term as the pasted source accidentally did.
    // -------------------------------------------------------------------------
    float2 cuv = Pr * 0.85 + slope * 0.30;
    float ca =
        caustic(cuv,t * 0.55) * 0.75 +
        caustic(Pr * 1.75 - slope * 0.18 + 3.7,t * 0.80) * 0.45;

    float3 caCol = float3(0.72,1.0,0.97);
    floorCol += ca * caCol * (0.42 - 0.18 * depth);

    // Shadows cast onto the underwater floor.
    floorCol *= 1.0 - 0.30 * padShadow(Pr);
    floorCol *= 1.0 - fishShadow(Pr);

    // -------------------------------------------------------------------------
    // Underwater koi
    // -------------------------------------------------------------------------
    float4 F = fishLayer(Pr);
    float3 tintedFish = lerp(F.rgb,float3(0.10,0.55,0.55),0.16);
    float3 col = lerp(floorCol,tintedFish,F.a);
    col += ca * caCol * F.a * 0.22;

    // -------------------------------------------------------------------------
    // Water surface / Fresnel-like edge response
    // -------------------------------------------------------------------------
    float fres = pow(saturate(1.0-N.z),1.4);
    col = lerp(
        col,
        float3(0.70,0.92,0.94),
        0.045 + fres * 0.5);

    float3 L = normalize(float3(0.42,0.58,0.70));
    float3 Hv = normalize(L + float3(0.0,0.0,1.0));
    float ndh = max(dot(N,Hv),0.0);

    col +=
        (pow(ndh,80.0) * 0.30 +
         pow(ndh,480.0) * 1.5) *
        float3(1.0,0.99,0.94);

    float shim = pow(
        fbm(P * 6.5 + float2(t * 0.30,-t * 0.23) + slope * 1.6),
        6.0);

    col += shim * float3(0.85,1.0,1.0) * 0.85;

    // Ripple crest light / trough darkening.
    float hVis = rip.z * 120.0;
    col += saturate(hVis) * float3(0.60,0.90,0.93) * 0.32;
    col -= saturate(-hVis) * float3(0.0,0.06,0.06);

    // -------------------------------------------------------------------------
    // Food + floating details, lily pads and flowers
    // -------------------------------------------------------------------------
    float4 FD = foodLayer(P);
    col = lerp(col, FD.rgb, FD.a);

    float4 FL = floatLayer(P,t);
    col = lerp(col,FL.rgb,FL.a);

    float ao;
    float4 PL = padLayer(P,t,ao);
    col *= 1.0 - 0.20 * ao;
    col = lerp(col,PL.rgb,PL.a);

    float4 FW = flowerLayer(P,t);
    col = lerp(col,FW.rgb,FW.a);

    // -------------------------------------------------------------------------
    // Final grade
    // -------------------------------------------------------------------------
    float vig = smoothstep(
        0.98,0.40,
        length((uv - 0.5) * float2(1.05,2.0)));

    col *= lerp(0.72,1.0,vig);
    col *= lerp(0.60,1.0,smoothstep(0.0,0.09,edgeD));

    col = clamp(col,0.0,1.6);

    // The engine owns HDR -> LDR conversion in postprocess.slang.
    // Keep this pass linear/HDR so exposure and grading remain centralized.
    return float4(max(col, 0.0), 1.0);
}
