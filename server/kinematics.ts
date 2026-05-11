/**
 * Stateless Kinematic Operations
 * 
 * Logic Constraint: Direct, stateless functions only.
 * Performance Constraint: Raw input processing, minimal allocation.
 */

export type Vec3 = Float32Array; // [x, y, z]
export type Quat = Float32Array; // [w, x, y, z]

export function vec3(x = 0, y = 0, z = 0): Vec3 {
    return new Float32Array([x, y, z]);
}

export function quat(w = 1, x = 0, y = 0, z = 0): Quat {
    return new Float32Array([w, x, y, z]);
}

export function sub(a: Vec3, b: Vec3, out = vec3()): Vec3 {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
}

export function add(a: Vec3, b: Vec3, out = vec3()): Vec3 {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
}

export function mag(a: Vec3): number {
    return Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2);
}

export function normalize(a: Vec3, out = vec3()): Vec3 {
    const m = mag(a);
    if (m === 0) {
        out.fill(0);
        return out;
    }
    const invM = 1 / m;
    out[0] = a[0] * invM;
    out[1] = a[1] * invM;
    out[2] = a[2] * invM;
    return out;
}

export function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function angleBetween(a: Vec3, b: Vec3): number {
    const d = dot(normalize(a), normalize(b));
    return Math.acos(Math.max(-1, Math.min(1, d))) * (180 / Math.PI);
}

export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

export function cross(a: Vec3, b: Vec3, out = vec3()): Vec3 {
    const ax = a[0], ay = a[1], az = a[2];
    const bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
}

/**
 * Calculates rotation from vector u to vector v.
 * Used for translating landmark pairs into bone rotations.
 */
export function quatFromVectors(u: Vec3, v: Vec3, out = quat()): Quat {
    const uN = normalize(u);
    const vN = normalize(v);
    const d = dot(uN, vN);

    if (d >= 0.999999) {
        out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
        return out;
    }
    
    if (d <= -0.999999) {
        // 180 degree rotation
        let ortho = cross(vec3(1, 0, 0), uN);
        if (mag(ortho) < 0.0001) ortho = cross(vec3(0, 1, 0), uN);
        normalize(ortho, ortho);
        out[0] = 0; out[1] = ortho[0]; out[2] = ortho[1]; out[3] = ortho[2];
        return out;
    }

    const c = cross(uN, vN);
    const s = Math.sqrt((1 + d) * 2);
    const invS = 1 / s;
    out[0] = s * 0.5;
    out[1] = c[0] * invS;
    out[2] = c[1] * invS;
    out[3] = c[2] * invS;
    return out;
}

/**
 * Bone Length Normalization
 *strips absolute distance, returns percentage of joint flexion relative to reference.
 */
export function normalizeBone(u: Vec3, v: Vec3, refLength: number): number {
    const d = mag(sub(v, u));
    return d / refLength;
}
