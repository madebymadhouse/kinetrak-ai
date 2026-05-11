/**
 * Zero-copy Rolling Ring Buffer
 * 
 * Performance Constraint: Fixed-size buffer to prevent GC churn during high-frequency pose updates.
 * Optimization: Stride-based memory access for raw Float32Array frames.
 */

export class PoseRingBuffer {
    private buffer: Float32Array;
    private head: number = 0;
    private size: number;
    private stride: number;
    private count: number = 0;

    /**
     * @param capacity Number of frames to store (e.g., 30 for 1 second at 30fps)
     * @param stride Elements per frame (e.g. 33 landmarks * 4 components [x,y,z,v])
     */
    constructor(capacity: number, stride: number) {
        this.size = capacity;
        this.stride = stride;
        this.buffer = new Float32Array(capacity * stride);
    }

    push(frame: Float32Array) {
        const offset = this.head * this.stride;
        this.buffer.set(frame, offset);
        this.head = (this.head + 1) % this.size;
        this.count = Math.min(this.count + 1, this.size);
    }

    getFrame(index: number): Float32Array {
        // index: 0 is latest, 1 is previous, etc.
        const actualIndex = (this.head - 1 - index + this.size) % this.size;
        const offset = actualIndex * this.stride;
        return this.buffer.subarray(offset, offset + this.stride);
    }

    /**
     * Calculates velocity vector (dx, dy, dz) for a landmark.
     * Logic: P(t) - P(t-1)
     */
    getVelocity(landmarkIndex: number, out = new Float32Array(3)): Float32Array {
        if (this.count < 2) return out;
        
        const f0 = this.getFrame(0);
        const f1 = this.getFrame(1);
        
        const off = landmarkIndex * 4;
        out[0] = f0[off] - f1[off];
        out[1] = f0[off+1] - f1[off+1];
        out[2] = f0[off+2] - f1[off+2];
        
        return out;
    }

    /**
     * Calculates acceleration vector (dv/dt) for a landmark.
     * Logic: V(t) - V(t-1)
     */
    getAcceleration(landmarkIndex: number, out = new Float32Array(3)): Float32Array {
        if (this.count < 3) return out;
        
        const v0 = this.getVelocity(landmarkIndex);
        
        // Calculate previous velocity V(t-1) = P(t-1) - P(t-2)
        const f1 = this.getFrame(1);
        const f2 = this.getFrame(2);
        const off = landmarkIndex * 4;
        
        const v1x = f1[off] - f2[off];
        const v1y = f1[off+1] - f2[off+1];
        const v1z = f1[off+2] - f2[off+2];
        
        out[0] = v0[0] - v1x;
        out[1] = v0[1] - v1y;
        out[2] = v0[2] - v1z;
        
        return out;
    }

    /**
     * Exponential Moving Average of Velocity for stability.
     */
    getAverageVelocity(landmarkIndex: number, window: number, out = new Float32Array(3)): Float32Array {
        const n = Math.min(this.count - 1, window);
        if (n < 1) return out;
        
        out.fill(0);
        for (let i = 0; i < n; i++) {
            const f0 = this.getFrame(i);
            const f1 = this.getFrame(i + 1);
            const off = landmarkIndex * 4;
            out[0] += (f0[off] - f1[off]);
            out[1] += (f0[off+1] - f1[off+1]);
            out[2] += (f0[off+2] - f1[off+2]);
        }
        
        const invN = 1 / n;
        out[0] *= invN;
        out[1] *= invN;
        out[2] *= invN;
        
        return out;
    }

    /**
     * Predicts future position based on Momentum Model (V2).
     * Prediction: P(t) + V*t*decay + 0.5*A*t^2*decay
     * Reason: Decay prevents massive overshoot during jerky sensor updates.
     */
    predictPosition(landmarkIndex: number, leadFrames: number, out = new Float32Array(3)): Float32Array {
        const f0 = this.getFrame(0);
        const off = landmarkIndex * 4;
        
        // Weighted Velocity (More weight to recent change)
        const v = this.getAverageVelocity(landmarkIndex, 2);
        const a = this.getAcceleration(landmarkIndex);
        
        const t = leadFrames;
        const decay = Math.exp(-t * 0.1); // Momentum decay
        
        out[0] = f0[off] + (v[0] * t + 0.5 * a[0] * t * t) * decay;
        out[1] = f0[off+1] + (v[1] * t + 0.5 * a[1] * t * t) * decay;
        out[2] = f0[off+2] + (v[2] * t + 0.5 * a[2] * t * t) * decay;
        
        return out;
    }
}
