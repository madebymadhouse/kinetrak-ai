export class Vector3 {
    constructor(public x: number, public y: number, public z: number) {}
    
    sub(v: Vector3) {
        return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
    }
    
    add(v: Vector3) {
        return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
    }

    mult(s: number) {
        return new Vector3(this.x * s, this.y * s, this.z * s);
    }
    
    normalize() {
        const mag = Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2);
        if (mag === 0) return new Vector3(0, 0, 0);
        return new Vector3(this.x / mag, this.y / mag, this.z / mag);
    }

    cross(v: Vector3) {
        return new Vector3(
            this.y * v.z - this.z * v.y,
            this.z * v.x - this.x * v.z,
            this.x * v.y - this.y * v.x
        );
    }
    
    dot(v: Vector3) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }
}

export class Quaternion {
    constructor(public w: number, public x: number, public y: number, public z: number) {}
    
    static fromVectors(u: Vector3, v: Vector3) {
        const uNorm = u.normalize();
        const vNorm = v.normalize();
        const dot = uNorm.dot(vNorm);
        
        if (dot >= 0.999999) {
            return new Quaternion(1, 0, 0, 0); // Identity
        }
        if (dot <= -0.999999) {
            // 180 degree rotation around any orthogonal vector
            let ortho = new Vector3(1, 0, 0).cross(uNorm);
            if (ortho.x === 0 && ortho.y === 0 && ortho.z === 0) ortho = new Vector3(0, 1, 0).cross(uNorm);
            ortho = ortho.normalize();
            return new Quaternion(0, ortho.x, ortho.y, ortho.z);
        }
        
        const cross = uNorm.cross(vNorm);
        const qw = Math.sqrt((1 + dot) * 2);
        const invQw = 1 / qw;
        return new Quaternion(qw / 2, cross.x * invQw, cross.y * invQw, cross.z * invQw);
    }

    toString() {
        return `[${this.w.toFixed(2)}, ${this.x.toFixed(2)}, ${this.y.toFixed(2)}, ${this.z.toFixed(2)}]`;
    }
}

export class TemporalRingBuffer<T> {
    private buffer: T[];
    private head: number = 0;
    private max: number;

    constructor(size: number) {
        this.max = size;
        this.buffer = [];
    }

    push(item: T) {
        if (this.buffer.length < this.max) {
            this.buffer.push(item);
        } else {
            this.buffer[this.head] = item;
            this.head = (this.head + 1) % this.max;
        }
    }

    getLatest(): T | null {
        if (this.buffer.length === 0) return null;
        if (this.buffer.length < this.max) return this.buffer[this.buffer.length - 1];
        let idx = this.head - 1;
        if (idx < 0) idx = this.max - 1;
        return this.buffer[idx];
    }
    
    getAll(): T[] {
        if (this.buffer.length < this.max) return [...this.buffer];
        return [...this.buffer.slice(this.head), ...this.buffer.slice(0, this.head)];
    }
}

export class KalmanFilter {
    private q: number; // process noise
    private r: number; // sensor noise
    private p: number; // estimated error
    private x: number; // value
    private initialized: boolean = false;

    constructor(q: number = 0.005, r: number = 0.05, p: number = 1.0) {
        this.q = q;
        this.r = r;
        this.p = p;
        this.x = 0;
    }

    update(measurement: number): number {
        if (!this.initialized) {
            this.x = measurement;
            this.initialized = true;
            return this.x;
        }
        // prediction update
        this.p = this.p + this.q;

        // measurement update
        const k = this.p / (this.p + this.r);
        this.x = this.x + k * (measurement - this.x);
        this.p = (1 - k) * this.p;

        return this.x;
    }
}

export class KalmanVector3 {
    private kx = new KalmanFilter();
    private ky = new KalmanFilter();
    private kz = new KalmanFilter();

    update(x: number, y: number, z: number) {
        return new Vector3(
            this.kx.update(x),
            this.ky.update(y),
            this.kz.update(z)
        );
    }
}
