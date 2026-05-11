import React, { useEffect, useRef, useState } from 'react';

/**
 * KINETRAK ENGINE CORE V14.0 - ZERO FLUFF EDITION
 * 
 * ARCHITECTURAL HARDENING:
 * - NO UI. ALL HOTKEY COMMANDS ([C]olor, [B]lack & White, [T]racking Only).
 * - HARDCODED FULL-FACIAL TOPOLOGY (Lips, Eyes, Brows, Oval).
 * - O(1) RAW DATA HUD (Fixed-position, high-frequency).
 * - ZERO-GC KINEMATIC RESOLUTION.
 */

const POSE_C = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const HAND_C = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const FACE_C = {
    lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 61],
    lEye: [33, 160, 158, 133, 153, 144, 33],
    rEye: [263, 387, 385, 362, 380, 373, 263],
    brows: [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    oval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
};

const BodyTrackEngine: React.FC = () => {
    const vRef = useRef<HTMLVideoElement>(null), cRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null), resRef = useRef<any>(null), teleRef = useRef<any>({});
    const [view, setView] = useState<'color' | 'bw' | 'tracking'>('bw');
    const [fps, setFps] = useState(0);

    useEffect(() => {
        const handleKeys = (e: KeyboardEvent) => {
            if (e.key === 'c') setView('color');
            if (e.key === 'b') setView('bw');
            if (e.key === 't') setView('tracking');
        };
        window.addEventListener('keydown', handleKeys);
        
        const ws = new WebSocket('ws://localhost:8080');
        ws.onmessage = (e) => {
            const m = JSON.parse(e.data);
            if (m.type === 'RIG_SOLVED') teleRef.current = m.data;
        };
        wsRef.current = ws;

        return () => {
            window.removeEventListener('keydown', handleKeys);
            ws.close();
        };
    }, []);

    useEffect(() => {
        const ctx = cRef.current?.getContext('2d'); if (!ctx) return;
        let id: number, fc = 0, lt = performance.now();

        const draw = (t: number) => {
            fc++; if (t - lt >= 1000) { setFps(fc); fc = 0; lt = t; }
            const v = vRef.current, c = cRef.current;
            if (!v || !c || v.readyState < 2) { id = requestAnimationFrame(draw); return; }
            if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }

            ctx.clearRect(0, 0, c.width, c.height);
            const res = resRef.current;
            if (res) {
                const line = (lms: any[], aI: number, bI: number, color: string, w = 1.0) => {
                    const a = lms?.[aI], b = lms?.[bI];
                    if (!a || !b || (a.visibility !== undefined && a.visibility < 0.4)) return;
                    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = w;
                    ctx.moveTo(a.x * c.width, a.y * c.height); ctx.lineTo(b.x * c.width, b.y * c.height);
                    ctx.stroke();
                };

                // BRUTALIST RENDERING: NO GLOW, NO TAPER. JUST DATA.
                POSE_C.forEach(([i,j]) => line(res.poseLandmarks, i, j, 'rgba(0, 255, 100, 0.6)', 1.5));
                if (res.leftHandLandmarks) HAND_C.forEach(([i,j]) => line(res.leftHandLandmarks, i, j, 'rgba(0, 200, 255, 0.7)', 1.0));
                if (res.rightHandLandmarks) HAND_C.forEach(([i,j]) => line(res.rightHandLandmarks, i, j, 'rgba(0, 200, 255, 0.7)', 1.0));

                if (res.faceLandmarks) {
                    ctx.lineWidth = 0.5; ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                    Object.values(FACE_C).forEach(indices => {
                        ctx.beginPath();
                        indices.forEach((idx, i) => {
                            const p = res.faceLandmarks[idx];
                            if (i === 0) ctx.moveTo(p.x * c.width, p.y * c.height); else ctx.lineTo(p.x * c.width, p.y * c.height);
                        });
                        ctx.stroke();
                    });
                }
            }
            id = requestAnimationFrame(draw);
        };
        id = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(id);
    }, []);

    useEffect(() => {
        const v = vRef.current; if (!v) return;
        const boot = async () => {
            if (!(window as any).Holistic) return;
            const engine = new (window as any).Holistic({ locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
            engine.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.7, minTrackingConfidence: 0.7 });
            engine.onResults((r: any) => {
                resRef.current = r;
                if (wsRef.current?.readyState === 1) {
                    const buf = new Float32Array(553 * 4);
                    const p = (lms: any[], o: number) => { if (lms) lms.forEach((lm, i) => { const off = (o+i)*4; buf[off]=lm.x; buf[off+1]=lm.y; buf[off+2]=lm.z; buf[off+3]=lm.visibility||0; }); };
                    p(r.poseLandmarks, 0); p(r.leftHandLandmarks, 33); p(r.rightHandLandmarks, 54); p(r.faceLandmarks, 75);
                    wsRef.current.send(buf.buffer);
                }
            });
            const cam = new (window as any).Camera(v, { onFrame: async () => { await engine.send({ image: v }); }, width: 1280, height: 720 });
            cam.start();
        };
        boot();
    }, []);

    return (
        <div className="w-full h-screen bg-black text-white font-mono overflow-hidden select-none cursor-none">
            {/* O(1) RAW DATA OVERLAY */}
            <div className="absolute top-4 left-4 z-50 text-[7px] text-white/40 leading-tight mix-blend-difference pointer-events-none">
                <div>[KINE_V14.0_RAW_FEED] SYNC_LOCKED | {fps} FPS</div>
                <div>VIEW: {view.toUpperCase()} | KEY_BIND: [C/B/T]</div>
                <div className="mt-4 space-y-0.5 text-white/60">
                    <div>ORIENT: P:{teleRef.current.pose?.pitch} Y:{teleRef.current.pose?.yaw} R:{teleRef.current.pose?.roll}</div>
                    <div>MUSCLE: Z:{teleRef.current.face?.zygomatic} C:{teleRef.current.face?.corrugator} O:{teleRef.current.face?.orbicularis}</div>
                    <div>GAZE_X: L:{teleRef.current.gaze?.l} R:{teleRef.current.gaze?.r}</div>
                </div>
            </div>

            <div className="w-full h-full relative">
                <video ref={vRef} className="absolute inset-0 w-full h-full object-cover scale-x-[-1]" style={{ opacity: view === 'tracking' ? 0 : view === 'bw' ? 0.3 : 0.4, filter: view === 'bw' ? 'grayscale(100%) contrast(150%) brightness(100%)' : 'none' }} playsInline muted />
                <canvas ref={cRef} className="absolute inset-0 w-full h-full z-10 scale-x-[-1]" />
            </div>
        </div>
    );
};

export default BodyTrackEngine;
