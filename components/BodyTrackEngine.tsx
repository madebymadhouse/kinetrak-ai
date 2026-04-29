import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Vector3, Quaternion, TemporalRingBuffer, KalmanVector3 } from '../lib/kinematics';

interface Point {
    x: number;
    y: number;
    z?: number;
    visibility?: number;
}

interface TelemetryData {
    engineActive: boolean;
    poseDetected: boolean;
    faceDetected: boolean;
    handsDetected: boolean;
    
    // Pose Deep Metrics
    shoulderAlignment?: number;
    hipAlignment?: number;
    spineFlexion?: number;
    leftElbowAngle?: number;
    rightElbowAngle?: number;
    leftKneeAngle?: number;
    rightKneeAngle?: number;
    
    // Kinetic Rates
    leftWristVelocity?: number;
    rightWristVelocity?: number;
    leftAnkleVelocity?: number;
    rightAnkleVelocity?: number;
    
    // Hand Deep Metrics
    leftPinchDistance?: number;
    rightPinchDistance?: number;
    leftFingerSpread?: number;
    rightFingerSpread?: number;
    
    // Face Deep Metrics
    mouthOpenness?: number;
    leftEyeOpenness?: number;
    rightEyeOpenness?: number;
    headTilt?: number;
    
    // Classified States & Z-Depth
    leftHandState?: 'Open' | 'Closed' | 'Pinching' | 'Unknown';
    rightHandState?: 'Open' | 'Closed' | 'Pinching' | 'Unknown';
    depthEstimation?: number;

    // Advanced Intelligent Metrics
    postureClass?: 'Optimal' | 'Kyphotic' | 'Unknown';
    shoulderDrop?: number;
    centerOfMassX?: number;
    centerOfMassY?: number;

    // Kinematic Solvers
    leftArmQuat?: string;
    rightArmQuat?: string;
}

interface LogEntry {
    id: number;
    message: string;
    level: 'Info' | 'Warning' | 'Error' | 'System';
    time: string;
}

const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Upper body
    [11, 23], [12, 24], [23, 24], // Torso
    [23, 25], [25, 27], [24, 26], [26, 28] // Lower body
];

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [5, 9], [9, 10], [10, 11], [11, 12], // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [0, 17] // Wrist to base
];

const calculateAngle = (a: Point, b: Point, c: Point) => {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
};

const calculateDistance = (a: Point, b: Point) => {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
};

const calculateHorizontalAngle = (a: Point, b: Point) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return (Math.atan2(dy, dx) * 180) / Math.PI;
};

const calculateVerticalAngle = (p1: Point, p2: Point) => {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    return Math.abs(angle + 90);
}

const BodyTrackEngine: React.FC = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const latestResults = useRef<any>(null);
    const prevTimeRef = useRef<number>(0);
    const prevPosRef = useRef<any>({});
    
    // Kinematic Buffers and Filters
    const poseBufferRef = useRef(new TemporalRingBuffer<any>(30));
    const kFiltersRef = useRef<Record<number, KalmanVector3>>({});
    
    const [loading, setLoading] = useState(true);
    const [engineActive, setEngineActive] = useState(true);
    const [cameraMode, setCameraMode] = useState<'Standard' | 'Grayscale' | 'Thermal' | 'Hidden'>('Standard');
    const [trackingLayer, setTrackingLayer] = useState<'System Status' | 'Skeleton' | 'Upper Body' | 'Cranial' | 'Hands'>('System Status');
    const [renderLines, setRenderLines] = useState(true);
    const [renderData, setRenderData] = useState(true);
    const [trackerColor, setTrackerColor] = useState<string>('#ffffff');
    const [renderGrid, setRenderGrid] = useState<boolean>(true);
    const [useDepth, setUseDepth] = useState<boolean>(true);
    
    const [telemetry, setTelemetry] = useState<TelemetryData>({ 
        engineActive: false, poseDetected: false, faceDetected: false, handsDetected: false 
    });
    
    const [fps, setFps] = useState(0);
    const [logs, setLogs] = useState<LogEntry[]>([]);

    const addLog = useCallback((message: string, level: LogEntry['level'] = 'Info') => {
        setLogs(prev => {
            const timeStr = new Date().toLocaleTimeString(undefined, { 
                hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 
            } as any);
            const newLogs = [{ id: Date.now() + Math.random(), message, level, time: timeStr }, ...prev];
            return newLogs.slice(0, 50);
        });
    }, []);

    const toggleEngine = () => {
        setEngineActive(prev => {
            const next = !prev;
            addLog(next ? "Sensory input resumed" : "Sensory input paused", next ? "Info" : "Warning");
            return next;
        });
    };

    const emasRef = useRef<Record<string, number>>({});
    
    // Data Processing Loop
    useEffect(() => {
        if (!engineActive) return;
        
        const getEMA = (key: string, val: number, alpha: number = 0.25) => {
            if (emasRef.current[key] === undefined || isNaN(emasRef.current[key])) { emasRef.current[key] = val; }
            else { emasRef.current[key] = (alpha * val) + (1 - alpha) * emasRef.current[key]; }
            return emasRef.current[key];
        };

        const interval = setInterval(() => {
            const res = latestResults.current;
            if (!res) return;

            const now = performance.now();
            const dt = (now - prevTimeRef.current) / 1000 || 0.01;
            prevTimeRef.current = now;

            const cw = canvasRef.current?.width || 1;
            const ch = canvasRef.current?.height || 1;
            
            const mapPt = (lmList: any[], idx: number) => {
                const lm = lmList ? lmList[idx] : null;
                return lm ? { x: lm.x * cw, y: lm.y * ch, z: lm.z, visibility: lm.visibility } : undefined;
            };

            const data: TelemetryData = {
                engineActive: true,
                poseDetected: !!res.poseLandmarks,
                faceDetected: !!res.faceLandmarks,
                handsDetected: !!(res.leftHandLandmarks || res.rightHandLandmarks),
            };

            if (res.poseLandmarks) {
                // Buffer raw input for temporal processing
                poseBufferRef.current.push(res.poseLandmarks);
                
                // Retrieve Kalman filtered landmarks
                const getSmoothed = (idx: number) => {
                    if (!kFiltersRef.current[idx]) {
                        kFiltersRef.current[idx] = new KalmanVector3();
                    }
                    const raw = res.poseLandmarks[idx];
                    if (!raw) return null;
                    const smoothed = kFiltersRef.current[idx].update(raw.x, raw.y, raw.z || 0);
                    return { x: smoothed.x * cw, y: smoothed.y * ch, z: smoothed.z, visibility: raw.visibility };
                };

                const lShoulder = getSmoothed(11);
                const rShoulder = getSmoothed(12);
                const lHip = getSmoothed(23);
                const rHip = getSmoothed(24);
                
                const lElbow = getSmoothed(13);
                const rElbow = getSmoothed(14);
                const lWrist = getSmoothed(15);
                const rWrist = getSmoothed(16);
                
                const lKnee = getSmoothed(25);
                const rKnee = getSmoothed(26);
                const lAnkle = getSmoothed(27);
                const rAnkle = getSmoothed(28);

                // IK & Quaternion State Extractor
                if (lShoulder && lElbow) {
                    const u = new Vector3(lShoulder.x, lShoulder.y, lShoulder.z!);
                    const v = new Vector3(lElbow.x, lElbow.y, lElbow.z!);
                    const boneVector = v.sub(u);
                    const quat = Quaternion.fromVectors(new Vector3(0, -1, 0), boneVector);
                    data.leftArmQuat = quat.toString();
                }
                if (rShoulder && rElbow) {
                    const u = new Vector3(rShoulder.x, rShoulder.y, rShoulder.z!);
                    const v = new Vector3(rElbow.x, rElbow.y, rElbow.z!);
                    const boneVector = v.sub(u);
                    const quat = Quaternion.fromVectors(new Vector3(0, -1, 0), boneVector);
                    data.rightArmQuat = quat.toString();
                }

                // Advanced Biomechanics: Center of Mass & Asymmetry
                if (lShoulder && rShoulder && lHip && rHip) {
                    const comX = (lShoulder.x + rShoulder.x + lHip.x + rHip.x) / 4;
                    const comY = (lShoulder.y + rShoulder.y + lHip.y + rHip.y) / 4;
                    data.centerOfMassX = getEMA('comX', comX, 0.4);
                    data.centerOfMassY = getEMA('comY', comY, 0.4);
                    
                    data.shoulderDrop = getEMA('shoulderDrop', lShoulder.y - rShoulder.y, 0.2);
                    if (data.spineFlexion && data.spineFlexion > 15) {
                        data.postureClass = 'Kyphotic';
                    } else {
                        data.postureClass = 'Optimal';
                    }
                }

                // Deep Tracking & Depth Estimation (from nose z)
                const nose = mapPt(res.poseLandmarks, 0);
                if (nose && nose.z !== undefined) {
                    // Depth is relative, we map it to an arbitrary pseudo-metric
                    data.depthEstimation = getEMA('depthEstimation', Math.abs(nose.z) * 1000);
                }

                // Deep Metric: Alignments
                if (lShoulder && rShoulder) data.shoulderAlignment = getEMA('shoulderAlignment', Math.abs(calculateHorizontalAngle(rShoulder, lShoulder)));
                if (lHip && rHip) data.hipAlignment = getEMA('hipAlignment', Math.abs(calculateHorizontalAngle(rHip, lHip)));
                if (lShoulder && rShoulder && lHip && rHip) {
                    const midShoulder = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
                    const midHip = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
                    data.spineFlexion = getEMA('spineFlexion', calculateVerticalAngle(midHip, midShoulder));
                }

                // Deep Metric: Angles
                if (lShoulder && lElbow && lWrist) data.leftElbowAngle = getEMA('lElbow', calculateAngle(lShoulder, lElbow, lWrist));
                if (rShoulder && rElbow && rWrist) data.rightElbowAngle = getEMA('rElbow', calculateAngle(rShoulder, rElbow, rWrist));
                if (lHip && lKnee && lAnkle) data.leftKneeAngle = getEMA('lKnee', calculateAngle(lHip, lKnee, lAnkle));
                if (rHip && rKnee && rAnkle) data.rightKneeAngle = getEMA('rKnee', calculateAngle(rHip, rKnee, rAnkle));
                
                // Deep Metric: Velocities
                const prev = prevPosRef.current;
                
                if (prev.lWrist && lWrist) data.leftWristVelocity = getEMA('lWristVel', calculateDistance(lWrist, prev.lWrist) / dt);
                if (prev.rWrist && rWrist) data.rightWristVelocity = getEMA('rWristVel', calculateDistance(rWrist, prev.rWrist) / dt);
                if (prev.lAnkle && lAnkle) data.leftAnkleVelocity = getEMA('lAnkleVel', calculateDistance(lAnkle, prev.lAnkle) / dt);
                if (prev.rAnkle && rAnkle) data.rightAnkleVelocity = getEMA('rAnkleVel', calculateDistance(rAnkle, prev.rAnkle) / dt);

                prevPosRef.current = { ...prev, lWrist, rWrist, lAnkle, rAnkle };
            }

            if (res.leftHandLandmarks) {
                const base = mapPt(res.leftHandLandmarks, 0);
                const thumb = mapPt(res.leftHandLandmarks, 4);
                const index = mapPt(res.leftHandLandmarks, 8);
                const pinky = mapPt(res.leftHandLandmarks, 20);
                
                if (thumb && index) data.leftPinchDistance = getEMA('lPinch', calculateDistance(thumb, index));
                if (thumb && pinky) data.leftFingerSpread = getEMA('lSpread', calculateDistance(thumb, pinky));
                
                if (base && index && pinky && thumb) {
                    const avgDist = (calculateDistance(base, index) + calculateDistance(base, pinky) + calculateDistance(base, thumb)) / 3;
                    if (data.leftPinchDistance! < 20) data.leftHandState = 'Pinching';
                    else if (avgDist < 60) data.leftHandState = 'Closed';
                    else data.leftHandState = 'Open';
                }
            }

            if (res.rightHandLandmarks) {
                const base = mapPt(res.rightHandLandmarks, 0);
                const thumb = mapPt(res.rightHandLandmarks, 4);
                const index = mapPt(res.rightHandLandmarks, 8);
                const pinky = mapPt(res.rightHandLandmarks, 20);
                
                if (thumb && index) data.rightPinchDistance = getEMA('rPinch', calculateDistance(thumb, index));
                if (thumb && pinky) data.rightFingerSpread = getEMA('rSpread', calculateDistance(thumb, pinky));
                
                if (base && index && pinky && thumb) {
                    const avgDist = (calculateDistance(base, index) + calculateDistance(base, pinky) + calculateDistance(base, thumb)) / 3;
                    if (data.rightPinchDistance! < 20) data.rightHandState = 'Pinching';
                    else if (avgDist < 60) data.rightHandState = 'Closed';
                    else data.rightHandState = 'Open';
                }
            }

            if (res.faceLandmarks) {
                const upperLip = mapPt(res.faceLandmarks, 13);
                const lowerLip = mapPt(res.faceLandmarks, 14);
                const leftEyeTop = mapPt(res.faceLandmarks, 159);
                const leftEyeBot = mapPt(res.faceLandmarks, 145);
                const rightEyeTop = mapPt(res.faceLandmarks, 386);
                const rightEyeBot = mapPt(res.faceLandmarks, 374);
                
                const leftSide = mapPt(res.faceLandmarks, 234);
                const rightSide = mapPt(res.faceLandmarks, 454);

                if (upperLip && lowerLip) data.mouthOpenness = getEMA('mouth', calculateDistance(upperLip, lowerLip));
                if (leftEyeTop && leftEyeBot) data.leftEyeOpenness = getEMA('lEye', calculateDistance(leftEyeTop, leftEyeBot));
                if (rightEyeTop && rightEyeBot) data.rightEyeOpenness = getEMA('rEye', calculateDistance(rightEyeTop, rightEyeBot));
                if (leftSide && rightSide) data.headTilt = getEMA('headTilt', Math.abs(calculateHorizontalAngle(rightSide, leftSide)));
            }

            setTelemetry(data);
        }, 100); 
        
        return () => clearInterval(interval);
    }, [engineActive]);

    // Rendering Pipeline
    useEffect(() => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;
        let lastTime = performance.now();
        let frameCount = 0;

        const renderLoop = (time: number) => {
            frameCount++;
            if (time - lastTime >= 1000) {
                setFps(frameCount);
                frameCount = 0;
                lastTime = time;
            }

            if (video.videoWidth && video.videoHeight) {
                if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    addLog(`Calibration aligned intrinsic optics: ${video.videoWidth}x${video.videoHeight}`);
                }
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (renderGrid) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
                ctx.lineWidth = 1;
                const gridSize = Math.max(40, canvas.height / 20);
                for(let x = 0; x < canvas.width; x += gridSize) {
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
                }
                for(let y = 0; y < canvas.height; y += gridSize) {
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                }
            }

            if (!engineActive || !latestResults.current || !renderLines) {
                animationFrameId = requestAnimationFrame(renderLoop);
                return;
            }

            const res = latestResults.current;

            const hexToRgb = (hex: string) => {
                const r = parseInt(hex.slice(1, 3), 16) || 255;
                const g = parseInt(hex.slice(3, 5), 16) || 255;
                const b = parseInt(hex.slice(5, 7), 16) || 255;
                return `${r}, ${g}, ${b}`;
            };
            const rgbColor = hexToRgb(trackerColor);

            const mapLandmarks = (landmarks: any[]) => {
                return landmarks.map(lm => ({
                    x: lm.x * canvas.width,
                    y: lm.y * canvas.height,
                    z: lm.z,
                    visibility: lm.visibility
                }));
            };

            const drawSkeleton = (landmarks: Point[], connections: number[][], defaultOpacity: string = '0.4') => {
                ctx.lineWidth = 2.0;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                
                connections.forEach(([startIdx, endIdx]) => {
                    const start = landmarks[startIdx];
                    const end = landmarks[endIdx];
                    if (start && end) {
                        if ((start.visibility === undefined || start.visibility > 0.6) && 
                            (end.visibility === undefined || end.visibility > 0.6)) {
                            
                            ctx.beginPath();
                            if (useDepth && start.z !== undefined && end.z !== undefined) {
                                const grad = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
                                const alpha1 = Math.max(0.1, 1 - (start.z + 1) * 0.5);
                                const alpha2 = Math.max(0.1, 1 - (end.z + 1) * 0.5);
                                grad.addColorStop(0, `rgba(${rgbColor}, ${alpha1})`);
                                grad.addColorStop(1, `rgba(${rgbColor}, ${alpha2})`);
                                ctx.strokeStyle = grad;
                            } else {
                                ctx.strokeStyle = `rgba(${rgbColor}, ${defaultOpacity})`;
                            }
                            
                            ctx.moveTo(start.x, start.y);
                            ctx.lineTo(end.x, end.y);
                            ctx.stroke();
                        }
                    }
                });
            };

            const drawNodes = (landmarks: Point[], radius: number, isFocus: boolean = false, defaultOpacity: string = '0.9') => {
                landmarks.forEach((lm) => {
                    if (!lm || (lm.visibility !== undefined && lm.visibility < 0.6)) return;
                    
                    ctx.beginPath();
                    if (useDepth && lm.z !== undefined) {
                        const alpha = Math.max(0.2, 1 - (lm.z + 1) * 0.5);
                        ctx.fillStyle = `rgba(${rgbColor}, ${alpha})`;
                    } else {
                        ctx.fillStyle = `rgba(${rgbColor}, ${defaultOpacity})`;
                    }
                    
                    ctx.arc(lm.x, lm.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                    
                    if (isFocus) {
                        ctx.strokeStyle = `rgba(${rgbColor}, 0.5)`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(lm.x, lm.y, radius * 2.5, 0, Math.PI * 2);
                        ctx.stroke();
                    }
                });
            };

            const drawTrackingReticle = (x: number, y: number, label: string) => {
                ctx.strokeStyle = '#00ff88';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y);
                ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10);
                ctx.stroke();
                
                ctx.font = '10px "JetBrains Mono", monospace';
                ctx.fillStyle = '#00ff88';
                ctx.fillText(label, x + 12, y - 12);
            };

            const drawMetricLabel = (text: string, x: number, y: number) => {
                ctx.font = '400 12px "Inter", sans-serif';
                ctx.textAlign = 'center';
                
                const metrics = ctx.measureText(text);
                const w = metrics.width + 16;
                const h = 22;
                
                ctx.fillStyle = 'rgba(10, 10, 10, 0.7)';
                ctx.beginPath();
                ctx.roundRect(x - w/2, y - h/2 - 2, w, h, 6);
                ctx.fill();
                
                ctx.fillStyle = '#ffffff';
                ctx.fillText(text, x, y + 3);
            };

            if (res.poseLandmarks && (trackingLayer === 'System Status' || trackingLayer === 'Skeleton' || trackingLayer === 'Upper Body')) {
                const pose = mapLandmarks(res.poseLandmarks);
                
                let activeConnections = POSE_CONNECTIONS;
                let activeNodes = pose;

                if (trackingLayer === 'Upper Body') {
                    activeConnections = POSE_CONNECTIONS.filter(([a, b]) => a <= 24 && b <= 24);
                    activeNodes = pose.map((p, i) => i > 24 ? { ...p, visibility: 0 } : p);
                }

                drawSkeleton(activeNodes, activeConnections, '0.4');
                drawNodes(activeNodes, 2.5, false, '0.9');
                
                const vitalJoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].map(i => activeNodes[i]).filter(Boolean);
                drawNodes(vitalJoints, 4, true, '1.0');

                // Render Center of Mass
                if (trackingLayer !== 'Upper Body' && telemetry.centerOfMassX && telemetry.centerOfMassY) {
                    drawTrackingReticle(telemetry.centerOfMassX, telemetry.centerOfMassY, 'CG');
                }

                if (renderData) {
                    if (pose[11] && pose[13] && pose[15] && pose[13].visibility! > 0.65) {
                        drawMetricLabel(`${calculateAngle(pose[11], pose[13], pose[15]).toFixed(1)}°`, pose[13].x + 35, pose[13].y);
                    }
                    if (pose[12] && pose[14] && pose[16] && pose[14].visibility! > 0.65) {
                        drawMetricLabel(`${calculateAngle(pose[12], pose[14], pose[16]).toFixed(1)}°`, pose[14].x - 35, pose[14].y);
                    }
                    if (trackingLayer !== 'Upper Body') {
                        if (pose[23] && pose[25] && pose[27] && pose[25].visibility! > 0.65) {
                            drawMetricLabel(`${calculateAngle(pose[23], pose[25], pose[27]).toFixed(1)}°`, pose[25].x + 35, pose[25].y);
                        }
                        if (pose[24] && pose[26] && pose[28] && pose[26].visibility! > 0.65) {
                            drawMetricLabel(`${calculateAngle(pose[24], pose[26], pose[28]).toFixed(1)}°`, pose[26].x - 35, pose[26].y);
                        }
                    }
                }
            }

            if (res.leftHandLandmarks && (trackingLayer === 'System Status' || trackingLayer === 'Hands')) {
                const hand = mapLandmarks(res.leftHandLandmarks);
                drawSkeleton(hand, HAND_CONNECTIONS, '0.6');
                drawNodes(hand, 2.5, false, '1.0');
            }

            if (res.rightHandLandmarks && (trackingLayer === 'System Status' || trackingLayer === 'Hands')) {
                const hand = mapLandmarks(res.rightHandLandmarks);
                drawSkeleton(hand, HAND_CONNECTIONS, '0.6');
                drawNodes(hand, 2.5, false, '1.0');
            }

            if (res.faceLandmarks && (trackingLayer === 'System Status' || trackingLayer === 'Cranial')) {
                const face = mapLandmarks(res.faceLandmarks);
                
                // Draw all 468 nodes for extreme density mapping
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                face.forEach((lm) => {
                    ctx.fillRect(lm.x, lm.y, 1.5, 1.5);
                });
                
                // Highlight critical cranial features
                const eyes = [159, 145, 386, 374].map(i => face[i]).filter(Boolean);
                const mouth = [13, 14, 78, 308].map(i => face[i]).filter(Boolean);
                drawNodes(eyes, 2, true, '1.0');
                drawNodes(mouth, 2, true, '1.0');
            }

            animationFrameId = requestAnimationFrame(renderLoop);
        };

        animationFrameId = requestAnimationFrame(renderLoop);
        return () => cancelAnimationFrame(animationFrameId);
    }, [renderLines, renderData, engineActive, trackingLayer]);

    // Engine Initialization
    useEffect(() => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        let camera: any;
        let engine: any;

        const initializeSystems = async () => {
            const onResults = (results: any) => {
                if (loading) {
                    setLoading(false);
                    addLog("Neural parsing established and active.");
                }
                latestResults.current = results;
            };

            if ((window as any).Holistic) {
                addLog("Loading kinesiology framework logic.");
                engine = new (window as any).Holistic({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
                });

                engine.setOptions({
                    modelComplexity: 2, 
                    smoothLandmarks: true,
                    enableSegmentation: false,
                    smoothSegmentation: true,
                    refineFaceLandmarks: true,
                    minDetectionConfidence: 0.7,
                    minTrackingConfidence: 0.7,
                });

                engine.onResults(onResults);
                addLog("Framework ready. Establishing hardware connection.");

                if ((window as any).Camera) {
                    camera = new (window as any).Camera(video, {
                        onFrame: async () => {
                            if (inferenceActiveRef.current) {
                                await engine.send({ image: video });
                            }
                        },
                        width: 1280,
                        height: 720,
                    });
                    camera.start();
                    addLog("Optical stream successfully connected.");
                }
            }
        };

        initializeSystems();

        return () => {
            if (camera) camera.stop();
            if (engine) engine.close();
        };
    }, []); 

    const inferenceActiveRef = useRef(engineActive);
    useEffect(() => {
        inferenceActiveRef.current = engineActive;
    }, [engineActive]);

    const getVideoStyle = () => {
        if (cameraMode === 'Standard') return { opacity: 1, filter: 'none' };
        if (cameraMode === 'Grayscale') return { opacity: 0.7, filter: 'grayscale(100%)' };
        if (cameraMode === 'Thermal') return { opacity: 0.8, filter: 'invert(100%) hue-rotate(180deg) saturate(200%)' };
        return { opacity: 0, filter: 'none' };
    };

    return (
        <div className="flex flex-col lg:flex-row w-full h-screen bg-neutral-950 text-neutral-200 font-sans p-6 gap-6 overflow-hidden">
            
            {/* Sidebar Controls */}
            <div className="w-full lg:w-80 flex flex-col gap-6 z-40 shrink-0 h-full overflow-y-auto custom-scrollbar pr-2 pb-6">
                
                {/* Header */}
                <div className="bg-neutral-900/40 p-6 rounded-xl flex flex-col gap-5">
                    <div>
                        <h1 className="text-xl font-medium text-white mb-1">Kinetrak AI</h1>
                        <p className="text-sm text-neutral-400">Live movement analysis</p>
                    </div>

                    <button 
                        onClick={toggleEngine}
                        disabled={loading}
                        className={`w-full py-4 px-6 rounded-md text-sm font-medium transition-all ${loading ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed' : engineActive ? 'bg-white text-black hover:bg-neutral-200 shadow-lg' : 'bg-neutral-800 text-white hover:bg-neutral-700'}`}
                    >
                        {loading ? 'Initializing Engine...' : engineActive ? 'Pause Tracking' : 'Resume Tracking'}
                    </button>
                </div>

                {/* Configurations */}
                <div className="bg-neutral-900/40 p-6 rounded-xl flex flex-col gap-6">
                    <div className="flex justify-between items-center text-sm font-medium text-neutral-100">
                        <span>Engine Settings</span>
                    </div>
                    
                    <div className="flex flex-col gap-3">
                        <span className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Input Visuals</span>
                        <div className="flex flex-wrap gap-2">
                            {(['Standard', 'Grayscale', 'Thermal', 'Hidden'] as const).map(mode => (
                                <button 
                                    key={mode}
                                    onClick={() => setCameraMode(mode)}
                                    className={`py-2 px-3 text-[13px] font-medium rounded-md transition-colors flex-1 min-w-[30%] ${cameraMode === mode ? 'bg-neutral-700 text-white' : 'bg-neutral-800 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'}`}
                                >
                                    {mode}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="flex flex-col gap-3">
                        <span className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Analysis Target</span>
                        <div className="flex flex-wrap gap-2">
                            {(['System Status', 'Skeleton', 'Upper Body', 'Cranial', 'Hands'] as const).map(layer => (
                                <button 
                                    key={layer}
                                    onClick={() => setTrackingLayer(layer)}
                                    className={`py-2 px-3 text-[13px] font-medium rounded-md transition-colors flex-1 min-w-[40%] ${trackingLayer === layer ? 'bg-neutral-700 text-white' : 'bg-neutral-800 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'}`}
                                >
                                    {layer}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="flex flex-col gap-3">
                        <span className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Tracker Color</span>
                        <div className="flex gap-2">
                            {['#ffffff', '#00ff88', '#00e5ff', '#ff0055', '#ffaa00'].map(color => (
                                <button 
                                    key={color}
                                    onClick={() => setTrackerColor(color)}
                                    className={`w-8 h-8 rounded-md transition-all ${trackerColor === color ? 'ring-2 ring-white scale-110' : 'ring-1 ring-white/20 hover:scale-105'}`}
                                    style={{ backgroundColor: color }}
                                />
                            ))}
                        </div>
                    </div>

                    <div className="flex flex-col gap-5 pt-5 border-t border-neutral-800/60">
                        <label className="flex items-center justify-between cursor-pointer group">
                            <span className="text-[13px] text-neutral-300 group-hover:text-white transition-colors">Render skeletal lines</span>
                            <div className={`relative inline-flex h-6 w-11 items-center rounded-md transition-colors ${renderLines ? 'bg-white' : 'bg-neutral-700'}`}>
                                <input type="checkbox" className="sr-only" checked={renderLines} onChange={(e) => setRenderLines(e.target.checked)} />
                                <span className={`inline-block h-5 w-5 transform rounded-md bg-neutral-900 transition-transform ${renderLines ? 'translate-x-[22px]' : 'translate-x-[2px]'}`} />
                            </div>
                        </label>
                        <label className="flex items-center justify-between cursor-pointer group">
                            <span className="text-[13px] text-neutral-300 group-hover:text-white transition-colors">Render joint metrics</span>
                            <div className={`relative inline-flex h-6 w-11 items-center rounded-md transition-colors ${renderData ? 'bg-white' : 'bg-neutral-700'}`}>
                                <input type="checkbox" className="sr-only" checked={renderData} onChange={(e) => setRenderData(e.target.checked)} />
                                <span className={`inline-block h-5 w-5 transform rounded-md bg-neutral-900 transition-transform ${renderData ? 'translate-x-[22px]' : 'translate-x-[2px]'}`} />
                            </div>
                        </label>
                        <label className="flex items-center justify-between cursor-pointer group">
                            <span className="text-[13px] text-neutral-300 group-hover:text-white transition-colors">Render spatial grid</span>
                            <div className={`relative inline-flex h-6 w-11 items-center rounded-md transition-colors ${renderGrid ? 'bg-white' : 'bg-neutral-700'}`}>
                                <input type="checkbox" className="sr-only" checked={renderGrid} onChange={(e) => setRenderGrid(e.target.checked)} />
                                <span className={`inline-block h-5 w-5 transform rounded-md bg-neutral-900 transition-transform ${renderGrid ? 'translate-x-[22px]' : 'translate-x-[2px]'}`} />
                            </div>
                        </label>
                        <label className="flex items-center justify-between cursor-pointer group">
                            <span className="text-[13px] text-neutral-300 group-hover:text-white transition-colors">Depth gradients (Z-Index)</span>
                            <div className={`relative inline-flex h-6 w-11 items-center rounded-md transition-colors ${useDepth ? 'bg-white' : 'bg-neutral-700'}`}>
                                <input type="checkbox" className="sr-only" checked={useDepth} onChange={(e) => setUseDepth(e.target.checked)} />
                                <span className={`inline-block h-5 w-5 transform rounded-md bg-neutral-900 transition-transform ${useDepth ? 'translate-x-[22px]' : 'translate-x-[2px]'}`} />
                            </div>
                        </label>
                    </div>
                </div>

                {/* Live Metrics */}
                <div className="bg-neutral-900/40 rounded-xl flex flex-col flex-1 min-h-[350px] overflow-hidden">
                    <div className="p-5 border-b border-neutral-800/40 flex justify-between items-center text-sm font-medium text-neutral-100">
                        <span>Live Telemetry</span>
                        <div className="flex items-center gap-2">
                            <span className="text-xs text-neutral-400">{engineActive ? 'Active' : 'Standby'}</span>
                            <div className={`w-2 h-2 rounded-md ${engineActive && telemetry.poseDetected ? 'bg-green-400 animate-pulse' : 'bg-neutral-600'}`}></div>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-5 space-y-6 custom-scrollbar">
                        {/* Deep System Intelligence */}
                        <div className={`space-y-3 transition-opacity duration-300 ${telemetry.poseDetected ? 'opacity-100' : 'opacity-30'}`}>
                            <div className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Kinetic Intelligence</div>
                            <div className="grid grid-cols-2 gap-2 text-sm font-mono transform">
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1 items-center justify-center">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Relative Z-Scale</span>
                                    <span className="text-neutral-200">{telemetry.depthEstimation ? (telemetry.depthEstimation / 100).toFixed(2) : '--'}</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1 items-center justify-center">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Spine Deviation</span>
                                    <span className="text-neutral-200">{telemetry.spineFlexion ? telemetry.spineFlexion.toFixed(1) : '--'}°</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1 items-center justify-center">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Shoulder Drop</span>
                                    <span className="text-neutral-200">{telemetry.shoulderDrop !== undefined ? Math.abs(telemetry.shoulderDrop).toFixed(0) : '--'} px</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1 items-center justify-center">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">L-Hand State</span>
                                    <span className={`text-neutral-200 uppercase ${telemetry.leftHandState === 'Pinching' ? 'text-green-400' : ''}`}>{telemetry.leftHandState || '--'}</span>
                                </div>
                            </div>
                        </div>

                        {/* Pose Metrics */}
                        <div className={`space-y-3 transition-opacity duration-300 ${telemetry.poseDetected ? 'opacity-100' : 'opacity-30'}`}>
                            <div className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Core Biometrics</div>
                            <div className="grid grid-cols-2 gap-2 text-sm font-mono transform">
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Shoulder Align</span>
                                    <span className="text-neutral-200">{telemetry.shoulderAlignment?.toFixed(1) || '--'}°</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Hip Align</span>
                                    <span className="text-neutral-200">{telemetry.hipAlignment?.toFixed(1) || '--'}°</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Cervical Flexion</span>
                                    <span className="text-neutral-200">{telemetry.spineFlexion?.toFixed(1) || '--'}°</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">Cranial Tilt</span>
                                    <span className="text-neutral-200">{telemetry.headTilt?.toFixed(1) || '--'}°</span>
                                </div>
                            </div>
                        </div>

                        {/* Limb Dynamics */}
                        <div className={`space-y-3 transition-opacity duration-300 ${telemetry.poseDetected ? 'opacity-100' : 'opacity-30'}`}>
                            <div className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Limb Kinematics</div>
                            <div className="grid grid-cols-2 gap-2 text-sm font-mono">
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">L-Arm Velocity</span>
                                    <span className="text-neutral-200">{telemetry.leftWristVelocity?.toFixed(0) || '0'} px/s</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">R-Arm Velocity</span>
                                    <span className="text-neutral-200">{telemetry.rightWristVelocity?.toFixed(0) || '0'} px/s</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">L-Wrist Pinch</span>
                                    <span className="text-neutral-200">{telemetry.leftPinchDistance?.toFixed(1) || '--'} px</span>
                                </div>
                                <div className="bg-neutral-800/40 p-3 rounded-xl flex flex-col gap-1">
                                    <span className="text-[11px] font-sans text-neutral-500 tracking-wide uppercase">R-Wrist Pinch</span>
                                    <span className="text-neutral-200">{telemetry.rightPinchDistance?.toFixed(1) || '--'} px</span>
                                </div>
                            </div>
                        </div>
                        
                        {/* Precision Features */}
                        <div className={`space-y-3 transition-opacity duration-300 ${(telemetry.faceDetected || telemetry.handsDetected || telemetry.leftArmQuat) ? 'opacity-100' : 'opacity-30'}`}>
                            <div className="text-xs text-neutral-500 font-medium tracking-wider uppercase">Solver / Quaternion Rotations</div>
                            <div className="bg-neutral-800/40 p-4 rounded-xl flex flex-col gap-3 font-mono text-[13px]">
                                <div className="flex justify-between items-center">
                                    <span className="font-sans text-neutral-500">L-Arm [w,x,y,z]</span>
                                    <span className={`text-neutral-200 text-[10px]`}>{telemetry.leftArmQuat || '--'}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="font-sans text-neutral-500">R-Arm [w,x,y,z]</span>
                                    <span className={`text-neutral-200 text-[10px]`}>{telemetry.rightArmQuat || '--'}</span>
                                </div>
                                <div className="flex justify-between items-center border-t border-neutral-700/50 pt-2">
                                    <span className="font-sans text-neutral-500">L-Hand State</span>
                                    <span className={`text-neutral-200 uppercase ${telemetry.leftHandState === 'Pinching' ? 'text-green-400' : ''}`}>{telemetry.leftHandState || '--'}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="font-sans text-neutral-500">R-Hand State</span>
                                    <span className={`text-neutral-200 uppercase ${telemetry.rightHandState === 'Pinching' ? 'text-green-400' : ''}`}>{telemetry.rightHandState || '--'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Minimal Log Output */}
                <div className="bg-neutral-900/40 rounded-xl p-5 min-h-[220px] max-h-[250px] overflow-y-auto custom-scrollbar font-sans text-[13px] flex flex-col gap-3">
                    <div className="text-neutral-500 font-medium tracking-wider uppercase mb-1 text-xs">System Log</div>
                    {logs.map(log => (
                        <div key={log.id} className="flex gap-3 leading-snug">
                            <span className="text-neutral-600 font-mono text-xs shrink-0 pt-[2px]">{log.time}</span>
                            <span className={
                                log.level === 'System' ? 'text-neutral-400' :
                                log.level === 'Error' ? 'text-red-400' :
                                log.level === 'Warning' ? 'text-amber-400' :
                                'text-neutral-300'
                            }>{log.message}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Viewport Area */}
            <div className="flex-1 relative z-10 bg-neutral-900 rounded-xl overflow-hidden flex flex-col shadow-2xl" ref={containerRef}>
                <div className="absolute top-6 left-6 z-40 bg-neutral-950/50 backdrop-blur-md px-4 py-2 rounded-md text-[13px] text-white flex items-center gap-3 shadow-sm border border-white/5">
                    <span className="text-neutral-400">Pipeline Status</span>
                    <span className="font-mono bg-white/10 px-2 py-0.5 rounded-md">{fps} FPS</span>
                </div>
                
                <div className="relative w-full h-full bg-neutral-950">
                    <video 
                        ref={videoRef} 
                        className="absolute inset-0 w-full h-full object-cover scale-x-[-1] transition-all duration-500" 
                        style={getVideoStyle()} 
                        playsInline autoPlay muted 
                    />
                    <canvas 
                        ref={canvasRef} 
                        className="absolute inset-0 w-full h-full object-cover scale-x-[-1] z-20 pointer-events-none" 
                    />
                </div>
            </div>
            
        </div>
    );
};

export default BodyTrackEngine;
