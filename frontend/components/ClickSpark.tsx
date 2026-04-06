'use client';

import React, { useRef, useState } from 'react';

type SparkCSSVars = React.CSSProperties & {
  '--end-x': string;
  '--end-y': string;
};

interface ClickSparkProps {
  sparkColor?: string;
  sparkSize?: number;
  sparkRadius?: number;
  sparkCount?: number;
  duration?: number;
  children?: React.ReactNode;
}

interface Spark {
  id: number;
  x: number;
  y: number;
  angle: number;
}

export default function ClickSpark({
  sparkColor = '#fff',
  sparkSize = 10,
  sparkRadius = 50,
  sparkCount = 8,
  duration = 600,
  children
}: ClickSparkProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [sparks, setSparks] = useState<Spark[]>([]);
  const sparkIdRef = useRef(0);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Create spark particles
    const newSparks: Spark[] = [];
    for (let i = 0; i < sparkCount; i++) {
      newSparks.push({
        id: sparkIdRef.current++,
        x: clickX,
        y: clickY,
        angle: (i / sparkCount) * Math.PI * 2
      });
    }

    setSparks((prev) => [...prev, ...newSparks]);

    // Remove sparks after animation
    const timer = setTimeout(() => {
      setSparks((prev) =>
        prev.filter((spark) => !newSparks.some((ns) => ns.id === spark.id))
      );
    }, duration);

    return () => clearTimeout(timer);
  };

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      style={{ position: 'relative', cursor: 'pointer', display: 'inline-block' }}
    >
      {children}
      {sparks.map((spark) => (
        <Particle
          key={spark.id}
          x={spark.x}
          y={spark.y}
          angle={spark.angle}
          sparkColor={sparkColor}
          sparkSize={sparkSize}
          sparkRadius={sparkRadius}
          duration={duration}
        />
      ))}
    </div>
  );
}

interface ParticleProps {
  x: number;
  y: number;
  angle: number;
  sparkColor: string;
  sparkSize: number;
  sparkRadius: number;
  duration: number;
}

function Particle({
  x,
  y,
  angle,
  sparkColor,
  sparkSize,
  sparkRadius,
  duration
}: ParticleProps) {
  const distance = sparkRadius + Math.random() * 30;
  const endX = Math.cos(angle) * distance;
  const endY = Math.sin(angle) * distance;

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        pointerEvents: 'none',
        zIndex: 50
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: -sparkSize / 2,
          top: -sparkSize / 2,
          width: sparkSize,
          height: sparkSize,
          backgroundColor: sparkColor,
          borderRadius: '50%',
          boxShadow: `0 0 ${Math.max(3, sparkSize / 2)}px ${sparkColor}`,
          animation: `clickSpark ${duration}ms ease-out forwards`,
          transform: 'translate(0, 0)',
          '--end-x': `${endX}px`,
          '--end-y': `${endY}px`
        } as SparkCSSVars}
      />
      <style>{`
        @keyframes clickSpark {
          0% {
            opacity: 1;
            transform: translate(0, 0) scale(1);
          }
          100% {
            opacity: 0;
            transform: translate(var(--end-x, 0), var(--end-y, 0)) scale(0);
          }
        }
      `}</style>
    </div>
  );
}
