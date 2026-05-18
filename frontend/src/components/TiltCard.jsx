import React, { useEffect, useRef } from 'react';
import VanillaTilt from 'vanilla-tilt';

export default function TiltCard({ options, className, style, children }) {
  const tiltRef = useRef(null);

  useEffect(() => {
    // Disable on mobile as per Coding_Rules.md
    if (window.innerWidth < 768) return;
    
    if (tiltRef.current) {
      VanillaTilt.init(tiltRef.current, {
        max: 15,
        speed: 400,
        glare: true,
        'max-glare': 0.2,
        scale: 1.02,
        ...options
      });
    }

    return () => {
      if (tiltRef.current?.vanillaTilt) {
        tiltRef.current.vanillaTilt.destroy();
      }
    };
  }, [options]);

  return (
    <div ref={tiltRef} className={className} style={{ ...style, transformStyle: 'preserve-3d' }}>
      {/* Wrapper to hold children with 3D translation */}
      <div style={{ transform: 'translateZ(20px)' }}>
        {children}
      </div>
    </div>
  );
}
