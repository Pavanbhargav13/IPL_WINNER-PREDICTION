import React, { useEffect, useRef } from 'react';

export function countUp(el, target, duration = 1200) {
  if (!el) return;
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 4); // easeOutQuart
    el.textContent = (target * ease).toFixed(1) + "%";
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

export default function AnimatedNumber({ value, duration = 1200, className, style }) {
  const numRef = useRef(null);

  useEffect(() => {
    if (numRef.current) {
      countUp(numRef.current, value, duration);
    }
  }, [value, duration]);

  return <span ref={numRef} className={className} style={style}>0.0%</span>;
}
