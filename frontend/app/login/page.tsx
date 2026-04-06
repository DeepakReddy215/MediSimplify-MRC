"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import AuthForm from "@/components/auth-form";
import LaserFlow from "@/components/LaserFlow";
import { login } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { setSession } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (payload: { email: string; password: string }) => {
    setLoading(true);
    setError(null);
    try {
      const result = await login(payload);
      setSession(result.access_token, result.user);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="grid min-h-screen place-items-center bg-hero-grid px-6 py-12">
      <div className="relative w-full max-w-md">
        <div className="pointer-events-none absolute -top-40 left-1/2 z-0 h-[360px] w-[460px] -translate-x-1/2 opacity-90">
          <LaserFlow
            color="#3b82f6"
            horizontalBeamOffset={0}
            verticalBeamOffset={0}
            horizontalSizing={0.45}
            verticalSizing={6.6}
            wispDensity={1.35}
            wispSpeed={12.5}
            wispIntensity={5.6}
            flowSpeed={0.35}
            flowStrength={0.25}
            fogIntensity={0.5}
            fogScale={0.3}
            fogFallSpeed={0.6}
            decay={1.1}
            falloffStart={1.2}
          />
        </div>
        <div className="pointer-events-none absolute -top-3 left-1/2 z-20 h-8 w-56 -translate-x-1/2 rounded-full bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.55)_0%,rgba(59,130,246,0.2)_45%,rgba(59,130,246,0)_75%)] blur-sm" />

        <div className="relative z-10">
          <AuthForm mode="login" onSubmit={handleSubmit} loading={loading} error={error} />
        </div>
      </div>
    </main>
  );
}
