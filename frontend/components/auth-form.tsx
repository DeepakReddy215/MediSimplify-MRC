"use client";

import { useState } from "react";
import Link from "next/link";
import ClickSpark from "./ClickSpark";
import StarBorder from "./StarBorder";

type AuthMode = "login" | "signup";

export default function AuthForm({
  mode,
  onSubmit,
  loading,
  error
}: {
  mode: AuthMode;
  onSubmit: (payload: { email: string; password: string; full_name?: string }) => Promise<void>;
  loading: boolean;
  error: string | null;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");

  const isSignup = mode === "signup";

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!email || !password || (isSignup && !fullName)) return;
    await onSubmit({ email, password, full_name: fullName });
  };

  return (
    <StarBorder
      as="div"
      className="mx-auto w-full max-w-md"
      color="#3b82f6"
      speed="7s"
      innerClassName="glass-card !rounded-2xl !p-8 !text-left"
    >
      <h1 className="text-2xl font-bold">{isSignup ? "Create account" : "Welcome back"}</h1>
      <p className="mt-2 text-sm text-slate-300">
        {isSignup ? "Start simplifying reports in minutes." : "Login to view your report dashboard."}
      </p>

      <form className="mt-6 space-y-4" onSubmit={submit}>
        {isSignup && (
          <div>
            <label className="mb-2 block text-sm text-slate-300">Full name</label>
            <input
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm outline-none ring-blue-400/50 transition focus:ring"
              placeholder="John Doe"
            />
          </div>
        )}
        <div>
          <label className="mb-2 block text-sm text-slate-300">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm outline-none ring-blue-400/50 transition focus:ring"
            placeholder="you@example.com"
          />
        </div>
        <div>
          <label className="mb-2 block text-sm text-slate-300">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm outline-none ring-blue-400/50 transition focus:ring"
            placeholder="••••••••"
          />
        </div>

        {error && <p className="text-sm text-rose-300">{error}</p>}

        <ClickSpark sparkColor="#3b82f6" sparkSize={8} sparkCount={12} duration={600}>
          <button
            disabled={loading}
            className="w-full rounded-xl bg-blue-500 px-4 py-3 text-sm font-medium transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {loading ? "Please wait..." : isSignup ? "Create account" : "Login"}
          </button>
        </ClickSpark>
      </form>

      <p className="mt-6 text-center text-sm text-slate-400">
        {isSignup ? "Already have an account?" : "New here?"} {" "}
        <Link
          href={isSignup ? "/login" : "/signup"}
          className="text-blue-300 transition hover:text-blue-200"
        >
          {isSignup ? "Login" : "Create account"}
        </Link>
      </p>
    </StarBorder>
  );
}
