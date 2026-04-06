import { LucideIcon } from "lucide-react";
import StarBorder from "./StarBorder";

export default function FeatureCard({
  icon: Icon,
  title,
  description
}: {
  icon: LucideIcon;
  title: string;
  description: string;
}) {
  return (
    <StarBorder
      as="article"
      className="w-full transition hover:-translate-y-1"
      color="#3b82f6"
      speed="7s"
      innerClassName="glass-card !h-full !rounded-2xl !p-6 !text-left"
    >
      <div className="mb-4 inline-flex rounded-xl bg-blue-500/20 p-3 text-blue-300">
        <Icon size={20} />
      </div>
      <h3 className="mb-2 text-lg font-semibold">{title}</h3>
      <p className="text-sm leading-relaxed text-slate-300">{description}</p>
    </StarBorder>
  );
}
