"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type MetadataResponse = {
  model_name: string;
  model_version: string;
  metrics: {
    rmse: number;
    mae: number;
    r2: number;
  };
  ui_options: {
    companies: string[];
    types: string[];
    ram_options: number[];
    screen_resolutions: string[];
    cpu_brands: string[];
    hdd_options: number[];
    ssd_options: number[];
    gpu_brands: string[];
    os_options: string[];
  };
};

type PredictionResponse = {
  predicted_price_inr: number;
  model_name: string;
  model_version: string;
  request_id: string;
  latency_ms: number;
  currency: string;
};

type FormState = {
  company: string;
  type_name: string;
  ram: number;
  weight: number;
  touchscreen: boolean;
  ips: boolean;
  screen_size: number;
  screen_resolution: string;
  cpu_brand: string;
  hdd: number;
  ssd: number;
  gpu_brand: string;
  os: string;
};

type Preset = {
  id: string;
  title: string;
  subtitle: string;
  values: Partial<FormState>;
};

const defaultFormState: FormState = {
  company: "",
  type_name: "",
  ram: 8,
  weight: 2.1,
  touchscreen: false,
  ips: true,
  screen_size: 15.6,
  screen_resolution: "1920x1080",
  cpu_brand: "",
  hdd: 1000,
  ssd: 256,
  gpu_brand: "",
  os: "",
};

const stepLabels = ["Preset", "Basics", "Hardware", "Review"];

const presets: Preset[] = [
  {
    id: "student",
    title: "Student",
    subtitle: "Balanced everyday setup",
    values: {
      type_name: "Notebook",
      ram: 8,
      cpu_brand: "Intel Core i5",
      ssd: 256,
      hdd: 0,
      gpu_brand: "Intel",
      os: "Windows",
      touchscreen: false,
      ips: true,
      screen_size: 14,
      screen_resolution: "1920x1080",
      weight: 1.5,
    },
  },
  {
    id: "gaming",
    title: "Gaming",
    subtitle: "Higher performance parts",
    values: {
      type_name: "Gaming",
      ram: 16,
      cpu_brand: "Intel Core i7",
      ssd: 512,
      hdd: 1000,
      gpu_brand: "Nvidia",
      os: "Windows",
      touchscreen: false,
      ips: true,
      screen_size: 15.6,
      screen_resolution: "1920x1080",
      weight: 2.5,
    },
  },
  {
    id: "creator",
    title: "Creator",
    subtitle: "Sharper display and SSD-first",
    values: {
      type_name: "Workstation",
      ram: 16,
      cpu_brand: "Intel Core i7",
      ssd: 512,
      hdd: 0,
      gpu_brand: "Nvidia",
      os: "Windows",
      touchscreen: false,
      ips: true,
      screen_size: 15.6,
      screen_resolution: "3840x2160",
      weight: 1.9,
    },
  },
  {
    id: "portable",
    title: "Portable",
    subtitle: "Lightweight travel setup",
    values: {
      type_name: "Ultrabook",
      ram: 8,
      cpu_brand: "Intel Core i5",
      ssd: 256,
      hdd: 0,
      gpu_brand: "Intel",
      os: "Windows",
      touchscreen: false,
      ips: true,
      screen_size: 13.3,
      screen_resolution: "1920x1080",
      weight: 1.2,
    },
  },
];

function toCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

export default function Home() {
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [formState, setFormState] = useState<FormState>(defaultFormState);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMetadata() {
      try {
        const response = await fetch("/api/metadata", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail ?? "Could not load model metadata.");
        }

        setMetadata(payload);
        setFormState({
          company: payload.ui_options.companies[0] ?? "Dell",
          type_name: payload.ui_options.types[0] ?? "Notebook",
          ram: payload.ui_options.ram_options.includes(8) ? 8 : payload.ui_options.ram_options[0] ?? 8,
          weight: 2.1,
          touchscreen: false,
          ips: true,
          screen_size: 15.6,
          screen_resolution: payload.ui_options.screen_resolutions.includes("1920x1080") ? "1920x1080" : payload.ui_options.screen_resolutions[0] ?? "1920x1080",
          cpu_brand: payload.ui_options.cpu_brands.includes("Intel Core i5") ? "Intel Core i5" : payload.ui_options.cpu_brands[0] ?? "Intel Core i5",
          hdd: payload.ui_options.hdd_options.includes(1000) ? 1000 : payload.ui_options.hdd_options[0] ?? 0,
          ssd: payload.ui_options.ssd_options.includes(256) ? 256 : payload.ui_options.ssd_options[0] ?? 0,
          gpu_brand: payload.ui_options.gpu_brands.includes("Nvidia") ? "Nvidia" : payload.ui_options.gpu_brands[0] ?? "Nvidia",
          os: payload.ui_options.os_options.includes("Windows") ? "Windows" : payload.ui_options.os_options[0] ?? "Windows",
        });
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Could not initialize the predictor.");
      } finally {
        setIsLoadingMetadata(false);
      }
    }

    void loadMetadata();
  }, []);

  const confidenceLabel = useMemo(() => {
    if (!metadata) {
      return "Loading";
    }
    if (metadata.metrics.r2 >= 0.8) {
      return "High confidence";
    }
    if (metadata.metrics.r2 >= 0.65) {
      return "Good confidence";
    }
    return "Baseline model";
  }, [metadata]);

  function updateField<Key extends keyof FormState>(key: Key, value: FormState[Key]) {
    setActivePreset(null);
    setFormState((current) => ({ ...current, [key]: value }));
  }

  function applyPreset(preset: Preset) {
    setActivePreset(preset.id);
    setPrediction(null);
    setFormState((current) => ({ ...current, ...preset.values }));
    setActiveStep(1);
  }

  function nextStep() {
    setActiveStep((current) => Math.min(current + 1, stepLabels.length - 1));
  }

  function previousStep() {
    setActiveStep((current) => Math.max(current - 1, 0));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formState),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Prediction request failed.");
      }
      setPrediction(payload);
    } catch (submitError) {
      setPrediction(null);
      setError(submitError instanceof Error ? submitError.message : "Prediction request failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-4 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <section className="app-frame rounded-[2rem] p-4 sm:p-6 lg:p-8">
          <header className="flex flex-col gap-6 border-b border-slate-900/8 pb-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-4">
              <LogoMark />
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-500">Laptop Price Predictor</p>
                <h1 className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-slate-950 sm:text-3xl">
                  Step-by-step laptop price estimation.
                </h1>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <InfoBadge label={isLoadingMetadata ? "Connecting" : "Ready"} tone="neutral" />
              <InfoBadge label={confidenceLabel} tone="accent" />
            </div>
          </header>

          <div className="mt-6 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <section className="panel p-5 sm:p-6">
              <div className="max-w-2xl">
                <p className="text-sm leading-6 text-slate-600">
                  Move through a short guided flow, review the selected configuration, and generate a price estimate.
                </p>
              </div>

              {error ? <div className="mt-5 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}

              <div className="mt-6 grid gap-3 sm:grid-cols-4">
                {stepLabels.map((label, index) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => setActiveStep(index)}
                    className={`step-pill ${index === activeStep ? "step-pill-active" : "step-pill-idle"}`}
                  >
                    <span className="step-pill-number">{index + 1}</span>
                    <span>{label}</span>
                  </button>
                ))}
              </div>

              <form onSubmit={handleSubmit} className="mt-6 space-y-8">
                {activeStep === 0 ? (
                  <section className="space-y-6">
                    <FieldGroup title="Choose a starting point" subtitle="Pick a preset to fill the form quickly, or continue with the default values.">
                      <div className="grid gap-4 sm:grid-cols-2">
                        {presets.map((preset) => (
                          <button
                            key={preset.id}
                            type="button"
                            onClick={() => applyPreset(preset)}
                            className={`preset-card ${activePreset === preset.id ? "preset-card-active" : "preset-card-idle"}`}
                          >
                            <span>
                              <span className="block text-base font-semibold text-slate-950">{preset.title}</span>
                              <span className="mt-1 block text-sm text-slate-500">{preset.subtitle}</span>
                            </span>
                            <span className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-700">Use preset</span>
                          </button>
                        ))}
                      </div>
                    </FieldGroup>
                  </section>
                ) : null}

                {activeStep === 1 ? (
                  <section className="space-y-8">
                    <FieldGroup title="Basic details" subtitle="Core identity and usage category.">
                      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        <SelectField label="Brand" value={formState.company} options={metadata?.ui_options.companies ?? []} disabled={!metadata} onChange={(value) => updateField("company", value)} />
                        <SelectField label="Type" value={formState.type_name} options={metadata?.ui_options.types ?? []} disabled={!metadata} onChange={(value) => updateField("type_name", value)} />
                        <SelectField label="Operating system" value={formState.os} options={metadata?.ui_options.os_options ?? []} disabled={!metadata} onChange={(value) => updateField("os", value)} />
                      </div>
                    </FieldGroup>
                  </section>
                ) : null}

                {activeStep === 2 ? (
                  <section className="space-y-8">
                    <FieldGroup title="Performance" subtitle="Processor, graphics, memory, and storage.">
                      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        <SelectField label="Processor" value={formState.cpu_brand} options={metadata?.ui_options.cpu_brands ?? []} disabled={!metadata} onChange={(value) => updateField("cpu_brand", value)} />
                        <SelectField label="Graphics" value={formState.gpu_brand} options={metadata?.ui_options.gpu_brands ?? []} disabled={!metadata} onChange={(value) => updateField("gpu_brand", value)} />
                        <SelectField label="RAM" value={String(formState.ram)} options={(metadata?.ui_options.ram_options ?? []).map(String)} suffix="GB" disabled={!metadata} onChange={(value) => updateField("ram", Number(value))} />
                        <SelectField label="SSD" value={String(formState.ssd)} options={(metadata?.ui_options.ssd_options ?? []).map(String)} suffix="GB" disabled={!metadata} onChange={(value) => updateField("ssd", Number(value))} />
                        <SelectField label="HDD" value={String(formState.hdd)} options={(metadata?.ui_options.hdd_options ?? []).map(String)} suffix="GB" disabled={!metadata} onChange={(value) => updateField("hdd", Number(value))} />
                      </div>
                    </FieldGroup>
                  </section>
                ) : null}

                {activeStep === 3 ? (
                  <section className="space-y-8">
                    <FieldGroup title="Display and review" subtitle="Finalize the screen details and check the summary before predicting.">
                      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        <SelectField label="Resolution" value={formState.screen_resolution} options={metadata?.ui_options.screen_resolutions ?? []} disabled={!metadata} onChange={(value) => updateField("screen_resolution", value)} />
                        <NumberField label="Screen size" value={formState.screen_size} min={10} max={20} step={0.1} suffix="in" onChange={(value) => updateField("screen_size", value)} />
                        <NumberField label="Weight" value={formState.weight} min={0.5} max={6} step={0.1} suffix="kg" onChange={(value) => updateField("weight", value)} />
                        <div className="sm:col-span-2 xl:col-span-3 grid gap-3 sm:grid-cols-2">
                          <SwitchField label="Touchscreen" description="For touch-enabled laptops and convertibles." checked={formState.touchscreen} onChange={(checked) => updateField("touchscreen", checked)} />
                          <SwitchField label="IPS display" description="Improved viewing angles and color quality." checked={formState.ips} onChange={(checked) => updateField("ips", checked)} />
                        </div>
                      </div>
                    </FieldGroup>

                    <FieldGroup title="Review" subtitle="A quick summary of the configuration being priced.">
                      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                        <ReviewChip label="Brand" value={formState.company} />
                        <ReviewChip label="Type" value={formState.type_name} />
                        <ReviewChip label="CPU" value={formState.cpu_brand} />
                        <ReviewChip label="GPU" value={formState.gpu_brand} />
                        <ReviewChip label="RAM" value={`${formState.ram} GB`} />
                        <ReviewChip label="Storage" value={`${formState.ssd} GB SSD + ${formState.hdd} GB HDD`} />
                        <ReviewChip label="Display" value={`${formState.screen_size} in • ${formState.screen_resolution}`} />
                        <ReviewChip label="Weight" value={`${formState.weight} kg`} />
                        <ReviewChip label="Features" value={`${formState.touchscreen ? "Touch" : "No touch"} • ${formState.ips ? "IPS" : "Standard"}`} />
                      </div>
                    </FieldGroup>
                  </section>
                ) : null}

                <div className="flex flex-col gap-4 border-t border-slate-900/8 pt-6 lg:flex-row lg:items-center lg:justify-between">
                  <p className="max-w-xl text-sm leading-6 text-slate-600">
                    Predictions are sent through the server-side proxy, so the API key never goes into the browser.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {activeStep > 0 ? (
                      <button type="button" onClick={previousStep} className="secondary-button inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold text-slate-700 transition">
                        Back
                      </button>
                    ) : null}
                    {activeStep < stepLabels.length - 1 ? (
                      <button type="button" onClick={nextStep} className="secondary-button inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold text-slate-700 transition">
                        Continue
                      </button>
                    ) : (
                      <button type="submit" disabled={!metadata || isLoadingMetadata || isSubmitting} className="primary-button inline-flex min-w-48 items-center justify-center rounded-full px-6 py-3 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-60">
                        {isSubmitting ? "Estimating price..." : "Estimate price"}
                      </button>
                    )}
                  </div>
                </div>
              </form>
            </section>

            <aside className="panel panel-strong p-5 sm:p-6 lg:sticky lg:top-6 lg:self-start">
              <p className="text-sm font-semibold uppercase tracking-[0.22em] text-sky-700">Prediction result</p>
              <div className="mt-6 space-y-6">
                <div>
                  <p className="text-sm text-slate-500">Estimated price</p>
                  <p className="mt-2 text-5xl font-semibold tracking-[-0.05em] text-slate-950 sm:text-6xl">
                    {prediction ? toCurrency(prediction.predicted_price_inr) : "INR --"}
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <ResultCard label="Latency" value={prediction ? `${prediction.latency_ms} ms` : "Waiting"} />
                  <ResultCard label="Model" value={prediction?.model_name ?? metadata?.model_name ?? "Loading"} />
                </div>

                <div className="rounded-[1.5rem] border border-slate-900/8 bg-white/80 p-4 text-sm leading-6 text-slate-600 shadow-[inset_0_1px_0_rgba(255,255,255,0.8)]">
                  {prediction
                    ? "Use this estimate as a starting point before checking current listings or resale offers."
                    : "Finish the guided steps and submit to generate a laptop price estimate."}
                </div>

                <div className="grid gap-3 sm:grid-cols-3">
                  <MiniStat label="Confidence" value={confidenceLabel} />
                  <MiniStat label="Accuracy" value={metadata ? metadata.metrics.r2.toFixed(3) : "..."} />
                  <MiniStat label="Version" value={metadata ? metadata.model_version.slice(-8) : "..."} />
                </div>
              </div>
            </aside>
          </div>
        </section>
      </div>
    </main>
  );
}

function LogoMark() {
  return (
    <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,#0f172a,#2563eb)] shadow-[0_14px_32px_rgba(37,99,235,0.22)]">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
        <rect x="3.5" y="5" width="21" height="14" rx="3" stroke="white" strokeWidth="2" />
        <path d="M10 22.5H18" stroke="white" strokeWidth="2" strokeLinecap="round" />
        <path d="M8 15.5L12 11.5L15 13.8L20 9.5" stroke="#93C5FD" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function FieldGroup({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-4">
        <h2 className="text-lg font-semibold tracking-[-0.02em] text-slate-950">{title}</h2>
        <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
      </div>
      {children}
    </section>
  );
}

function SelectField({ label, value, options, onChange, disabled, suffix }: { label: string; value: string; options: string[]; onChange: (value: string) => void; disabled?: boolean; suffix?: string }) {
  return (
    <label className="block space-y-2">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <div className="input-shell">
        <select className="input-base" value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
          {options.map((option) => (
            <option key={option} value={option}>
              {suffix ? `${option} ${suffix}` : option}
            </option>
          ))}
        </select>
        <span className="pointer-events-none text-slate-400">▾</span>
      </div>
    </label>
  );
}

function NumberField({ label, value, min, max, step, onChange, suffix }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void; suffix?: string }) {
  return (
    <label className="block space-y-2">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <div className="input-shell">
        <input className="input-base" type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
        {suffix ? <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{suffix}</span> : null}
      </div>
    </label>
  );
}

function SwitchField({ label, description, checked, onChange }: { label: string; description: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <button type="button" onClick={() => onChange(!checked)} className="switch-card" role="switch" aria-checked={checked}>
      <span className="text-left">
        <span className="block text-sm font-medium text-slate-700">{label}</span>
        <span className="mt-1 block text-xs leading-5 text-slate-500">{description}</span>
      </span>
      <span className="flex items-center gap-3">
        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{checked ? "On" : "Off"}</span>
        <span className={`switch-track ${checked ? "switch-track-on" : "switch-track-off"}`}>
          <span className={`switch-thumb ${checked ? "translate-x-5" : "translate-x-0"}`} />
        </span>
      </span>
    </button>
  );
}

function ReviewChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.25rem] border border-slate-900/8 bg-white/82 px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.72)]">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-2 text-sm font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function ResultCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.35rem] border border-slate-900/8 bg-white/78 px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-2 text-sm font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.35rem] border border-slate-900/8 bg-white/72 px-4 py-4 text-center shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-2 text-sm font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function InfoBadge({ label, tone }: { label: string; tone: "neutral" | "accent" }) {
  return (
    <span className={`inline-flex items-center rounded-full px-4 py-2 text-sm font-medium ${tone === "accent" ? "bg-sky-100 text-sky-800" : "bg-slate-100 text-slate-700"}`}>
      {label}
    </span>
  );
}
