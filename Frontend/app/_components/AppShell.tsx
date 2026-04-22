'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Layers, ScanLine, Moon, Sun, Menu, X } from 'lucide-react';

const NAV_ITEMS = [
  {
    href: '/surface',
    label: 'Surface Detection',
    description: 'Upload TIFFs • Visualization + U-Net / nnU-Net segmentation',
    icon: Layers,
  },
  {
    href: '/ink',
    label: 'Ink Detection',
    description: 'Pre-staged Winner fragments • TimeSformer ink prediction',
    icon: ScanLine,
  },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);

  useEffect(() => {
    const saved = window.localStorage.getItem('cv-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldDark = saved ? saved === 'dark' : prefersDark;
    const root = window.document.documentElement;
    if (shouldDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    setIsDarkMode(shouldDark);
  }, []);

  const toggleDarkMode = () => {
    const root = window.document.documentElement;
    const next = !isDarkMode;
    if (next) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    window.localStorage.setItem('cv-theme', next ? 'dark' : 'light');
    setIsDarkMode(next);
  };

  return (
    <div className="relative min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <div className="fixed inset-0 z-0 pointer-events-none opacity-40 dark:opacity-20 [background-image:linear-gradient(to_right,#0f172a10_1px,transparent_1px),linear-gradient(to_bottom,#0f172a10_1px,transparent_1px)] dark:[background-image:linear-gradient(to_right,#e2e8f010_1px,transparent_1px),linear-gradient(to_bottom,#e2e8f010_1px,transparent_1px)] [background-size:40px_40px]" />
      <div className="fixed inset-0 z-0 pointer-events-none bg-[radial-gradient(circle_at_8%_0%,#ecfeff_0%,transparent_35%),radial-gradient(circle_at_100%_12%,#cffafe_0%,transparent_35%)] dark:bg-[radial-gradient(circle_at_8%_0%,#0891b210_0%,transparent_40%),radial-gradient(circle_at_100%_12%,#06b6d410_0%,transparent_40%)]" />

      <div className="relative z-10 flex min-h-screen">
        <aside
          className={`${mobileOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-40 w-72 transform border-r border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md transition-transform duration-200 md:sticky md:top-0 md:h-screen md:translate-x-0`}
        >
          <div className="flex h-full flex-col p-5">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700 dark:text-cyan-400">Vesuvius</p>
                <h2 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">CV Pipeline</h2>
              </div>
              <button
                onClick={() => setMobileOpen(false)}
                className="md:hidden rounded-lg p-2 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                aria-label="Close menu"
              >
                <X size={18} />
              </button>
            </div>

            <nav className="flex-1 space-y-2">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.href || (item.href !== '/' && pathname?.startsWith(item.href));
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className={`block rounded-xl border px-4 py-3 transition ${
                      active
                        ? 'border-cyan-300 dark:border-cyan-700 bg-cyan-50 dark:bg-cyan-900/30 shadow-sm'
                        : 'border-transparent hover:border-slate-200 dark:hover:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/60'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <Icon
                        size={18}
                        className={active ? 'text-cyan-700 dark:text-cyan-300' : 'text-slate-600 dark:text-slate-400'}
                      />
                      <span className={`text-sm font-semibold ${active ? 'text-cyan-900 dark:text-cyan-100' : 'text-slate-900 dark:text-slate-100'}`}>
                        {item.label}
                      </span>
                    </div>
                    <p className={`mt-1 pl-7 text-xs ${active ? 'text-cyan-700/80 dark:text-cyan-300/80' : 'text-slate-500 dark:text-slate-400'}`}>
                      {item.description}
                    </p>
                  </Link>
                );
              })}
            </nav>

            <div className="mt-6 flex items-center justify-between rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 px-4 py-3">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Theme</span>
              <button
                onClick={toggleDarkMode}
                className="flex items-center gap-2 rounded-lg bg-white dark:bg-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-700 dark:text-cyan-200 shadow-sm hover:bg-slate-100 dark:hover:bg-slate-600"
                aria-label="Toggle dark mode"
              >
                {isDarkMode ? <Sun size={14} /> : <Moon size={14} />}
                {isDarkMode ? 'Light' : 'Dark'}
              </button>
            </div>
          </div>
        </aside>

        {mobileOpen && (
          <button
            aria-label="Close menu overlay"
            className="fixed inset-0 z-30 bg-black/40 md:hidden"
            onClick={() => setMobileOpen(false)}
          />
        )}

        <div className="flex-1 min-w-0">
          <div className="sticky top-0 z-20 flex items-center justify-between border-b border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70 px-4 py-3 backdrop-blur md:hidden">
            <button
              onClick={() => setMobileOpen(true)}
              className="flex items-center gap-2 rounded-lg border border-slate-200 dark:border-slate-700 px-3 py-2 text-sm font-semibold text-slate-700 dark:text-slate-200"
              aria-label="Open menu"
            >
              <Menu size={16} />
              Menu
            </button>
            <span className="text-sm font-semibold text-slate-600 dark:text-slate-300">CV Pipeline</span>
          </div>
          <main className="px-4 py-6 md:px-8 md:py-8">{children}</main>
        </div>
      </div>
    </div>
  );
}
