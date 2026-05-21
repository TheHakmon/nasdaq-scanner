"""Shared theme CSS + script for all auxiliary HTML pages (archive index,
performance summary, trade journal). Dark mode by default; toggle saved in
localStorage under 'nasdaq_theme'."""

THEME_HEAD = """
<script>
  (function() {
    const saved = (function(){ try { return localStorage.getItem("nasdaq_theme"); } catch(e) { return null; } })();
    const theme = saved || "dark";
    if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
  })();
</script>
<style>
  :root {
    color-scheme: dark;
    --bg: #0b1220; --bg-card: #131c2f; --bg-elevated: #1a2236;
    --text: #e2e8f0; --text-muted: #94a3b8; --text-subtle: #64748b;
    --border: #2a3450; --border-strong: #3a4664; --hover: #1e293b;
    --accent: #818cf8; --accent-text: #c7d2fe; --accent-bg: rgba(129,140,248,0.15);
    --success: #10b981; --success-text: #6ee7b7;
    --danger: #ef4444; --danger-text: #fca5a5;
    --warning: #f59e0b; --warning-text: #fcd34d;
    --table-head-bg: #1a2236;
  }
  [data-theme="light"] {
    color-scheme: light;
    --bg: #fafafa; --bg-card: #ffffff; --bg-elevated: #ffffff;
    --text: #111; --text-muted: #6b7280; --text-subtle: #9ca3af;
    --border: #e5e7eb; --border-strong: #d1d5db; --hover: #f9fafb;
    --accent: #4f46e5; --accent-text: #3730a3; --accent-bg: #eef2ff;
    --success: #059669; --success-text: #065f46;
    --danger: #dc2626; --danger-text: #991b1b;
    --warning: #d97706; --warning-text: #92400e;
    --table-head-bg: #f9fafb;
  }
  body { background: var(--bg); color: var(--text); }
  .nav-btn, .theme-toggle {
    background: var(--bg-card); color: var(--text);
    border: 1px solid var(--border);
  }
  .nav-btn:hover, .theme-toggle:hover { background: var(--hover); }
  .theme-toggle {
    border-radius: 8px; padding: 8px 12px; font-size: 14px; cursor: pointer;
  }
  table { background: var(--bg-card); }
  th { background: var(--table-head-bg); color: var(--text-muted); }
  th, td { border-bottom: 1px solid var(--border); }
  tr:hover { background: var(--hover); }
  .stat, .empty, .help, .note {
    background: var(--bg-card); border: 1px solid var(--border);
    color: var(--text);
  }
  .help, .note {
    background: var(--accent-bg); border-color: var(--accent);
    color: var(--accent-text);
  }
  a { color: var(--accent); }
  code { background: var(--bg-elevated); color: var(--text); }
</style>
"""

THEME_TOGGLE_SCRIPT = """
<script>
  window.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    const updateIcon = () => {
      const isLight = document.documentElement.getAttribute("data-theme") === "light";
      btn.textContent = isLight ? "☀️" : "🌙";
    };
    updateIcon();
    btn.addEventListener("click", () => {
      const isLight = document.documentElement.getAttribute("data-theme") === "light";
      if (isLight) {
        document.documentElement.removeAttribute("data-theme");
        try { localStorage.setItem("nasdaq_theme", "dark"); } catch(e){}
      } else {
        document.documentElement.setAttribute("data-theme", "light");
        try { localStorage.setItem("nasdaq_theme", "light"); } catch(e){}
      }
      updateIcon();
    });
  });
</script>
"""

THEME_TOGGLE_BUTTON = '<button class="theme-toggle" id="theme-toggle" title="החלף בין כהה לבהיר">🌙</button>'
