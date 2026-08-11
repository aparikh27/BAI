/**
 * record-demo.mjs — Captures a full voice-command run from the React dashboard.
 *
 * Drives the real dashboard at http://localhost:5173 with Playwright, records
 * the browser viewport at 1920x1080, and hands the raw capture to ffmpeg for a
 * 30 FPS MP4.
 *
 * Scenes:
 *   1. Standby      — live idle feed from the robot camera
 *   2. Voice input  — replay Demo.m4a, wait for the transcription to render
 *   3. Execution    — follow plan steps while the robot manoeuvres
 *   4. Complete     — hold on the "Task Completed" state
 *
 * Usage:  node scripts/record-demo.mjs [--url http://localhost:5173]
 */

import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { mkdir, readdir, rm, stat } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i !== -1 && args[i + 1] ? args[i + 1] : fallback;
};

// mic=0: telemetry without live microphone capture, so ambient room noise
// cannot be transcribed into a competing robot command mid-take.
const APP_URL = getArg("url", "http://localhost:5173/?mic=0");
const PROJECT_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const OUT_DIR = path.join(PROJECT_ROOT, "recordings");
const RAW_DIR = path.join(OUT_DIR, "raw");
const OUT_FILE = path.join(OUT_DIR, "robot-dashboard-demo.mp4");
const FPS = 30;
const VIEWPORT = { width: 1920, height: 1080 };

// Scene budgets (ms)
const STANDBY_HOLD = 2000;
const EXECUTION_MIN = 5000;
const EXECUTION_MAX = 10000;
const COMPLETE_HOLD = 2000;

const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];

const START_MS = Date.now();
const log = (msg) =>
  console.log(`[record ${((Date.now() - START_MS) / 1000).toFixed(1)}s] ${msg}`);

function run(cmd, cmdArgs) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, cmdArgs, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}\n${stderr.slice(-2000)}`))
    );
  });
}

async function main() {
  await rm(RAW_DIR, { recursive: true, force: true });
  await mkdir(RAW_DIR, { recursive: true });

  log(`launching chromium, recording ${VIEWPORT.width}x${VIEWPORT.height}`);
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
    recordVideo: { dir: RAW_DIR, size: VIEWPORT },
  });

  const page = await context.newPage();

  // ── Diagnostics: everything the self-correction loop needs ──────────
  page.on("console", (m) => {
    if (m.type() === "error") {
      consoleErrors.push(m.text());
      console.error(`[browser-console-error] ${m.text()}`);
    }
  });
  page.on("pageerror", (e) => {
    pageErrors.push(e.message);
    console.error(`[browser-page-error] ${e.message}`);
  });
  page.on("requestfailed", (r) => {
    const entry = `${r.method()} ${r.url()} — ${r.failure()?.errorText}`;
    // The camera feed is an endless multipart/x-mixed-replace response. It has
    // no natural end, so tearing down the page always aborts it — that is the
    // normal way this request terminates, not a failure.
    const isFeedTeardown =
      r.url().includes("/api/video-feed") && r.failure()?.errorText === "net::ERR_ABORTED";
    if (isFeedTeardown) {
      console.log(`[record] (benign) video feed closed: ${r.url()}`);
      return;
    }
    failedRequests.push(entry);
    console.error(`[network-failed] ${entry}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400) {
      const entry = `${r.status()} ${r.request().method()} ${r.url()}`;
      failedRequests.push(entry);
      console.error(`[network-error] ${entry}`);
    }
  });

  try {
    log(`opening ${APP_URL}`);
    await page.goto(APP_URL, { waitUntil: "domcontentloaded", timeout: 30_000 });
    await page.getByTestId("start-detect").waitFor({ state: "visible", timeout: 15_000 });

    // ── Setup: bring the perception + telemetry pipeline online ───────
    log("starting detection");
    await page.getByTestId("start-detect").click();

    log("waiting for telemetry to go green");
    await page
      .locator('[data-testid="telemetry-indicator"][data-connected="true"]')
      .waitFor({ state: "visible", timeout: 60_000 });

    log("waiting for the video feed to deliver frames");
    await page.getByTestId("video-feed").waitFor({ state: "visible", timeout: 30_000 });
    // A visible <img> is not a decoded frame — wait for real pixels.
    await page.waitForFunction(
      () => {
        const img = document.querySelector('[data-testid="video-feed"]');
        return !!img && img.naturalWidth > 0 && img.naturalHeight > 0;
      },
      undefined,
      { timeout: 60_000 }
    );
    log("video stream live");

    // ── Scene 1: standby ──────────────────────────────────────────────
    log("scene 1 — standby");
    await page.waitForTimeout(STANDBY_HOLD);

    // ── Scene 2: voice input ──────────────────────────────────────────
    log("scene 2 — injecting voice command");
    await page.getByTestId("run-voice-command").click();

    await page.waitForFunction(
      () => {
        const el = document.querySelector('[data-testid="transcript"]');
        if (!el) return false;
        const t = (el.textContent || "").toLowerCase();
        // Hold until real speech replaces the placeholder / progress text.
        return (
          t.length > 0 &&
          !t.includes("waiting for voice command") &&
          !t.includes("listening for a voice command") &&
          !t.includes("transcribing voice command")
        );
      },
      undefined,
      { timeout: 180_000 }
    );
    const transcript = (await page.getByTestId("transcript").textContent())?.trim();
    log(`transcript rendered: "${transcript}"`);
    await page.waitForTimeout(1500);

    // ── Scene 3: reasoning + execution ────────────────────────────────
    log("scene 3 — reasoning and execution");
    // Surface step transitions so a stalled run is diagnosable from the log
    // rather than only as a timeout at the end.
    let lastProgress = "";
    const progressTimer = setInterval(async () => {
      try {
        const now = await page.evaluate(() => {
          const step = document.querySelector('[data-testid="step-progress"]');
          const state = document.querySelector('[data-testid="task-state"]');
          return `${state?.getAttribute("data-state") ?? "?"} | ${step?.textContent ?? ""}`;
        });
        if (now !== lastProgress) {
          lastProgress = now;
          log(`  progress: ${now}`);
        }
      } catch {
        /* page may be closing */
      }
    }, 500);

    const execStart = Date.now();
    const terminal = page.locator(
      '[data-testid="task-state"][data-state="completed"], [data-testid="task-state"][data-state="failed"]'
    );

    // Let the agent work; keep filming for at least EXECUTION_MIN so the
    // scene has substance even if the plan resolves instantly.
    while (Date.now() - execStart < EXECUTION_MAX) {
      const settled = (await terminal.count()) > 0;
      if (settled && Date.now() - execStart >= EXECUTION_MIN) break;
      await page.waitForTimeout(250);
    }

    // ── Scene 4: completion ───────────────────────────────────────────
    log("scene 4 — waiting for terminal task state");
    try {
      await terminal.first().waitFor({ state: "visible", timeout: 300_000 });
    } finally {
      clearInterval(progressTimer);
    }
    const finalState = await terminal.first().getAttribute("data-state");
    const finalMessage = (await page.getByTestId("task-message").textContent().catch(() => ""))?.trim();
    log(`final task state: ${finalState}${finalMessage ? ` — ${finalMessage}` : ""}`);

    await page.waitForTimeout(COMPLETE_HOLD);

    if (finalState !== "completed") {
      throw new Error(`Run ended in state "${finalState}": ${finalMessage || "no detail"}`);
    }
  } finally {
    // Video is only flushed to disk on context close.
    await context.close();
    await browser.close();
  }

  // ── Render ──────────────────────────────────────────────────────────
  const raw = (await readdir(RAW_DIR)).filter((f) => f.endsWith(".webm"));
  if (raw.length === 0) throw new Error("Playwright produced no video file");
  const rawPath = path.join(RAW_DIR, raw[0]);
  log(`encoding ${rawPath} -> ${OUT_FILE}`);

  await run("ffmpeg", [
    "-y",
    "-i", rawPath,
    "-r", String(FPS),
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "20",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    OUT_FILE,
  ]);

  const info = await stat(OUT_FILE);
  if (info.size === 0) throw new Error("ffmpeg produced an empty file");
  log(`wrote ${OUT_FILE} (${(info.size / 1_048_576).toFixed(2)} MB)`);

  const problems = [...pageErrors, ...consoleErrors, ...failedRequests];
  if (problems.length > 0) {
    console.error(`\n[record] ${problems.length} browser/network problem(s) recorded:`);
    problems.forEach((p) => console.error(`  - ${p}`));
    process.exitCode = 2;
  } else {
    log("clean run — no console, page, or network errors");
  }
}

main().catch((err) => {
  console.error(`[record] FAILED: ${err.message}`);
  process.exit(1);
});
