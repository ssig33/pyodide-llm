// ci.mjs (T142): start workflows and wait for their runs, or wait for runs, always by the runs' IDs and with a
// deadline, printing a line whenever a run or one of its jobs changes state, and at the end one table of them all.
// Never by searching the latest runs (a new run pushes the one waited for out of the list) and never by pgrep -f or
// pkill -f (they match the loop that runs them: a wait of the 2026-09-26 review round went on 52 minutes after what
// it waited for had ended).
//
//   node tests/ci.mjs run <workflow file> [input=value ...] [+ <workflow file> [input=value ...] ...]
//   node tests/ci.mjs wait <run id> [<run id> ...]
//   node tests/ci.mjs nightly                      T193: the latest nightly (scheduled) runs of the full suites,
//                                                  tests.yml and gpu-prompt.yml, waited for or read by their IDs
//   node tests/ci.mjs deploy [--sha <commit>]      the deploy of a commit (HEAD); if a newer push cancelled it while
//                                                  it waited for its turn, the newer deploy (which has the commit too)
//   options: [--minutes 60 (deploy: 20)] [--grep <regex> | --grep none] [--ref main]
//
//   node tests/ci.mjs run models.yml models=hf-qwen3-0.6b browser=webkit + models.yml models=tiny-lm os=windows-latest
//
// Meant to run in the background (the agent's Bash with run_in_background): it always ends, by the deadline at the
// latest. Once a run has ended, the lines of its jobs' logs that match --grep are printed (models.yml, bench.yml,
// browsers.yml, preview.yml and gpu-prompt.yml have lines of their own by default), and for a job that failed, the
// lines up to its first error. Everything printed also goes to .tmp/ci/<time>-<pid>.log, and a run already waited for
// by another ci.mjs is not waited for twice: that one's log is named instead.
// Exit: 0 every run succeeded, 1 one ended otherwise (failure, cancelled...), 2 the deadline passed with one still
// going, 3 there was no run to wait for (a dispatch failed, no deploy of that commit appeared, no run of that ID), 4
// all were waited for by another ci.mjs already.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clock = () => new Date().toISOString().slice(11, 19);

// ---- what is printed, also kept in .tmp/ci/ (in the repository, not in $HOME)
const DIR = path.join(new URL("..", import.meta.url).pathname, ".tmp", "ci");
fs.mkdirSync(DIR, { recursive: true });
const logPath = path.join(DIR, `${new Date().toISOString().replace(/[:.]/g, "-")}-${process.pid}.log`);
const out = (line) => {
  console.log(line);
  fs.appendFileSync(logPath, line + "\n");
};
const say = (line) => out(`${clock()} ${line}`);

// ---- one waiter a run: a lock file with the waiter's pid and log
const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};
function claim(id) {
  const lock = path.join(DIR, `${id}.lock`);
  try {
    const held = JSON.parse(fs.readFileSync(lock, "utf8"));
    if (held.pid !== process.pid && alive(held.pid)) return held;
  } catch {
    // no lock, or a broken one: take it
  }
  fs.writeFileSync(lock, JSON.stringify({ pid: process.pid, log: logPath }));
  return null;
}
function release(id) {
  const lock = path.join(DIR, `${id}.lock`);
  try {
    if (JSON.parse(fs.readFileSync(lock, "utf8")).pid === process.pid) fs.unlinkSync(lock);
  } catch {
    // gone already
  }
}

// ---- gh, tried again on a failure of the network or of the API, and waiting out a rate limit: a single failed
// call must not end a long wait. What cannot get better (a wrong path, a refused dispatch) ends it.
async function gh(args) {
  for (let attempt = 1; ; attempt++) {
    try {
      return execFileSync("gh", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], maxBuffer: 256 * 2 ** 20 });
    } catch (error) {
      const why = String(error.stderr || error.message).trim().split("\n").at(-1);
      const limited = /rate limit|HTTP 429/i.test(why);
      if (!limited && (attempt >= 5 || /HTTP 4(0[0-4]|22)/.test(why))) throw new Error(`gh ${args.slice(0, 3).join(" ")}: ${why}`);
      if (attempt >= 20) throw new Error(`gh ${args.slice(0, 3).join(" ")}: ${why}`);
      say(`(gh failed, trying again: ${why})`);
      await sleep(limited ? 60000 : 5000 * attempt);
    }
  }
}
const api = async (route, ...rest) => JSON.parse(await gh(["api", route, ...rest]));

// ---- the arguments
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  if (at < 0) return fallback;
  const [, value] = args.splice(at, 2);
  return value;
};
const [command] = args;
const minutes = Number(option("minutes", command === "deploy" ? 20 : 60));
const grep = option("grep", null);
const ref = option("ref", "main");
const sha = option("sha", null);
const deadline = Date.now() + minutes * 60000;
const rest = args.slice(1);

// the lines of a workflow's logs worth reading when nothing else is asked for
const LINES = {
  "models.yml": "ready in|^then |^again|^offline|thread reports during|FAILED|timed out",
  "bench.yml": "^### (chromium|firefox|webkit|chrome|msedge)$|^sections:|^model path:|WRONG|^failed:",
  "browsers.yml": "ready in|FAILED|timed out",
  "preview.yml": "^sections:|^model path:|FAILED|timed out",
  // T193: which suite ran, and the seconds of each of its parts (tests/suite.sh)
  "tests.yml": "^suite: |^--- .*: \\d+ s$",
  "deploy.yml": "^suite: |^--- .*: \\d+ s$",
  // T183: gpu-check's lines of each model and run, its tables by layer (against NumPy's and E16), each run's line and
  // ratio, and the seconds of each step
  "gpu-prompt.yml": "^suite: |^## |layers, .* heads|^  .*: keys and values |^  a token|^keys and values of |^\\| (layer|all|\\d)|^\\|---|^ *- |^seconds of |FAILED",
};

// T193: the workflows whose full suites run every night
const NIGHTLY = ["tests.yml", "gpu-prompt.yml"];

async function start([workflow, ...pairs]) {
  const fields = ["-f", `ref=${ref}`];
  for (const pair of pairs) {
    const at = pair.indexOf("=");
    if (at < 1) throw new Error(`an input is key=value: ${pair}`);
    fields.push("-f", `inputs[${pair.slice(0, at)}]=${pair.slice(at + 1)}`);
  }
  // the dispatch answers with the ID of the run it made, in the API's version of 2026-03-10 (gh 2.46 asks for an
  // older one by default, which answers 204 and nothing else)
  const made = await api(`repos/{owner}/{repo}/actions/workflows/${workflow}/dispatches`, "-X", "POST",
    "-H", "X-GitHub-Api-Version: 2026-03-10", ...fields);
  if (!made.workflow_run_id) throw new Error(`the dispatch gave no run: ${JSON.stringify(made)}`);
  say(`started ${[workflow, ...pairs].join(" ")}: run ${made.workflow_run_id} ${made.html_url}`);
  return made.workflow_run_id;
}

async function deployOf(commit) {
  const full = execFileSync("git", ["rev-parse", commit ?? "HEAD"], { encoding: "utf8" }).trim();
  for (;;) {
    const { workflow_runs: runs } = await api(`repos/{owner}/{repo}/actions/workflows/deploy.yml/runs?head_sha=${full}&per_page=5`);
    if (runs.length) {
      say(`the deploy of ${full.slice(0, 7)}: run ${runs[0].id} ${runs[0].html_url}`);
      return runs[0].id;
    }
    if (Date.now() > deadline) return null;
    await sleep(10000);
  }
}

const minutesSince = (from, to) => ((Date.parse(to) - Date.parse(from)) / 60000).toFixed(1);

// one look at a run and its jobs: a line for each change since the last look
async function look(target) {
  const run = await api(`repos/{owner}/{repo}/actions/runs/${target.id}`);
  target.title = run.display_title;
  target.url = run.html_url;
  target.workflow = path.basename(run.path ?? "");
  const state = run.status === "completed" ? `completed ${run.conclusion}` : run.status;
  if (state !== target.state) say(`run ${target.id} (${run.display_title}): ${state}`);
  target.state = state;
  const { jobs } = await api(`repos/{owner}/{repo}/actions/runs/${target.id}/jobs?per_page=100`);
  for (const job of jobs) {
    const jobState = job.status === "completed" ? `completed ${job.conclusion} (${minutesSince(job.started_at, job.completed_at)} min)` : job.status;
    if (target.jobs.get(job.id) !== jobState) say(`  ${target.id} job ${job.name}: ${jobState}`);
    target.jobs.set(job.id, jobState);
  }
  if (run.status === "completed") target.run = run;
  return run;
}

// a deploy cancelled while it waited for its turn: a newer push made a newer deploy, which has the commit too
async function newerDeploy(target) {
  const { workflow_runs: runs } = await api(`repos/{owner}/{repo}/actions/workflows/deploy.yml/runs?branch=main&per_page=10`);
  const newer = runs.filter((r) => r.id > target.id).sort((a, b) => a.id - b.id).at(-1);
  if (!newer) return false;
  say(`superseded by the deploy of ${newer.head_sha.slice(0, 7)}: run ${newer.id}`);
  release(target.id);
  Object.assign(target, { id: newer.id, state: "", run: null, jobs: new Map() });
  claim(newer.id);
  return true;
}

// what to read of a run that ended: the lines asked for (or the workflow's), and a failed job's lines up to its
// first error
async function report(target) {
  const { jobs } = await api(`repos/{owner}/{repo}/actions/runs/${target.id}/jobs?per_page=100`);
  const pattern = grep === "none" ? null : grep ?? LINES[target.workflow];
  for (const job of jobs) {
    const failed = job.conclusion && !["success", "skipped"].includes(job.conclusion);
    if (!pattern && !failed) continue;
    let text;
    try {
      text = await gh(["api", `repos/{owner}/{repo}/actions/jobs/${job.id}/logs`]);
    } catch (error) {
      out(`== ${target.id} ${job.name}: no log (${error.message})`);
      continue;
    }
    const lines = text.split("\n").map((line) => line.replace(/^\d{4}-\d\d-\d\dT[\d:.]+Z /, "").replace(/\r$/, ""));
    out(`== ${target.id} ${job.name}${failed ? ` (${job.conclusion})` : ""}`);
    if (pattern) {
      const matching = new RegExp(pattern);
      for (const line of lines) if (matching.test(line)) out(line);
    }
    if (failed) {
      // what the failed step printed up to its first error: from the step's start (the runner's "##[group]Run"), without
      // its echo of the script and its groups
      const error = lines.findIndex((line) => line.startsWith("##[error]"));
      const end = error < 0 ? lines.length : error + 1;
      const step = lines.slice(0, end).findLastIndex((line) => line.startsWith("##[group]Run "));
      const printed = lines.slice(Math.max(0, step), end)
        .filter((line) => line.trim() && !/^(\x1b\[36;1m|##\[(group|endgroup)\]|shell: |env:$|  [A-Z0-9_]+: )/.test(line));
      out("-- up to the first error:");
      for (const line of printed.slice(-15)) out(line);
    }
  }
}

// ---- the runs to wait for
const targets = [];
let startFailed = false;
try {
  if (command === "run" && rest.length) {
    const specs = [[]];
    for (const word of rest) word === "+" ? specs.push([]) : specs.at(-1).push(word);
    for (const spec of specs.filter((s) => s.length)) targets.push({ id: await start(spec) });
  } else if (command === "wait" && rest.length && rest.every((word) => /^\d+$/.test(word))) {
    targets.push(...rest.map((word) => ({ id: Number(word) })));
  } else if (command === "nightly") {
    // T193: the full suites run every night on main; the latest scheduled run of each, then by its ID as always
    for (const workflow of NIGHTLY) {
      const { workflow_runs: runs } = await api(`repos/{owner}/{repo}/actions/workflows/${workflow}/runs?event=schedule&per_page=1`);
      if (!runs.length) say(`no nightly run of ${workflow} yet`);
      else {
        say(`the nightly ${workflow} of ${runs[0].created_at}: run ${runs[0].id} ${runs[0].html_url}`);
        targets.push({ id: runs[0].id });
      }
    }
    if (!targets.length) process.exit(3);
  } else if (command === "deploy") {
    const id = await deployOf(sha);
    if (!id) {
      say(`no deploy of ${sha ?? "HEAD"} appeared in ${minutes} min`);
      process.exit(3);
    }
    targets.push({ id, deploy: true });
  } else {
    console.error("usage: node tests/ci.mjs run <workflow> [input=value ...] [+ ...] | wait <run id> ... | nightly | deploy [--sha <commit>]" +
      "  [--minutes N] [--grep <regex> | none] [--ref main]");
    process.exit(3);
  }
} catch (error) {
  say(`no run to wait for: ${error.message}`);
  if (!targets.length) process.exit(3);
  startFailed = true;
}
const mine = targets.filter((target) => {
  const held = claim(target.id);
  if (held) say(`run ${target.id} is being waited for by ci.mjs ${held.pid} already: ${held.log}`);
  return !held;
});
if (!mine.length) process.exit(4);
say(`waiting for ${mine.map((t) => t.id).join(", ")} until ${new Date(deadline).toISOString().slice(11, 19)} (log: ${logPath})`);

// ---- the wait: every 15 seconds for the first ten minutes, then every 30 (long runs, and the API's hourly limit)
const began = Date.now();
for (;;) {
  for (const target of mine.filter((t) => !t.run && !t.missing)) {
    target.jobs ??= new Map();
    try {
      await look(target);
      if (target.deploy && target.run?.conclusion === "cancelled" && (await newerDeploy(target))) await look(target);
    } catch (error) {
      // a run that is not there (a mistyped ID) is not a run that failed: its wait ends with exit 3. What gh could not
      // get past after its tries is said, and the wait goes on to the deadline (a crash here read as a failed run)
      if (!target.state && /HTTP 404/.test(error.message)) {
        target.missing = true;
        say(`no run ${target.id}: ${error.message}`);
      } else {
        say(`(could not look at run ${target.id}: ${error.message})`);
      }
    }
  }
  if (mine.every((t) => t.run || t.missing) || Date.now() > deadline) break;
  await sleep(Date.now() - began < 600000 ? 15000 : 30000);
}
for (const target of mine) {
  if (target.run) await report(target).catch((error) => out(`== ${target.id}: no logs (${error.message})`));
  release(target.id);
}

// ---- one table of them all
out("");
out("| run | title | result | min | url |");
out("|---|---|---|---:|---|");
for (const t of mine) {
  const result = t.run ? t.run.conclusion : t.missing ? "no such run" : `still ${t.state || "unknown"} (deadline)`;
  const took = t.run ? minutesSince(t.run.run_started_at, t.run.updated_at) : "";
  out(`| ${t.id} | ${t.title ?? ""} | ${result} | ${took} | ${t.url ?? ""} |`);
}
if (startFailed) out("(a dispatch failed: see the top)");
const going = mine.some((t) => !t.run && !t.missing);
process.exit(going ? 2 : startFailed || mine.some((t) => t.missing) ? 3 : mine.every((t) => t.run.conclusion === "success") ? 0 : 1);
