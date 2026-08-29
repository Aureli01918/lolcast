/**
 * Update-button proxy.
 *
 * The dashboard is a static page, so it cannot start a rebuild on its own.
 * Triggering GitHub Actions needs an access token, and a token pasted into
 * a public web page is a token anyone can take and use.
 *
 * This tiny worker holds the token instead. The page sends it an empty
 * POST; the worker starts the workflow. The token never leaves Cloudflare.
 *
 * Free tier is far more than enough — this runs a handful of times a day.
 *
 * ── Setup ──────────────────────────────────────────────────────────────
 * 1. Make a fine-grained GitHub token:
 *    github.com/settings/personal-access-tokens
 *    - Repository access: only your lolcast repo
 *    - Permissions: Actions -> Read and write. Nothing else.
 *    Scope it to the one repo. It is the only thing this worker can do.
 *
 * 2. Create a Cloudflare Worker (dash.cloudflare.com -> Workers -> Create),
 *    paste this file in, and deploy.
 *
 * 3. Worker Settings -> Variables, add:
 *      GH_TOKEN   (encrypted)  your token
 *      GH_REPO                 yourname/lolcast
 *      ALLOW_ORIGIN            https://yourname.github.io
 *
 * 4. Put the worker URL in config.yaml under dashboard.refresh.proxyUrl,
 *    then run predict once so it lands in data.json.
 *
 * If you would rather not set this up, leave proxyUrl blank and set
 * actionsUrl instead. The button then opens the Actions page and you press
 * "Run workflow" yourself — two taps, no token anywhere.
 */

const WORKFLOW = "update.yml";
const BRANCH = "main";

// Don't let the button be spammed into burning your Actions minutes.
const COOLDOWN_SECONDS = 120;
let lastRun = 0;

export default {
  async fetch(request, env) {
    const origin = env.ALLOW_ORIGIN || "*";
    const cors = {
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }
    if (request.method !== "POST") {
      return json({ error: "Send a POST to start a rebuild." }, 405, cors);
    }
    if (!env.GH_TOKEN || !env.GH_REPO) {
      return json({ error: "Worker is missing GH_TOKEN or GH_REPO." }, 500, cors);
    }

    const now = Date.now() / 1000;
    if (now - lastRun < COOLDOWN_SECONDS) {
      const wait = Math.ceil(COOLDOWN_SECONDS - (now - lastRun));
      return json({ error: `A rebuild just started. Try again in ${wait}s.` }, 429, cors);
    }

    const res = await fetch(
      `https://api.github.com/repos/${env.GH_REPO}/actions/workflows/${WORKFLOW}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.GH_TOKEN}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "lolcast-update-proxy",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ref: BRANCH }),
      }
    );

    if (res.status === 204) {
      lastRun = now;
      return json({ started: true }, 202, cors);
    }

    // Pass GitHub's reason through, but never the token.
    const detail = await res.text();
    return json({ error: "GitHub refused the request.", status: res.status, detail },
                502, cors);
  },
};

function json(body, status, cors) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...cors },
  });
}
