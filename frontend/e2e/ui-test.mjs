import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:3111";
const errors = [];

const browser = await chromium.launch();
const page = await browser.newPage();

page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
});
page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
page.on("crash", () => errors.push("PAGE CRASHED"));

const step = async (name, fn) => {
  try {
    await fn();
    console.log(`  PASS  ${name}`);
  } catch (e) {
    console.log(`  FAIL  ${name}\n        ${e.message.split("\n")[0]}`);
    process.exitCode = 1;
  }
};

console.log("--- load ---");
await page.goto(BASE, { waitUntil: "networkidle" });

await step("centered heading reads 'Research Assistant'", async () => {
  await page
    .getByRole("heading", { name: "Research Assistant" })
    .waitFor({ timeout: 10000 });
});

await step("sidebar thread list is present", async () => {
  await page
    .locator('[data-slot="aui_threadlist-sidebar"]')
    .waitFor({ timeout: 5000 });
  await page
    .locator('[data-slot="aui_thread-list-root"]')
    .waitFor({ timeout: 5000 });
});

await step("New Thread button is present", async () => {
  await page
    .locator('[data-slot="aui_thread-list-new"]')
    .waitFor({ timeout: 5000 });
});

await step("model pill shows Llama 3 70B Instruct", async () => {
  await page
    .getByText("Llama 3 70B Instruct")
    .first()
    .waitFor({ timeout: 5000 });
});

console.log("--- send a query (thread 1) ---");
const composer = page.getByRole("textbox", { name: "Message input" });
await composer.fill("what do these papers say about data poisoning?");
await composer.press("Enter");

await step("user message appears", async () => {
  await page.locator('[data-role="user"]').first().waitFor({ timeout: 15000 });
});

await step("assistant streams text back", async () => {
  await page.waitForFunction(
    () => {
      const el = document.querySelector(
        '[data-slot="aui_assistant-message-content"]',
      );
      return el && el.innerText.trim().length > 40;
    },
    { timeout: 120000 },
  );
});

await step("source citation chips render", async () => {
  await page
    .locator(".aui-message-sources a")
    .first()
    .waitFor({ timeout: 120000 });
});

// The render-loop bug that crashed Firefox would show up here: a looping tree
// pegs the CPU and the page stops responding to a trivial evaluate.
await step(
  "page still responsive after sources render (no render loop)",
  async () => {
    for (let i = 0; i < 5; i++) {
      await page.evaluate(() => performance.now());
      await page.waitForTimeout(300);
    }
    const chips = await page.locator(".aui-message-sources a").count();
    if (chips < 1) throw new Error("expected at least one source chip");
  },
);

const firstAnswer = await page
  .locator('[data-slot="aui_assistant-message-content"]')
  .first()
  .innerText();

console.log("--- thread list ---");
await step("thread appears in sidebar titled from the question", async () => {
  await page
    .locator('[data-slot="aui_thread-list-item-title"]')
    .filter({ hasText: /data poisoning/i })
    .first()
    .waitFor({ timeout: 20000 });
});

await step("New Thread creates an empty thread", async () => {
  await page.locator('[data-slot="aui_thread-list-new"]').click();
  await page
    .getByRole("heading", { name: "Research Assistant" })
    .waitFor({ timeout: 10000 });
  const msgs = await page.locator('[data-role="user"]').count();
  if (msgs !== 0)
    throw new Error(`expected empty thread, found ${msgs} user messages`);
});

await step("second thread accepts its own query", async () => {
  const c2 = page.getByRole("textbox", { name: "Message input" });
  await c2.fill("what is hybrid search?");
  await c2.press("Enter");
  await page.waitForFunction(
    () => {
      const el = document.querySelector(
        '[data-slot="aui_assistant-message-content"]',
      );
      return el && el.innerText.trim().length > 40;
    },
    { timeout: 120000 },
  );
});

await step("switching back to thread 1 restores its messages", async () => {
  await page
    .locator('[data-slot="aui_thread-list-item-title"]')
    .filter({ hasText: /data poisoning/i })
    .first()
    .click();

  // Wait for the switch to actually land rather than guessing at a delay.
  await page
    .locator('[data-role="user"]')
    .filter({ hasText: /data poisoning/i })
    .first()
    .waitFor({ timeout: 20000 });

  const restored = await page
    .locator('[data-slot="aui_assistant-message-content"]')
    .first()
    .innerText();
  if (!restored.trim().startsWith(firstAnswer.trim().slice(0, 60))) {
    throw new Error(
      `thread 1 answer not restored.\n  expected prefix: ${firstAnswer.trim().slice(0, 60)}\n  got:             ${restored.trim().slice(0, 60)}`,
    );
  }
});

await step("two threads listed in the sidebar", async () => {
  const count = await page
    .locator('[data-slot="aui_thread-list-item"]')
    .count();
  if (count < 2) throw new Error(`expected >= 2 threads, found ${count}`);
});

await page.screenshot({ path: "e2e/last-run.png", fullPage: false });

console.log("--- browser errors ---");
if (errors.length === 0) console.log("  none");
else {
  for (const e of errors.slice(0, 10)) console.log("  " + e);
  process.exitCode = 1;
}

await browser.close();
