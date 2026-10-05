"use strict";

// Fragment and question text is inserted with textContent only.
// The answer arrives as server-escaped HTML (app/render.py) and is the only innerHTML use.

(function () {
  const $ = (id) => document.getElementById(id);
  const form = $("ask-form");
  const input = $("question");
  const button = $("ask-button");
  let maxChars = 500;

  function fmtInt(value) {
    return Number(value || 0).toLocaleString("ru-RU");
  }

  function fmtRub(value) {
    return "≈ " + Number(value).toLocaleString("ru-RU", { maximumFractionDigits: 2 }) + " ₽";
  }

  function setBudget(budget) {
    if (!budget) return;
    let text = "Токенов сегодня: " + fmtInt(budget.tokens_today) + " из " + fmtInt(budget.daily_token_budget);
    if (budget.rub_today !== null && budget.rub_today !== undefined) text += " (" + fmtRub(budget.rub_today) + ")";
    $("budget").textContent = text + ", сутки по UTC";
    if (budget.rate_limit_per_hour) $("rate-limit").textContent = String(budget.rate_limit_per_hour);
    if (budget.hourly_token_budget) $("hourly-limit").textContent = fmtInt(budget.hourly_token_budget);
  }

  function updateCount() {
    const n = input.value.trim().length;
    $("char-count").textContent = n + " / " + maxChars + (n > maxChars ? " — будет обрезано" : "");
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function rankText(f) {
    if (f.pinned) return "найдено по номеру статьи в вопросе";
    const parts = [];
    if (f.bm25_rank) parts.push("BM25 #" + f.bm25_rank);
    if (f.dense_rank) parts.push("по смыслу #" + f.dense_rank);
    if (f.dense_score !== null && f.dense_score !== undefined) parts.push("косинус " + Number(f.dense_score).toFixed(3));
    return parts.join(" · ") || "—";
  }

  function renderFragments(fragments) {
    const box = $("fragments");
    box.replaceChildren();
    for (const f of fragments || []) {
      const details = el("details", "fragment");
      details.id = f.anchor;
      const summary = el("summary");
      summary.append(el("span", "title", f.header), el("span", "score", rankText(f)));
      details.append(summary);
      if (f.chapter) details.append(el("p", "muted", f.chapter));
      details.append(el("p", "body", f.text));
      if (f.source_url) {
        const src = el("p", "src");
        const link = el("a", null, "Источник");
        link.href = f.source_url;
        link.rel = "noopener noreferrer";
        link.target = "_blank";
        src.append(link);
        if (f.edition_date) src.append(document.createTextNode(" · редакция: " + f.edition_date));
        details.append(src);
      }
      box.append(details);
    }
    $("fragments-block").hidden = !(fragments && fragments.length);
  }

  function showNotice(text) {
    const notice = $("notice");
    notice.textContent = text || "";
    notice.hidden = !text;
  }

  function renderAnswer(data) {
    const block = $("answer-block");
    if (!data.answer) {
      block.hidden = true;
      return;
    }
    $("answer").innerHTML = data.answer.html; // escaped on the server
    const meta = [];
    if (data.mode === "live" && data.cached) {
      meta.push("модель " + data.model);
      meta.push("ответ из кэша" + (data.cached_at ? " от " + data.cached_at.slice(0, 10) : "") + ", бесплатно");
    } else if (data.mode === "live") {
      meta.push("модель " + data.model);
      if (data.usage) {
        meta.push(fmtInt(data.usage.input_tokens) + " → " + fmtInt(data.usage.output_tokens) + " токенов");
      }
      if (data.cost_rub !== null && data.cost_rub !== undefined) meta.push(fmtRub(data.cost_rub));
    } else {
      meta.push("демо-режим без ключа");
    }
    meta.push((data.latency_ms / 1000).toFixed(1) + " с");
    $("answer-meta").textContent = meta.join(" · ");
    block.hidden = false;
  }

  async function ask(question) {
    button.disabled = true;
    button.textContent = "Ищу…";
    $("result").hidden = false;
    showNotice("");
    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      let data = {};
      try { data = await resp.json(); } catch (_) { data = {}; }
      setBudget(data.budget);
      if (!resp.ok) {
        const msg = data.message ||
          (resp.status === 422 || resp.status === 413 ? "Вопрос слишком длинный." : "Ошибка " + resp.status);
        showNotice(msg);
        $("answer-block").hidden = true;
        renderFragments(data.fragments || []);
        return;
      }
      if (data.truncated) showNotice("Вопрос обрезан до " + data.max_question_chars + " символов.");
      renderAnswer(data);
      renderFragments(data.fragments);
    } catch (err) {
      showNotice("Сервер недоступен, попробуйте позже.");
    } finally {
      button.disabled = false;
      button.textContent = "Спросить";
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const q = input.value.trim();
    if (!q) {
      showNotice("Введите вопрос.");
      $("result").hidden = false;
      return;
    }
    ask(q);
  });

  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) form.requestSubmit();
  });

  $("examples").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    input.value = chip.textContent;
    updateCount();
    form.requestSubmit();
  });

  $("answer").addEventListener("click", (event) => {
    const link = event.target.closest("a.cite");
    if (!link) return;
    event.preventDefault();
    const target = document.getElementById(link.dataset.anchor);
    if (!target) return;
    target.open = true;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1200);
  });

  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      $("mock-banner").hidden = h.mode !== "mock";
      setBudget(h.budget);
      if (h.max_question_chars) {
        maxChars = h.max_question_chars;
        $("max-chars").textContent = String(maxChars);
        updateCount();
      }
      if (h.status !== "ok") {
        const box = $("status-error");
        box.textContent = h.error || "Сервис работает не полностью.";
        box.hidden = false;
      }
    })
    .catch(() => {});

  updateCount();
})();
