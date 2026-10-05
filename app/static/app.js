"use strict";

// Question, fragment and article text is inserted with textContent only.
// The final answer arrives as server-escaped HTML (app/render.py) and is the only innerHTML use;
// while it streams, the raw text goes in with textContent.

(function () {
  const $ = (id) => document.getElementById(id);
  const form = $("ask-form");
  const input = $("question");
  const button = $("ask-button");
  let maxChars = 500;
  let articlesByNumber = {};
  let fragmentByArticle = {};
  let lastAnswer = null;

  function fmtInt(value) {
    return Number(value || 0).toLocaleString("ru-RU");
  }

  function fmtRub(value) {
    return "≈ " + Number(value).toLocaleString("ru-RU", { maximumFractionDigits: 2 }) + " ₽";
  }

  function plural(n, one, few, many) {
    const m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
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

  // ---- progress steps -------------------------------------------------------

  const STEP_ORDER = ["search", "read", "write", "check"];

  function setStep(name, label) {
    const steps = $("steps");
    steps.hidden = false;
    const at = STEP_ORDER.indexOf(name);
    for (const li of steps.children) {
      const i = STEP_ORDER.indexOf(li.dataset.step);
      li.className = i < at ? "done" : i === at ? "active" : "";
    }
    if (label) steps.querySelector('[data-step="' + name + '"]').textContent = label;
  }

  function finishSteps() {
    for (const li of $("steps").children) li.className = "done";
  }

  function resetSteps() {
    const labels = { search: "Поиск по кодексу", read: "Модель читает статьи", write: "Пишет ответ",
                     check: "Код проверяет ссылки и числа" };
    for (const li of $("steps").children) {
      li.className = "";
      li.textContent = labels[li.dataset.step];
    }
  }

  // ---- fragments and articles -------------------------------------------------

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
    fragmentByArticle = {};
    for (const f of fragments || []) {
      if (!fragmentByArticle[f.article]) fragmentByArticle[f.article] = f;
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

  function shortTitle(a) {
    return (a.header || "").replace(/^Статья\s+[\d.\-]+\.\s*/, "");
  }

  function citeLink(a) {
    const link = el("a", "cite", "ст. " + a.article);
    link.dataset.article = a.article;
    if (a.anchor) link.dataset.anchor = a.anchor;
    link.href = a.source_url || (a.anchor ? "#" + a.anchor : "#");
    if (a.source_url) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
    return link;
  }

  function renderArticles(articles, cited) {
    const list = $("articles");
    list.replaceChildren();
    articlesByNumber = {};
    const citedSet = new Set(cited || []);
    for (const a of articles || []) {
      articlesByNumber[a.article] = a;
      const li = el("li", citedSet.has(a.article) ? "art cited" : "art");
      li.append(citeLink(a), el("span", "art-title", shortTitle(a)), el("span", "muted", a.coverage));
      if (citedSet.has(a.article)) li.append(el("span", "badge", "в ответе"));
      list.append(li);
    }
    $("articles-block").hidden = !(articles && articles.length);
  }

  // ---- answer ---------------------------------------------------------------

  function showNotice(text) {
    const notice = $("notice");
    notice.textContent = text || "";
    notice.hidden = !text;
  }

  function checksText(answer) {
    const c = answer.checks || {};
    if (answer.no_answer || !c.citations_total) return "";
    const parts = [];
    if (c.removed && c.removed.length) {
      parts.push("убрано " + c.removed.length + " " + plural(c.removed.length, "ссылка", "ссылки", "ссылок") +
        " на статьи, которых модель не читала (ст. " + c.removed.join(", ст. ") + ")");
    } else {
      parts.push(c.citations_total + " " + plural(c.citations_total, "ссылка", "ссылки", "ссылок") +
        " — все на прочитанные статьи");
    }
    if (c.numbers_total) {
      if (c.numbers_missing && c.numbers_missing.length) {
        parts.push("чисел найдено в тексте статей " + c.numbers_found + " из " + c.numbers_total +
          ", не найдены подчёркнуты: " + c.numbers_missing.join(", "));
      } else {
        parts.push(c.numbers_total === 1 ? "число есть в тексте статьи"
          : "все " + c.numbers_total + " " + plural(c.numbers_total, "число", "числа", "чисел") + " есть в тексте статей");
      }
    }
    return "Проверено кодом: " + parts.join(" · ");
  }

  function renderAnswer(data) {
    const block = $("answer-block");
    if (!data.answer) {
      block.hidden = true;
      return;
    }
    lastAnswer = data.answer;
    const box = $("answer");
    box.classList.remove("streaming");
    box.innerHTML = data.answer.html; // escaped on the server
    const checks = checksText(data.answer);
    $("answer-checks").textContent = checks;
    $("answer-checks").hidden = !checks;
    $("copy-button").hidden = data.answer.no_answer;
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

  function startStreamingAnswer() {
    const box = $("answer");
    box.classList.add("streaming");
    box.textContent = "";
    $("answer-checks").hidden = true;
    $("copy-button").hidden = true;
    $("answer-meta").textContent = "";
    $("answer-block").hidden = false;
  }

  function onMeta(meta) {
    renderArticles(meta.articles, []);
    renderFragments(meta.fragments);
    const n = (meta.articles || []).length;
    if (meta.cached) {
      setStep("read", "Статей в ответе: " + n);
      setStep("write", "Ответ из кэша, токены не тратятся");
      setStep("check", "Проверки сохранены с первого ответа");
    } else setStep("read", n ? "Модель читает " + n + " " + plural(n, "статью", "статьи", "статей") : "Статей не найдено");
  }

  function onDone(data) {
    finishSteps();
    if (data.truncated) showNotice("Вопрос обрезан до " + data.max_question_chars + " символов.");
    setBudget(data.budget);
    renderAnswer(data);
    renderArticles(data.articles, (data.answer && data.answer.citations || []).map((c) => c.article));
    renderFragments(data.fragments);
  }

  function onError(data, status) {
    $("steps").hidden = true;
    setBudget(data.budget);
    const msg = data.message ||
      (status === 422 || status === 413 ? "Вопрос слишком длинный." : "Ошибка " + status);
    showNotice(msg);
    $("answer-block").hidden = true;
    renderArticles(data.articles || [], []);
    renderFragments(data.fragments || []);
  }

  // ---- requests -------------------------------------------------------------

  async function readEvents(resp, handlers) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        let name = "message", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) name = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (handlers[name]) handlers[name](JSON.parse(data));
      }
    }
  }

  async function askStream(question) {
    const resp = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    if (!resp.ok || !resp.body || !resp.body.getReader) {
      let data = {};
      try { data = await resp.json(); } catch (_) { data = {}; }
      if (resp.ok) return askJson(question); // no stream support: ask again as JSON (free from the cache)
      onError(data, resp.status);
      return;
    }
    let writing = false;
    let finished = false;
    await readEvents(resp, {
      meta: onMeta,
      delta: (d) => {
        if (!writing) {
          writing = true;
          setStep("write", "Пишет ответ");
          startStreamingAnswer();
        }
        $("answer").textContent += d.t;
      },
      done: (data) => { finished = true; setStep("check"); onDone(data); },
      error: (data) => { finished = true; onError(data, data.status); },
    });
    if (!finished) showNotice("Соединение оборвалось, попробуйте ещё раз.");
  }

  async function askJson(question) {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });
    let data = {};
    try { data = await resp.json(); } catch (_) { data = {}; }
    if (!resp.ok) onError(data, resp.status);
    else onDone(data);
  }

  async function ask(question) {
    button.disabled = true;
    button.textContent = "Думаю…";
    closeCard();
    $("result").hidden = false;
    showNotice("");
    resetSteps();
    setStep("search");
    // on a phone the examples push the result below the fold: bring the progress into view
    if (!window.matchMedia("(min-width: 640px)").matches) $("steps").scrollIntoView({ behavior: "smooth", block: "start" });
    $("answer-block").hidden = true;
    $("articles-block").hidden = true;
    $("fragments-block").hidden = true;
    try {
      if (window.TextDecoder && window.ReadableStream) await askStream(question);
      else await askJson(question);
    } catch (err) {
      $("steps").hidden = true;
      showNotice("Сервер недоступен, попробуйте позже.");
    } finally {
      button.disabled = false;
      button.textContent = "Спросить";
    }
  }

  // ---- citation card ----------------------------------------------------------

  const card = $("cite-card");

  function closeCard() {
    card.hidden = true;
  }

  function openCard(link) {
    const a = articlesByNumber[link.dataset.article];
    if (!a) return false;
    const frag = fragmentByArticle[a.article];
    $("cite-title").textContent = a.header;
    $("cite-chapter").textContent = a.chapter || "";
    let quote = frag ? frag.text : "";
    // a fragment of a long article can start mid-sentence: start from its next paragraph
    const nl = quote.indexOf("\n");
    if (/^[a-zа-яё]/.test(quote) && nl > 0 && nl < quote.length - 40) quote = "… " + quote.slice(nl + 1);
    if (quote.length > 420) quote = quote.slice(0, 420).replace(/\s+\S*$/, "") + " …";
    $("cite-quote").textContent = quote;
    $("cite-quote").hidden = !quote;
    $("cite-coverage").textContent = "Модель прочитала: " + a.coverage +
      (a.edition_date ? " · редакция " + a.edition_date : "");
    const source = $("cite-source");
    source.hidden = !a.source_url;
    if (a.source_url) source.href = a.source_url;
    const toFragment = $("cite-fragment");
    toFragment.hidden = !(frag && frag.anchor);
    if (frag) toFragment.dataset.anchor = frag.anchor;
    card.hidden = false;
    if (window.matchMedia("(min-width: 640px)").matches) {
      const r = link.getBoundingClientRect();
      const width = card.offsetWidth;
      const left = Math.min(Math.max(8, r.left + window.scrollX), window.scrollX + document.documentElement.clientWidth - width - 8);
      card.style.left = left + "px";
      card.style.top = (r.bottom + window.scrollY + 6) + "px";
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } else {
      card.style.left = "";
      card.style.top = "";
    }
    source.focus({ preventScroll: true });
    return true;
  }

  function onCiteClick(event) {
    const link = event.target.closest("a.cite");
    if (!link) return;
    if (openCard(link)) event.preventDefault();
  }

  $("answer").addEventListener("click", onCiteClick);
  $("articles").addEventListener("click", onCiteClick);
  $("cite-close").addEventListener("click", closeCard);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeCard(); });
  document.addEventListener("click", (event) => {
    if (!card.hidden && !card.contains(event.target) && !event.target.closest("a.cite")) closeCard();
  });
  $("cite-fragment").addEventListener("click", (event) => {
    event.preventDefault();
    closeCard();
    const target = document.getElementById(event.currentTarget.dataset.anchor);
    if (!target) return;
    target.open = true;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1200);
  });

  // ---- copy -----------------------------------------------------------------

  $("copy-button").addEventListener("click", async () => {
    if (!lastAnswer) return;
    const lines = [lastAnswer.text];
    const cited = (lastAnswer.citations || []).filter((c) => c.found && c.source_url);
    if (cited.length) {
      lines.push("", "Статьи:");
      for (const c of cited) lines.push("ст. " + c.article + " ТК РФ — " + c.source_url);
    }
    const btn = $("copy-button");
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      btn.textContent = "Скопировано";
    } catch (_) {
      btn.textContent = "Не удалось";
    }
    setTimeout(() => { btn.textContent = "Скопировать"; }, 1500);
  });

  // ---- form -----------------------------------------------------------------

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
