/* Interactive quiz: multiple choice and flashcards. Safe to load more than once. */
(function () {
  if (document.documentElement.dataset.folioQuiz === "ready") return;
  document.documentElement.dataset.folioQuiz = "ready";

  function boot() {
    var roots = document.querySelectorAll(".quiz");
    for (var i = 0; i < roots.length; i++) {
      try {
        mount(roots[i]);
      } catch (err) {
        /* Leave the static fallback in place. */
      }
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  function mount(root) {
    if (root.getAttribute("data-quiz-ready") === "true") return;
    var dataEl = root.querySelector(".quiz-data");
    var app = root.querySelector(".quiz-app");
    if (!dataEl || !app) return;

    var data = JSON.parse(dataEl.textContent);
    var questions = data.questions || [];
    if (!questions.length) return;

    root.setAttribute("data-quiz-ready", "true");

    var viaKeyboard = false;
    var state = {
      mode: readMode(data.id) === "cards" ? "cards" : "mcq",
      deck: [],
      order: [],
      options: {},
      mcqIndex: 0,
      answers: {},
      phase: "question",
      queue: [],
      cardIndex: 0,
      flipped: false,
      done: 0,
      focusSelector: "",
    };

    shuffleDeck(false);
    buildShell();
    renderBody();
    root.classList.add("is-enhanced");

    root.addEventListener("keydown", onKeydown);
    root.addEventListener("pointerdown", function (event) {
      viaKeyboard = false;
      if (event.target.closest("button, a, input, textarea, select, summary")) return;
      if (!root.hasAttribute("tabindex")) root.setAttribute("tabindex", "-1");
      root.focus({ preventScroll: true });
    });

    function onKeydown(event) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.target.closest("input, textarea, select")) return;
      viaKeyboard = true;
      if (state.mode === "mcq") onMcqKey(event);
      else onCardKey(event);
    }

    function onMcqKey(event) {
      if (state.phase !== "question") return;
      var q = currentQuestion();
      if (!q) return;
      var answered = hasAnswer(q);
      if (event.key >= "1" && event.key <= "9") {
        var n = Number(event.key);
        var opts = state.options[keyOf(q)] || [];
        if (answered || n < 1 || n > opts.length) return;
        event.preventDefault();
        choose(q, opts[n - 1]);
        return;
      }
      if (event.key === "Enter" && answered) {
        if (event.target.closest(".quiz-next")) return;
        event.preventDefault();
        nextQuestion();
      }
    }

    function onCardKey(event) {
      if (!state.queue.length) return;
      if (event.key === " " || event.key === "Spacebar") {
        if (event.target.closest(".quiz-again, .quiz-gotit, .quiz-prev, .quiz-card-next, .quiz-mode, .quiz-shuffle, .quiz-download, .quiz-restart")) {
          return;
        }
        event.preventDefault();
        flip();
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        moveCard(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        moveCard(1);
      }
    }

    function buildShell() {
      var header = el("header", "quiz-header");
      header.appendChild(el("h2", "quiz-title", data.title || "Quiz"));
      if (data.description) header.appendChild(el("p", "quiz-description", data.description));

      var toolbar = el("div", "quiz-toolbar");
      var toggle = el("div", "quiz-toggle");
      toggle.setAttribute("role", "group");
      toggle.setAttribute("aria-label", "Quiz mode");
      toggle.appendChild(modeButton("mcq", "Multiple choice"));
      toggle.appendChild(modeButton("cards", "Flashcards"));
      toolbar.appendChild(toggle);

      var shuffleBtn = el("button", "quiz-shuffle", "Shuffle");
      shuffleBtn.type = "button";
      shuffleBtn.addEventListener("click", function () {
        viaKeyboard = false;
        shuffleDeck(true);
        announce("Questions shuffled.");
        renderBody();
      });
      toolbar.appendChild(shuffleBtn);
      header.appendChild(toolbar);
      app.appendChild(header);

      var live = el("div", "quiz-live");
      live.setAttribute("aria-live", "polite");
      live.setAttribute("aria-atomic", "true");
      app.appendChild(live);
      app.appendChild(el("div", "quiz-body"));
    }

    function modeButton(mode, label) {
      var button = el("button", "quiz-mode", label);
      button.type = "button";
      button.setAttribute("data-mode", mode);
      button.setAttribute("aria-pressed", state.mode === mode ? "true" : "false");
      button.addEventListener("click", function () {
        if (state.mode === mode) return;
        state.mode = mode;
        writeMode(data.id, mode);
        var buttons = root.querySelectorAll(".quiz-mode");
        for (var i = 0; i < buttons.length; i++) {
          buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-mode") === mode ? "true" : "false");
        }
        announce(mode === "cards" ? "Flashcards." : "Multiple choice.");
        renderBody();
      });
      return button;
    }

    function renderBody() {
      var body = root.querySelector(".quiz-body");
      var keepFocus = viaKeyboard && root.contains(document.activeElement);
      clear(body);
      if (state.mode === "cards") renderCards(body);
      else if (state.phase === "results") renderResults(body);
      else renderQuestion(body);
      if (keepFocus && state.focusSelector) {
        var target = body.querySelector(state.focusSelector);
        if (target) target.focus({ preventScroll: true });
      }
      state.focusSelector = "";
    }

    function renderQuestion(body) {
      var q = currentQuestion();
      if (!q) return;
      var total = state.order.length;
      var position = state.mcqIndex + 1;
      body.appendChild(el("p", "quiz-progress-label", "Question " + position + " of " + total));

      var progress = el("div", "quiz-progress");
      progress.setAttribute("role", "progressbar");
      progress.setAttribute("aria-valuemin", "1");
      progress.setAttribute("aria-valuemax", String(total));
      progress.setAttribute("aria-valuenow", String(position));
      progress.setAttribute("aria-label", "Question " + position + " of " + total);
      var bar = el("span", "quiz-progress-bar");
      bar.style.width = (position / total) * 100 + "%";
      progress.appendChild(bar);
      body.appendChild(progress);

      appendSection(body, q.section);
      var stem = el("h3", "quiz-stem", q.question || "");
      body.appendChild(stem);

      var answered = hasAnswer(q);
      var correct = correctText(q);
      var options = state.options[keyOf(q)] || q.options || [];
      var list = el("div", "quiz-options");
      for (var i = 0; i < options.length; i++) {
        list.appendChild(optionButton(q, options[i], answered, correct));
      }
      body.appendChild(list);

      if (answered) {
        var explain = el("p", "quiz-explain");
        var verdict = state.answers[keyOf(q)] === correct ? "Correct. " : "Incorrect. ";
        explain.textContent = verdict + (q.explanation || "");
        body.appendChild(explain);
      }

      var actions = el("div", "quiz-actions");
      var next = el("button", "quiz-next", "Next");
      next.type = "button";
      next.disabled = !answered;
      next.addEventListener("click", function () {
        nextQuestion();
      });
      actions.appendChild(next);
      body.appendChild(actions);
    }

    function optionButton(q, choice, answered, correct) {
      var button = el("button", "quiz-option");
      button.type = "button";
      if (answered) {
        button.disabled = true;
        if (choice === correct) {
          button.classList.add("is-correct");
          button.appendChild(el("span", "quiz-mark", "✓ Correct"));
        } else if (choice === state.answers[keyOf(q)]) {
          button.classList.add("is-wrong");
          button.appendChild(el("span", "quiz-mark", "✗ Incorrect"));
        }
        if (answered && (choice === correct || choice === state.answers[keyOf(q)])) {
          button.appendChild(document.createTextNode(" "));
        }
      }
      button.appendChild(el("span", "quiz-option-text", choice));
      button.addEventListener("click", function () {
        choose(q, choice);
      });
      return button;
    }

    function renderResults(body) {
      var score = scoreRound();
      body.appendChild(el("p", "quiz-score", score.correct + " / " + score.total));
      var summary = el(
        "p",
        "quiz-description",
        score.correct === score.total ? "You got every question in this round right." : "Questions you missed:"
      );
      body.appendChild(summary);

      if (score.missed.length) {
        var list = el("ol", "quiz-missed");
        for (var i = 0; i < score.missed.length; i++) {
          var q = questions[score.missed[i]];
          var item = el("li");
          item.appendChild(el("p", null, q.question || ""));
          item.appendChild(el("p", null, "Correct answer: " + correctText(q)));
          list.appendChild(item);
        }
        body.appendChild(list);
      }

      var actions = el("div", "quiz-actions");
      var retry = el("button", "quiz-restart", "Retry all");
      retry.type = "button";
      retry.addEventListener("click", function () {
        startRound(state.deck.slice());
        announce("Quiz restarted.");
      });
      actions.appendChild(retry);
      if (score.missed.length) {
        var missed = el("button", "quiz-retry-missed", "Retry missed only");
        missed.type = "button";
        var missedIds = score.missed.slice();
        missed.addEventListener("click", function () {
          startRound(missedIds);
          announce("Retrying missed questions.");
        });
        actions.appendChild(missed);
      }
      body.appendChild(actions);
    }

    function renderCards(body) {
      body.appendChild(el("p", "quiz-count", state.queue.length + " left · " + state.done + " done"));
      if (!state.queue.length) {
        body.appendChild(el("p", "quiz-score", "Deck complete"));
        body.appendChild(el("p", "quiz-description", "You marked " + state.done + " " + (state.done === 1 ? "card" : "cards") + " as known."));
        var doneActions = el("div", "quiz-actions");
        var restart = el("button", "quiz-restart", "Restart deck");
        restart.type = "button";
        restart.addEventListener("click", function () {
          state.queue = state.deck.slice();
          state.cardIndex = 0;
          state.flipped = false;
          state.done = 0;
          announce("Deck restarted.");
          state.focusSelector = ".quiz-card";
          renderBody();
        });
        doneActions.appendChild(restart);
        doneActions.appendChild(downloadButton());
        body.appendChild(doneActions);
        return;
      }

      var q = questions[state.queue[state.cardIndex]];
      var card = el("button", "quiz-card" + (state.flipped ? " is-flipped" : ""));
      card.type = "button";
      card.setAttribute("aria-pressed", state.flipped ? "true" : "false");
      var inner = el("span", "quiz-card-inner");
      var front = el("span", "quiz-card-face quiz-card-front");
      appendSection(front, q.section);
      front.appendChild(el("p", "quiz-card-text", q.card_front || q.question || ""));
      var back = el("span", "quiz-card-face quiz-card-back");
      back.appendChild(el("p", "quiz-card-text", q.card_back || ""));
      front.setAttribute("aria-hidden", state.flipped ? "true" : "false");
      back.setAttribute("aria-hidden", state.flipped ? "false" : "true");
      inner.appendChild(front);
      inner.appendChild(back);
      card.appendChild(inner);
      card.addEventListener("click", function () {
        flip();
      });
      body.appendChild(card);

      var grades = el("div", "quiz-actions quiz-grades");
      if (!state.flipped) grades.hidden = true;
      var again = el("button", "quiz-again", "Again");
      again.type = "button";
      again.addEventListener("click", gradeAgain);
      var got = el("button", "quiz-gotit", "Got it");
      got.type = "button";
      got.addEventListener("click", gradeGotIt);
      grades.appendChild(again);
      grades.appendChild(got);
      body.appendChild(grades);

      var actions = el("div", "quiz-actions");
      var prev = el("button", "quiz-prev", "Previous");
      prev.type = "button";
      prev.disabled = state.cardIndex <= 0;
      prev.addEventListener("click", function () {
        moveCard(-1);
      });
      var next = el("button", "quiz-card-next", "Next");
      next.type = "button";
      next.disabled = state.cardIndex >= state.queue.length - 1;
      next.addEventListener("click", function () {
        moveCard(1);
      });
      actions.appendChild(prev);
      actions.appendChild(next);
      actions.appendChild(downloadButton());
      body.appendChild(actions);
    }

    function downloadButton() {
      var button = el("button", "quiz-download", "Download for Anki");
      button.type = "button";
      button.addEventListener("click", downloadAnki);
      return button;
    }

    function choose(q, choice) {
      if (hasAnswer(q)) return;
      state.answers[keyOf(q)] = choice;
      var correct = correctText(q);
      if (choice === correct) announce("Correct. " + (q.explanation || ""));
      else announce("Incorrect. The answer is " + correct + ". " + (q.explanation || ""));
      state.focusSelector = ".quiz-next";
      renderBody();
    }

    function nextQuestion() {
      if (state.mcqIndex + 1 >= state.order.length) {
        state.phase = "results";
        var score = scoreRound();
        announce("Finished. " + score.correct + " out of " + score.total + ".");
        state.focusSelector = ".quiz-restart";
      } else {
        state.mcqIndex += 1;
        state.focusSelector = ".quiz-option";
      }
      renderBody();
    }

    function startRound(indices) {
      state.order = indices.slice();
      state.mcqIndex = 0;
      state.answers = {};
      state.phase = "question";
      state.focusSelector = ".quiz-option";
      renderBody();
    }

    function flip() {
      state.flipped = !state.flipped;
      var card = root.querySelector(".quiz-card");
      if (!card) return;
      card.classList.toggle("is-flipped", state.flipped);
      card.setAttribute("aria-pressed", state.flipped ? "true" : "false");
      var front = card.querySelector(".quiz-card-front");
      var back = card.querySelector(".quiz-card-back");
      if (front) front.setAttribute("aria-hidden", state.flipped ? "true" : "false");
      if (back) back.setAttribute("aria-hidden", state.flipped ? "false" : "true");
      var grades = root.querySelector(".quiz-grades");
      if (grades) grades.hidden = !state.flipped;
      announce(state.flipped ? "Showing the answer." : "Showing the question.");
    }

    function moveCard(delta) {
      var next = state.cardIndex + delta;
      if (next < 0 || next >= state.queue.length) return;
      state.cardIndex = next;
      state.flipped = false;
      state.focusSelector = ".quiz-card";
      renderBody();
    }

    function gradeAgain() {
      var card = state.queue.splice(state.cardIndex, 1)[0];
      state.queue.push(card);
      if (state.queue.length === 1 || state.cardIndex >= state.queue.length - 1) state.cardIndex = 0;
      state.flipped = false;
      announce("Card returned to the deck. " + state.queue.length + " left.");
      state.focusSelector = ".quiz-card";
      renderBody();
    }

    function gradeGotIt() {
      state.queue.splice(state.cardIndex, 1);
      state.done += 1;
      state.flipped = false;
      if (state.cardIndex >= state.queue.length) state.cardIndex = Math.max(0, state.queue.length - 1);
      announce(state.queue.length + " left, " + state.done + " done.");
      state.focusSelector = state.queue.length ? ".quiz-card" : ".quiz-restart";
      renderBody();
    }

    function downloadAnki() {
      var lines = [];
      for (var i = 0; i < questions.length; i++) {
        var q = questions[i];
        lines.push(ankiField(q.card_front || q.question) + "\t" + ankiField(q.card_back));
      }
      var blob = new Blob(["\uFEFF" + lines.join("\n") + "\n"], { type: "text/plain;charset=utf-8" });
      var url = URL.createObjectURL(blob);
      var link = el("a");
      link.href = url;
      link.download = (data.id || "quiz") + "-anki.txt";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(function () {
        URL.revokeObjectURL(url);
      }, 1000);
    }

    function shuffleDeck(randomize) {
      var indices = range(questions.length);
      state.deck = randomize ? shuffle(indices) : indices;
      state.options = {};
      for (var i = 0; i < state.deck.length; i++) {
        var q = questions[state.deck[i]];
        var opts = (q.options || []).slice();
        state.options[keyOf(q)] = randomize ? shuffle(opts) : opts;
      }
      state.order = state.deck.slice();
      state.mcqIndex = 0;
      state.answers = {};
      state.phase = "question";
      state.queue = state.deck.slice();
      state.cardIndex = 0;
      state.flipped = false;
      state.done = 0;
    }

    function currentQuestion() {
      return questions[state.order[state.mcqIndex]];
    }

    function hasAnswer(q) {
      return Object.prototype.hasOwnProperty.call(state.answers, keyOf(q));
    }

    function scoreRound() {
      var correct = 0;
      var missed = [];
      for (var i = 0; i < state.order.length; i++) {
        var qi = state.order[i];
        var q = questions[qi];
        if (state.answers[keyOf(q)] === correctText(q)) correct += 1;
        else missed.push(qi);
      }
      return { correct: correct, total: state.order.length, missed: missed };
    }

    function appendSection(parent, section) {
      if (!section) return;
      var href = sectionHref(section);
      var node = href ? el("a", "quiz-section", section) : el("p", "quiz-section", section);
      if (href) node.setAttribute("href", href);
      parent.appendChild(node);
    }

    function announce(text) {
      var live = root.querySelector(".quiz-live");
      if (!live) return;
      live.textContent = "";
      window.setTimeout(function () {
        live.textContent = text;
      }, 30);
    }
  }

  function sectionHref(section) {
    var scope = document.getElementById("markdown-content") || document;
    var headings = scope.querySelectorAll("h1, h2, h3, h4");
    var target = normalize(section);
    var exact = null;
    var best = null;
    var bestScore = 0;
    var targetTokens = tokens(target);
    for (var i = 0; i < headings.length; i++) {
      var heading = headings[i];
      if (heading.closest(".quiz") || !heading.id) continue;
      var text = normalize(heading.textContent || "");
      if (text === target) {
        exact = heading;
        break;
      }
      var shared = 0;
      var headingTokens = tokens(text);
      for (var t = 0; t < targetTokens.length; t++) {
        if (headingTokens.indexOf(targetTokens[t]) !== -1) shared += 1;
      }
      var score = targetTokens.length ? shared / targetTokens.length : 0;
      if (shared >= 2 && score > bestScore) {
        bestScore = score;
        best = heading;
      }
    }
    var match = exact || (bestScore >= 0.55 ? best : null);
    return match ? "#" + match.id : "";
  }

  function normalize(text) {
    return String(text)
      .toLowerCase()
      .replace(/^\s*\d+\.\s*/, "")
      .replace(/[^a-z0-9]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function tokens(text) {
    var parts = text.split(" ");
    var out = [];
    var skip = { the: 1, and: 1, for: 1, with: 1, from: 1 };
    for (var i = 0; i < parts.length; i++) {
      if (parts[i].length > 2 && !skip[parts[i]]) out.push(parts[i]);
    }
    return out;
  }

  function keyOf(q) {
    return q.id || q.question || "";
  }

  function correctText(q) {
    var options = q.options || [];
    return options[q.answer] || "";
  }

  function range(n) {
    var list = [];
    for (var i = 0; i < n; i++) list.push(i);
    return list;
  }

  function shuffle(list) {
    var copy = list.slice();
    for (var i = copy.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = copy[i];
      copy[i] = copy[j];
      copy[j] = tmp;
    }
    return copy;
  }

  function ankiField(value) {
    return String(value || "")
      .replace(/[\t\r\n]+/g, " ")
      .trim();
  }

  function readMode(id) {
    try {
      return window.localStorage.getItem("quiz-mode:" + (id || "quiz"));
    } catch (err) {
      return null;
    }
  }

  function writeMode(id, mode) {
    try {
      window.localStorage.setItem("quiz-mode:" + (id || "quiz"), mode);
    } catch (err) {
      /* Storage can be blocked. */
    }
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }
})();
