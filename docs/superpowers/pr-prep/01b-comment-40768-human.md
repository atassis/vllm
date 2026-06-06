<!--
  Человечный вариант комментария в #40768. Без AI-ремарки, без markdown-заголовков/
  списков/жирного, обычные дефисы (не em-dash). Абзацами.

  ЧЕСТНОСТЬ:
  - AGENTS.md требует disclosure об AI в ОПИСАНИИ PR (он есть в 03-pr-description.md).
    Коммент к чужому PR — не описание PR, так что убрать ремарку тут нормально.
  - НО: перед постингом убедись, что оба фактических утверждения ниже — твои наблюдения,
    а не мои: (1) "backing out your change brings the crash back" — ты это реально видел;
    (2) нюанс с _update_after_schedule на :943 — проверь номер строки на свежем main,
    он мог сдвинуться. Ты human submitter, ты это защищаешь.
-->

Hey @z1ying, I've been working on getting MTP speculative decoding running under pipeline
parallelism (PP > 1) and ran straight into the same placeholder leak you're fixing here.
To be upfront: I ended up porting your three changes onto current main just to unblock my
own testing, so the scheduler part of my branch is basically your code. I don't want to
open a second PR that does the same thing, so I'd rather lean on this one and help where I
can.

A couple of things that might be useful. First, your fix matters for the multi-GPU case
too, not just the single-GPU repro in the description. On the non-last pipeline rank the
worker-side overwrite gets skipped for the same reason, so the -1 can leak there as well.
MTP is one of the async-eligible methods @Sandermage mentioned above, so unlike the ngram
case this path actually fires, and I've been hitting it - backing your change out brings
the crash right back for me.

Second, a heads-up for the rebase mergify is asking for. On current main
_update_after_schedule moved to the end of schedule() (around scheduler.py:943), and it
re-sets the placeholder intent for the next step right after
_consume_spec_decode_tokens_for_step clears it. When I rebased I had to account for that or
the count gets reserved twice. Might save you a few minutes.

I also have a couple of CPU-only tests for the re-added-after-preemption and
freshly-scheduled cases if they'd help, otherwise I'll keep them in my follow-up and link
back here.

One more thing so you can plan around it: my correctness work sits on top of this, so I'm
somewhat blocked until it lands. If you have time for the rebase that's by far the
cleanest path. If you'd rather not, or things are just busy, I'm happy to fold this change
into my PR and keep you on it as a co-author - the credit stays yours either way. And if I
don't hear back, I'll go that route but still credit you for it rather than reinvent it.

Happy to help get this landed however suits you best. What do you think?
