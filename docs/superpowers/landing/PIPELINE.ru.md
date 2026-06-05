# Конвейер выполнения vLLM V1 — от начала до конца

> **Аудитория.** Инженер, который хочет *по-настоящему* понять, как запрос проходит
> через vLLM V1: каждый этап, каждый компонент, точную терминологию (и где
> терминология vLLM расходится с принятой в индустрии), что такое pipeline parallelism (PP,
> параллелизм конвейера) и как с ним сочетается speculative decoding (спекулятивное
> декодирование) — а также, честно говоря, где реализация архитектурно слабовата,
> избыточно связана или неудачно названа.
>
> **Область охвата.** vLLM **V1** (runner по умолчанию для наших моделей). V2
> упоминается только там, где есть отличия. Этот документ **расширяет** существующую
> кодово-реферируемую базу знаний в `docs/superpowers/research/pp-mtp/` (кирпичи 10–81)
> от «spec-под-PP» до «весь конвейер V1». Кирпичи не повторяются — на них даются ссылки.
>
> **Происхождение / как доверять этому.** Каждое архитектурное утверждение снабжено
> ссылкой `file:line`. Номера строк — это **якоря** из рабочего дерева на HEAD
> `e45e5d462` (ветка `feat/pp-mtp-spec-decode`) — `vllm/v1/worker/gpu_model_runner.py`
> и `vllm/v1/core/sched/scheduler.py` содержат **незакоммиченные изменения spec-под-PP**,
> поэтому часть номеров там из модифицированного дерева. Если строка сдвинулась —
> перегрепайте символ. Факты без `file:line` помечены `HYPOTHESIS`. Ключевые ссылки
> в §0, §1, §4, §8, §9 были перепроверены непосредственно по дереву; более широкие
> карты получены параллельным чтением подсистем и точны в пределах нескольких строк.

---

## 0. Общая форма системы (один экран)

Жизнь запроса сверху вниз:

```
                    ┌─────────────────── FRONTEND PROCESS ───────────────────┐
  HTTP / Python →   │  AsyncLLM (async_llm.py:70)  /  LLMEngine (47)         │
                    │    Processor  → EngineCoreRequest                       │
                    │    OutputProcessor ← EngineCoreOutputs                  │
                    │      └ IncrementalDetokenizer + LogprobsProcessor       │
                    │    EngineCoreClient  (core_client.py)  ── ZMQ ──┐       │
                    └─────────────────────────────────────────────────┼──────┘
                                                                       │ msgpack
                    ┌──────────────── ENGINECORE PROCESS ──────────────┼──────┐
                    │  EngineCoreProc (core.py:858)                     ▼      │
                    │   input thread → input_queue → run_busy_loop (1216)      │
                    │       │                                                  │
                    │       ▼  step_fn() each iteration (217)                  │
                    │   ┌── EngineCore.step / step_with_batch_queue ──┐        │
                    │   │  Scheduler.schedule()  (sched/scheduler.py)  │        │
                    │   │     → SchedulerOutput                        │        │
                    │   │  Executor.execute_model(non_block=True)      │        │
                    │   │     → Future[ModelRunnerOutput]              │        │
                    │   │  Scheduler.update_from_output()              │        │
                    │   └──────────────────────────────────────────────┘       │
                    │   post_step()  (sync spec draft-token pull, 474)         │
                    │   outputs → output_queue → output thread → ZMQ           │
                    └──────────────────────┬───────────────────────────────────┘
                                           │ collective_rpc (executor)
                    ┌──────────────────────▼─ WORKER PROCESS(es), 1 per rank ─┐
                    │  Worker (gpu_worker.py)                                  │
                    │    GPUModelRunner (gpu_model_runner.py:422)              │
                    │      _update_states → _prepare_inputs → model.forward    │
                    │      → sample → (rejection oracle) → ModelRunnerOutput   │
                    │      PP: IntermediateTensors send/recv between stages    │
                    └──────────────────────────────────────────────────────────┘
```

Три уровня процессов, связанных ZMQ (frontend↔core) и RPC исполнителя
(core↔workers). Цикл движка — это **синхронный busy-loop (цикл занятого ожидания)**, не asyncio —
единственный asyncio находится во frontend-компоненте `AsyncLLM`. Всё нижеследующее
является расширением этой картины.

---

## 1. Верхнеуровневая архитектура и модель процессов/потоков

### 1.1 Карта компонентов

| Компонент | `file:line` | Роль |
|---|---|---|
| `LLMEngine` | `vllm/v1/engine/llm_engine.py:47` | Синхронный in-process фасад (`llm.generate(...)`, офлайн-батч). |
| `AsyncLLM` | `vllm/v1/engine/async_llm.py:70` (наследует `EngineClient`) | Asyncio-фасад для API-сервера; владеет фоновой задачей `output_handler`. |
| `EngineCore` | `vllm/v1/engine/core.py:95` | **Логика** движка: scheduler + executor + KV + structured-output. Без I/O. |
| `EngineCoreProc` | `vllm/v1/engine/core.py:858` (подкласс) | `EngineCore`, обёрнутый для **отдельного процесса**: ZMQ-сокеты + 2 демон-потока I/O + busy-loop. |
| `DPEngineCoreProc` | `vllm/v1/engine/core.py:1676` | Data-parallel вариант: координация волн + all-reduce барьеры по DP-рангам. |
| `EngineCoreClient` | `vllm/v1/engine/core_client.py` | Абстрактный клиентский фасад. Конкретные реализации: `InprocClient` (без MP), `SyncMPClient`, `AsyncMPClient`, DP-варианты. |
| `Processor` | `vllm/v1/engine/input_processor.py` | Сырой промпт → токенизация/мультимодальная обработка → `EngineCoreRequest`. |
| `OutputProcessor` | `vllm/v1/engine/output_processor.py` | `EngineCoreOutputs` → `RequestOutput`; потоковое состояние на запрос. |
| `IncrementalDetokenizer` | `vllm/v1/engine/detokenizer.py` | Статефул-детокенизатор токенов в текст, обнаружение стоп-строк. |
| `LogprobsProcessor` | `vllm/v1/engine/logprobs.py` | Накапливает и форматирует logprobs (логарифмы вероятностей). |
| `DPCoordinator` | `vllm/v1/engine/coordinator.py:23` | DP-прокси на стороне frontend; порождает брокер-процесс для координации волн и нагрузки. |

### 1.2 Границы процессов и потоков

- **Frontend-процесс.** `AsyncLLM`/`LLMEngine` + `Processor` + `OutputProcessor` +
  `EngineCoreClient`. В `AsyncLLM` цикл asyncio; сторона вывода работает
  как фоновая asyncio-задача `_run_output_handler` (`async_llm.py`). В
  `SyncMPClient` сторона вывода — это **демон-поток**, опрашивающий ZMQ-сокет
  (`core_client.py`).
- **EngineCore-процесс** (`EngineCoreProc`). Порождает **два демон-потока** —
  `process_input_sockets` (ZMQ → `input_queue`) и `process_output_sockets`
  (`output_queue` → ZMQ) — и запускает `run_busy_loop` на главном потоке
  (`core.py:1216`). Busy-loop — это `while ...: _process_input_queue();
  _process_engine_step()`. Вывод передаётся через `output_queue.put_nowait(...)`
  (`core.py:~1264`), поэтому цикл **никогда не блокируется на сериализации вывода**.
- **Worker-процесс(ы)**, по одному на каждый глобальный ранг под `MultiprocExecutor`. Каждый запускает
  `Worker`, оборачивающий `GPUModelRunner`; исполнитель диспатчит `execute_model` через
  `collective_rpc` поверх очередей сообщений.

> Разделение между **логикой** (`EngineCore`) и **транспортом** (`EngineCoreProc` +
> `EngineCoreClient`) — самый чёткий шов в системе: `InprocClient` запускает
> `EngineCore` непосредственно in-process для офлайн-использования, тогда как MP-клиенты
> помещают его за ZMQ. Плата за это — одна концепция («движок») размазана по трём именам
> классов.

### 1.3 Что пересекает границу client↔core

`EngineCoreRequest` (`vllm/v1/engine/__init__.py:~83`) идёт client→core;
`EngineCoreOutputs` / `EngineCoreOutput` (`__init__.py:~170`/`~215`) идут core→client.
Оба являются `msgspec.Struct` с `array_like=True` для компактного позиционного msgpack. Тип
запроса помечается через `EngineCoreRequestType` (`ADD`/`ABORT`/`UTILITY`/
`START_DP_WAVE`/...). Анализ типизации — в кирпиче 81; о пробеле, который создаёт
`array_like=True` (порядок полей как хрупкий wire-контракт), — в §9.

---

## 2. Жизненный цикл запроса и конечный автомат состояний

Основной файл: `vllm/v1/request.py`.

### 2.1 Объект `Request` и учёт токенов

`Request` (обычный класс, не `@dataclass`) хранит состояние на один запрос. Поля
учёта токенов — это сердце контракта планировщика:

| Поле / свойство | `file:line` | Смысл |
|---|---|---|
| `prompt_token_ids` / `num_prompt_tokens` | `request.py:~130` | Входные токены; иммутабельны. |
| `_output_token_ids` → `output_token_ids` | `request.py:~133`, добавление на `~226` | Сгенерированные токены (фаза декодирования), доступны только для чтения через `ConstantList`. |
| `_all_token_ids` → `all_token_ids` | `request.py:~134` | prompt + output, синхронизируются через `append_output_token_ids()`. |
| `num_tokens` (property) | `request.py:249` | `len(_all_token_ids)` — **подтверждённая** длина последовательности. |
| `num_tokens_with_spec` (property) | `request.py:253` | `num_tokens + len(spec_token_ids)` — **оптимистичная** длина при условии принятия всех драфтов. |
| `spec_token_ids` | `request.py:~148` | Кандидат-токены драфтера (или заглушки `[-1]*k` при async). Потребляются по одному разу за шаг. |
| `num_computed_tokens` | `request.py:~149` | Сколько токенов **планировщик решил вычислить**. Откатывается назад при отклонении спекуляции; сбрасывается в 0 при вытеснении (preemption). |
| `num_output_placeholders` | `request.py:141` | **Только async-планирование.** Выходные токены, обещанные, но ещё не материализованные. |

Ключевое неочевидное соотношение (см. глоссарий, §10): `num_computed_tokens`
— это **счётчик решений планировщика**, а не факт GPU. Он может превышать `num_tokens`
(токены спекуляции запланированы оптимистично) или быть меньше (после отката при отклонении).

### 2.2 `RequestStatus` и переходы

`RequestStatus(enum.IntEnum)` в `request.py:329`. Состояния:

- `WAITING` (332), а также заблокированные варианты ожидания для grammar
  structured-output (структурированного вывода), удалённой загрузки KV и стриминга
  (`WAITING_FOR_*`, регион 333–335).
- `RUNNING` (336), `PREEMPTED` (337) — вытеснено.
- Терминальные (> `PREEMPTED`, см. `is_finished` на `request.py:~351`): `FINISHED_STOPPED`
  (340), `FINISHED_LENGTH_CAPPED` (341), `FINISHED_ABORTED` (342),
  `FINISHED_IGNORED` (343), `FINISHED_ERROR` (344), `FINISHED_REPETITION` (345).
- `_FINISHED_REASON_MAP` (`request.py:363`) отображает каждый терминальный статус в
  `FinishReason` (`STOP`/`LENGTH`/`ABORT`/`ERROR`/`REPETITION`).

Что управляет каждым переходом (всё в `sched/scheduler.py`):

- **WAITING → RUNNING**: `schedule()` принимает запрос, когда доступны токенный бюджет + KV-блоки
  (статус устанавливается ~`scheduler.py:831`).
- **RUNNING → PREEMPTED**: `_preempt_request()` (~`scheduler.py:960-977`) при
  `allocate_slots()`, вернувшем `None`; освобождает блоки, `num_computed_tokens = 0`,
  очищает `spec_token_ids`, возвращает в начало очереди `waiting`.
- **RUNNING → FINISHED_***: проверки остановки в `sched/utils.py` (`check_stop`) и
  `update_from_output`; прерывание через `finish_requests()`.
- **WAITING → WAITING_FOR_REMOTE_KVS → (WAITING|PREEMPTED)**: путь async-загрузки через
  KV-коннектор.

Переходы **не** охраняются единым объектом конечного автомата — каждый переход —
это голый `request.status = ...`, разбросанный по планировщику (§8, слабое место
W2).

---

## 3. Планировщик и continuous batching (непрерывное батчирование)

Файлы: `vllm/v1/core/sched/{scheduler,async_scheduler,output}.py`,
`vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_coordinator}.py`,
`vllm/v1/kv_cache_interface.py`. (Замечание: `scheduler.py` — ~2370 строк / 109 KB —
«большой», но не как runner с 7583 строками.)

### 3.1 Фазы prefill нет — только цикл догоняющего бюджета токенов

Собственный комментарий планировщика (`scheduler.py:~338`) формулирует модель: *нет
фазы decode или prefill; у каждого запроса есть `num_computed_tokens` и
`num_tokens_with_spec`, и каждый шаг назначает токены так, чтобы `num_computed_tokens`
догонял.* `schedule()` (`scheduler.py:336`):

1. `token_budget = max_num_scheduled_tokens` (ограничение на шаг).
2. **Очередь running — в первую очередь** (`~372-513`): для каждого запроса вычисляется
   `num_new_tokens = num_tokens_with_spec + num_output_placeholders -
   num_computed_tokens`, зажимается до бюджета / порога chunked-prefill, вызывается
   `kv_cache_manager.allocate_slots(...)`. Если возвращается `None` — **вытесняется**
   запрос с наименьшим приоритетом (или последний) и повторяется попытка.
3. **Очередь waiting — во вторую очередь** (`~558-851`): поиск по prefix-кешу,
   опциональная удалённая KV-загрузка, расчёт токенов для chunked-prefill,
   `allocate_slots`, принятие → `RUNNING`.
4. Строится `SchedulerOutput`; `_update_after_schedule()` продвигает
   `num_computed_tokens += num_scheduled_tokens` и устанавливает `is_prefill_chunk`.

**Continuous batching** вытекает из этого: никакого фиксированного батча нет — каждый шаг
заново выводит работающее множество, завершённые запросы выпадают, а ожидающие присоединяются.

**Chunked prefill** — просто зажатие бюджетом: `num_new_tokens` длинного промпта
ограничивается оставшимся `token_budget` (и `long_prefill_token_threshold`), поэтому
промпт потребляется за несколько шагов по мере продвижения `num_computed_tokens`.

### 3.2 Prefix caching (кеширование префиксов)

`kv_cache_manager.get_computed_blocks()` (`kv_cache_manager.py:~196`) хеширует
последовательность блоков запроса (`Request.block_hashes`) и вызывает
`coordinator.find_longest_cache_hit(...)`, возвращая самый длинный закешированный префикс и
количество покрытых им токенов. Хеш-таблица — `BlockHashToBlockMap` в
`block_pool.py:~34`. Тонкость: поиск обрезает до `num_tokens - 1`, так что **последний
токен всегда перевычисляется** для получения logit'ов (`kv_cache_manager.py:~221`).

### 3.3 KV-cache manager и пул блоков

`BlockPool` (`block_pool.py:~130`) владеет всеми `num_gpu_blocks` объектами
`KVCacheBlock`, `FreeKVCacheBlockQueue` (порядок вытеснения) и хеш-таблицей
prefix-кеша. На запрос: `SingleTypeKVCacheManager.req_to_blocks[req_id]` — append-only
список блоков. `allocate_slots()` (`kv_cache_manager.py:238`) — привратник: освобождает
блоки вне окна (sliding window), проверяет наличие свободных блоков (возвращает `None`
→ инициирует вытеснение), выделяет и кеширует заполненные блоки. Блоки
подсчитываются по ссылкам; блок возвращается в очередь свободных только при `ref_cnt == 0`.
Гибридные модели используют несколько `kv_cache_groups` (см. §7).

### 3.4 Scheduler vs AsyncScheduler — ключевое различие

`AsyncScheduler(Scheduler)` (`async_scheduler.py:12`) переопределяет пост-шаговые
bookkeeping-операции (учёт состояния), чтобы движок мог **планировать шаг N+1 до того,
как вернётся вывод шага N**:

- `_update_after_schedule`: для каждого не-prefill запроса
  `num_output_placeholders += 1 + cur_num_spec_tokens` (`async_scheduler.py:31`) и
  `request.spec_token_ids = self._spec_token_placeholders` (разделяемый список `[-1]*num_spec`,
  `async_scheduler.py:~16`). **Это `-1` является источником заглушки, которую путь
  spec-под-PP должен заполнять** (см. §8).
- Под PP также устанавливается `next_decode_eligible_step = current_step + pp_size`
  (`async_scheduler.py:45`) — ограничение следующего шага декодирования запроса в соответствии
  с ритмом конвейера.
- `_update_request_with_output`: `num_output_placeholders -= len(new_token_ids)`;
  `assert num_output_placeholders >= 0` (`async_scheduler.py:63-64`).
- `prev_step_scheduled_req_ids` отслеживает принадлежность к предыдущему батчу — именно тот сигнал,
  который использует upstream PR #40768 для решения о безопасности вставки заглушки `-1`.

### 3.5 Контракт `SchedulerOutput` (scheduler → executor, на каждый шаг)

`SchedulerOutput` (`sched/output.py:180`) содержит: `scheduled_new_reqs:
list[NewRequestData]` (полное состояние, по одному разу на запрос, `output.py:31`),
`scheduled_cached_reqs: CachedRequestData` (дельты каждый шаг, `output.py:111`),
`num_scheduled_tokens: dict[str,int]`, `total_num_scheduled_tokens`,
`scheduled_spec_decode_tokens: dict[str, list[int]]` (драфт-токены),
`num_common_prefix_blocks` (cascade-attention),`finished_req_ids`,
`preempted_req_ids`. Заметим: `scheduled_spec_decode_tokens` использует **неявное «отсутствующий ключ =
нет драфтов»** (§9, пробел типизации).

---

## 4. Цикл движка и режимы батчирования

Файл: `vllm/v1/engine/core.py`. Здесь фактически происходит конвейеризация PP.

### 4.1 step_fn выбирается один раз

`self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue`
(`core.py:217`). `batch_queue` существует тогда и только тогда, когда `max_concurrent_batches > 1`
(`core.py:192-198`, это `deque(maxlen=...)`).
`max_concurrent_batches` (`config/vllm.py:497-507`) возвращает **`pp_size`** в базовом
случае, `pp_size+1` для async+V2, `2` для async при pp≤1. Таким образом: **PP>1 ⇒ batch_queue
включён**, даже без async. Печально известный комментарий живёт здесь:
`config/vllm.py:504` — *"V1 Model Runner does not fully support async scheduling with
PP."*

### 4.2 `step()` — простой синхронный путь

`core.py:443`: `schedule()` → `execute_model(non_block=True)` → **`future.result()`
блокирует** (~`core.py:461`) → `update_from_output()`. Один батч в полёте; поток движка
блокируется на GPU.

### 4.3 `step_with_batch_queue()` — конвейерный путь

`core.py:484`. На каждой итерации:

1. Если дек не заполнен, планирует **новый** батч через `schedule()`, запускает
   `execute_model(non_block=True)`, выполняет `appendleft((future, scheduler_output,
   exec_future))`, и **ранний возврат `(None, executed)`** — *без ожидания* — пока
   есть работа и в очереди есть место (`core.py:~541`).
2. Как только заполнен (или нет новой работы), извлекает **самую старую** запись через
   `batch_queue.pop()` и блокируется на её `future.result()` (`core.py:~555`),
   затем вызывает `update_from_output()`.

Поскольку очередь хранит до `pp_size` батчей в полёте и применяется только самый старый,
**`update_from_output` отстаёт от планирования приблизительно на `pp_size − 1` шагов**.
Эта задержка и есть смысл — она держит все PP-стадии занятыми (заполняет конвейер).
`SchedulerOutput`, захваченный на шаге N, является **снимком состояния**; `spec_token_ids`,
`num_output_placeholders`, `is_prefill_chunk` могут все измениться до того, как
соответствующий вывод будет применён через k шагов. Этот window нестабильности — именно то место,
где живёт корректность spec-под-PP (кирпич 40).

Есть также ветка `deferred_scheduler_output` для **structured output** (откладывает
семплинг до применения предыдущего вывода, `core.py:~574-596`).

### 4.4 `post_step()` — SYNC-путь drafte-токенов спекуляции

`core.py:474`:

```python
def post_step(self, model_executed):
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
```

Вызывается на каждой итерации после `step_fn` (`core.py:~1264`). Это **sync**-сантехника
спекуляции: извлечь вывод драфтера из исполнителя и передать его планировщику.
**При async — это no-op**: worker напрямую вставляет драфт-токены во входной батч.
Два пути не имеют общей абстракции (§8, W4). Кирпич 40 §«Session 5» фиксирует, что
sync-путь **дедлочит** для MTP+PP на текущем main (ранг ждёт вывода спекуляции, который
не приходит), тогда как async-путь **падает** (утечка заглушки `-1`) — то есть *оба* режима
не завершены для MTP+PP.

### 4.5 Путь вывода и busy-loop

Выводы попадают в `output_queue` через `put_nowait`; демон-поток `process_output_sockets`
сериализует их и отправляет через ZMQ, так что цикл никогда не блокируется на I/O.
`run_busy_loop` (`core.py:1216`; DP-переопределение `core.py:1844`) — это синхронный
цикл `_process_input_queue(); _process_engine_step()`.

---

## 5. Executor и worker

Файлы: `vllm/v1/executor/{abstract,uniproc_executor,multiproc_executor,ray_executor}.py`;
`vllm/v1/worker/{worker_base,gpu_worker,gpu_model_runner,gpu_input_batch}.py`.

### 5.1 Executors

`Executor(ABC)` (`abstract.py:~37`) предоставляет `collective_rpc(method, args, ...,
non_block=...)` — широковещательный вызов всем worker'ам с сбором результатов.
`execute_model` — просто `collective_rpc("execute_model", ...)`. Конкретные реализации:

- `UniProcExecutor` (`uniproc_executor.py:~45`): один in-process worker, прямые вызовы.
- `MultiprocExecutor` (`multiproc_executor.py:~103`, `supports_pp=True`): один worker-**процесс
  на ранг**; RPC через shared-memory `MessageQueue`s (широковещание на вход,
  ответы per-rank на выход); мониторинг-поток следит за жизнеспособностью worker'ов.
- `RayDistributedExecutor` (`ray_executor.py:~64`): Ray-акторы на ранг, опционально
  **скомпилированный DAG**, соединяющий TP-группы внутри каждой PP-стадии и передающий
  `IntermediateTensors` между стадиями.

### 5.2 Worker

`WorkerBase` (`worker_base.py:~39`) / `WorkerWrapperBase` (`~187`, лениво создаёт
конкретный worker и делегирует через `__getattr__`). `gpu_worker.Worker`
(`gpu_worker.py:~112`): `init_device` (установить CUDA-устройство, инициализировать distributed),
`load_model`, **`determine_available_memory`** (фиктивный прогон профилирования KV,
измеряющий пиковую память активаций, затем вычисляющий `num_gpu_blocks`,
`gpu_worker.py:~360`), `execute_model`. Recv/send PP-тензоров `IntermediateTensors`
происходит в worker'е вокруг прямого прохода runner'а.

### 5.3 GPUModelRunner — объект-бог

`GPUModelRunner` (`gpu_model_runner.py:422`, **~7583 строки**) делает *всё*:
согласование персистентного батча, подготовку входов, метаданные внимания, прямой проход,
семплинг, предложение спекуляции, PP-broadcast, мультимодальность, пулинг, CUDA-графы,
профилирование. Грубая карта (диапазоны строк приблизительны, модифицированное дерево):

| Раздел | ~строки | Ключевые методы |
|---|---|---|
| init / async output wrappers | 243–913 | `AsyncGPUModelRunnerOutput`, `ExecuteModelState` |
| **согласование персистентного батча** | 1132–1559 | `_update_states`, `_update_states_after_model_execute` |
| подготовка входов | 1560–2505 | `_prepare_inputs`, `_prepare_input_ids` (1708), `_build_attention_metadata` |
| оркестрация прямого прохода | 3419–3800 | `_preprocess`, `_model_forward` |
| **execute_model** | 4010–4388 | основной forward+PP send |
| **sample_tokens** | 4389–4649 | sampler + rejection oracle + предложение драфта |
| **PP spec transport** | 4650–4699 | `_pp_broadcast_prev_sampled_token_ids`, `_pp_receive_*` |
| предложение спекуляции | 4701–5083 | `propose_draft_token_ids`, `take_draft_token_ids` |
| загрузка / профилирование | 5096–6469 | `load_model`, `profile_run` |

Драфтер создаётся **только на последнем PP-ранге** (`gpu_model_runner.py:~542`),
иначе `self.drafter = None` — источник пяти `AttributeError` на не-последних рангах
в сессии 3 (README §SESSION 3). Разделение execute на
**`execute_model` (прямой проход, все ранги) + `sample_tokens` (семплинг, последний ранг)** существует
для поддержки скомпилированного DAG Ray и перекрытия семплинга с подготовкой следующей стадии.

### 5.4 Персистентный батч (`InputBatch` / `CachedRequestState`)

`gpu_input_batch.py`: `CachedRequestState` (`:34`) — состояние worker'а на запрос;
`InputBatch` (`:91`) — **персистентный батч** — CPU-буферы (`token_ids_cpu`,
`is_token_ids`, `num_computed_tokens_cpu`, `block_table`), которые **сохраняются между шагами**,
чтобы не перестраивать тензоры на каждой итерации. `_update_states`
(`gpu_model_runner.py:1132`) согласует его с каждым `SchedulerOutput`: удаляет
завершённые/незапланированные, добавляет новые/возобновлённые, обновляет работающие, затем
**`condense()`** (`gpu_input_batch.py:683`) компактирует плотный префикс после удалений
(переставляет запрос с наибольшим индексом на освободившийся слот с наименьшим). После condense
`prev_req_id_to_index` перестраивается — известный риск устаревшего маппинга под конвейерной
задержкой (§8, W6; кирпич 80 §3).

---

## 6. Распределённый параллелизм (TP / PP / DP / EP / CP)

Файл: `vllm/distributed/parallel_state.py`. `GroupCoordinator` (`:~290`) оборачивает
torch `ProcessGroup` (`device_group` на NCCL + `cpu_group` на gloo). По одному синглтону
на ось на процесс (`_TP/_PP/_DP/_EP/_EPLB/_PCP/_DCP`, `:~1257-1318`), у каждого есть
accessor (`get_pp_group()` и т.д.). `initialize_model_parallel(...)` (`:~1522`)
вызывается один раз с размерами **целевой** модели;
`world_size = ExternalDP × DP × PP × PCP × TP` (`:~1588`). Кирпич 10 — это глубокий
разбор PP; данный раздел добавляет остальные оси.

| Ось | Синглтон | Что **шардируется** | Что **реплицируется** | Что **коммуницируется** |
|---|---|---|---|---|
| **TP** tensor | `_TP` | веса *внутри* слоя (`ColumnParallelLinear` `linear.py:~407` шардирует out-dim; `RowParallelLinear` `:~1389` шардирует in-dim; `VocabParallelEmbedding` шардирует словарь `vocab_parallel_embedding.py:~192`) | input ids | **all-reduce / all-gather** активаций *на каждый слой* (PCIe-нагруженный трафик, которого мы избегаем, не используя TP через пару без NVLink) |
| **PP** pipeline | `_PP` | слои *по рангам* (`make_layers` + `get_pp_indices`, `distributed/utils.py:~109`) | — | `IntermediateTensors` {hidden_states, residual}, **передаваемые stage→stage** (`send_tensor_dict`/`recv_tensor_dict` `:~852-1069`; метаданные — на cpu_group, тензоры — на device_group) |
| **DP** data | `_DP` | **батч** (разные запросы на реплику) | **вся модель** | координация волн (DP coordinator); синхронизация числа запросов на шаг + all-reduce барьеры (`DPEngineCoreProc`) |
| **EP** expert | `_EP`/`_EPLB` | **MoE-эксперты** по рангам (`FusedMoE`); размер EP-группы = DP×PCP×TP, один на PP-стадию | не-экспертные веса | **all-to-all** токенов к/от их экспертов |
| **CP** context | `_PCP` (prefill) / `_DCP` (decode) | измерение **последовательности/контекста** | модель | all-gather / all-reduce (или all-to-all) срезов Q/K/V; `_DCP` переиспользует TP-GPU с `dcp_size ≤ tp_size` |

`broadcast` — это `GroupCoordinator.broadcast` (`:~637`) на device-группе.
**В coordinator'е нет специализированного broadcast-хелпера для спекуляции** —
broadcast сэмплированных токенов написан вручную в runner'е (`pp_spec_broadcast.py`),
что и является швом, которым владеет наша работа B1a (§8).

### Глубокий разбор: pipeline parallelism

PP разделяет **слои** модели на смежные стадии по рангам. При `L` слоях
и `pp_size` стадиях `get_pp_indices` даёт каждому рангу срез `[start, end)` — равное
разбиение `L // pp_size` с остатком, переданным средним рангам (последний ранг сохраняет
output norm). `VLLM_PP_LAYER_PARTITION` переопределяет это (рычаг памяти Q13, кирпич 40/70).
**Что пересекает границу стадии** — мало: словарь
`{hidden_states, residual}` на токен, а не тяжёлый per-layer all-reduce как у TP — именно
поэтому PP, а не TP выбирается для PCIe-пары без NVLink.

**Соглашение о размещении** (кирпич 10/20): `embed_tokens` живёт на **первом** ранге
(или на последнем тоже, если `tie_word_embeddings`); `norm` + `lm_head` — на **последнем**.
Нелокальные слои — это `PPMissingLayer` (no-op `nn.Identity`), их веса пропускаются при
загрузке. **Канонический PP forward** (`if is_first_rank: embed; else: read
intermediate_tensors; ... ; if not is_last_rank: return IntermediateTensors; else
norm`) — это точный паттерн, который воспроизводит MTP-драфт, и причина, по которой Design C
требует флага «standalone draft», чтобы драфт (который работает на последнем ранге, где
`is_first_rank==False`) не попадал в ветку чтения промежуточных тензоров (кирпичи 10/60).

**Пузыри.** Конвейер имеет пузыри заполнения/слива (первые/последние `pp_size−1` шагов,
когда не все стадии заняты). vLLM скрывает устойчивые пузыри, держа `pp_size` батчей
в полёте через `batch_queue` (§4.3). Runner V2 (#42187) дополнительно сокращает пузыри,
но **недоступен нашей квантованной гибридной Qwen3.5** (кирпич 40: V2 требует
`not is_moe and not is_quantized` и находится в whitelist архитектур).

**Композиция.** TP×PP образуют сетку (`linear.py` шардирует внутри стадии, PP — между
стадиями). PP×DP реплицируют PP-конвейер на каждый DP-ранг. PP×spec — сложная часть:
драфтер работает только на последней стадии, и его сэмплированные токены необходимо
широковещательно передать ранним стадиям, чтобы те могли построить следующий вход (§8).

---

## 7. KV-кеш и механизм внимания

Файлы: `vllm/v1/kv_cache_interface.py`, `vllm/v1/worker/block_table.py`,
`vllm/v1/core/kv_cache_utils.py`, бэкенды внимания.

- **PagedAttention.** KV хранится в фиксированных **блоках** (аналогия с пейджингом ОС).
  На каждый запрос — **таблица блоков** (`block_table.py:~70`,
  `[max_reqs, max_blocks_per_req]` int32), отображающая логический индекс блока → физический
  id блока. На каждый токен — **slot mapping** (маппинг слотов, `block_table.py:~75`), задающий
  физический слот токена `block_id * block_size + offset`; дополняющие записи используют
  `PAD_SLOT_ID = -1`. Бэкенды внимания читают slot mapping для записи/чтения KV без
  обращения к таблице блоков.
- **Типы KVCacheSpec** (`kv_cache_interface.py`): `FullAttentionSpec` (`~203`),
  `SlidingWindowSpec` (`~459`), `MLAAttentionSpec` (`~352`, DeepSeek latent),
  `ChunkedLocalAttentionSpec`, **`MambaSpec`** (`~605`, SSM/conv-состояние — фиксированная
  форма, *не* индексируется по токенам), `CrossAttentionSpec`, `EncoderOnlyAttentionSpec`.
  `KVCacheGroupSpec` (`~837`) группирует слои, разделяющие спецификацию; гибридные модели
  получают несколько групп, дополненных до единого размера страницы.
- **Гибридные модели (mamba/GDN + attention).** Qwen3.5 смешивает attention KV с
  Mamba conv/SSM-состоянием. Слои attention используют paged-блоки; слои Mamba используют
  фиксированные per-request буферы состояния (без таблицы блоков).
  `_update_states_after_model_execute`
  (`gpu_model_runner.py:~1502`, **только гибридный**) вычисляет per-request принятые счётчики
  `(sampled != -1).sum(dim=1)` — релевантно для учёта спекуляции (кирпич 80 §3, Q17).
  GDN-ядра `causal_conv1d` JIT-компилируются и выполняются (кирпич 70 — *не* memory wall).
- **KV-профилирование.** `determine_available_memory` запускает фиктивный прямой проход при
  `max_num_batched_tokens`, измеряет пиковую память активаций, вычитает её (+ накладные расходы
  + cudagraph-резервации) из общего VRAM и делит на размер страницы для получения
  `num_gpu_blocks` (`gpu_model_runner.py` путь профилирования + `kv_cache_utils.py:~1258`).
  `num_gpu_blocks_override` обходит это.

Специально для стороны драфта (кирпич 30): MTP-драфт имеет **собственные** KV-тензоры,
но входит в **ту же** `kv_cache_group`, что и целевая модель, **разделяет таблицы блоков /
slot mapping** и откатывает отклонения через общий `seq_lens` (перезапись при повторном
использовании, а не явная очистка).

---

## 8. Speculative decoding — и как оно сочетается с PP + async

Файлы: `vllm/v1/spec_decode/*`, `vllm/v1/sample/rejection_sampler.py`,
`vllm/v1/worker/pp_spec_broadcast.py`. Кирпичи 30/40/80 — глубокий разбор; данный
раздел — самодостаточная карта.

### 8.1 Таксономия proposer'ов (драфтеров)

`SpeculativeMethod` (`config/speculative.py:~59`) перечисляет методы; runner строит
один proposer:

| Proposer | `file:line` | Источник драфта |
|---|---|---|
| `NgramProposer` | `spec_decode/ngram_proposer.py:~12` | CPU suffix n-gram match по последовательности (без модели). |
| `NgramGPUProposer` | `spec_decode/ngram_proposer_gpu.py:~216` | То же, векторизовано на GPU; поддерживает async. |
| `EagleProposer` | `spec_decode/eagle.py:~10` | Голова EAGLE/EAGLE3; `pass_hidden_states_to_model=True`. |
| **MTP** | через базу Eagle; модели `qwen3_5_mtp.py`, `mimo_mtp.py` | Голова multi-token-prediction; **переиспользует конечное скрытое состояние целевой модели** как вход драфта. |
| `DraftModelProposer` | `spec_decode/draft_model.py:~17` | Отдельная малая LLM; `pass_hidden_states_to_model=False`, словарь/TP должны совпадать. |
| `MedusaProposer`, `SuffixDecodingProposer`, другие | `spec_decode/{medusa,suffix_decoding}.py` | Дополнительные головы / suffix-деревья. |

`propose(...)` (`llm_base_proposer.py:~427`) принимает `target_hidden_states`,
`next_token_ids` и `common_attn_metadata` и возвращает `[batch, num_spec]` драфт-токенов.
**Преимущество MTP**: вход драфта — это скрытое состояние цели, уже находящееся на последнем
ранге — ноль дополнительных вычислений/PCIe (кирпич 30, Q9).

Async-планирование **автоматически включается** для Eagle/MTP/ngram_gpu/draft_model
(`config/vllm.py:~935-979`); CPU ngram/medusa/suffix его отключают.

### 8.2 Rejection sampler (семплер отклонений) = оракул корректности

`rejection_sampler.py`. **Greedy**-ядро (`rejection_greedy_sample_kernel`,
`:708`): `target_argmax = target_logits.argmax(-1)` (`:452`); драфт-токен
**принимается тогда и только тогда, когда совпадает с target argmax**, иначе он
**заменяется на target argmax** и всё после него отклоняется. Следовательно, greedy-вывод —
это **чистая функция от (последовательность target argmax, последовательность драфта)** →
**не зависит от весов**: веса влияют только на *скорость* принятия, но никогда на *равенство*
spec-vs-non-spec. Именно это позволяет dummy/MiMo-весам валидировать каскад (кирпич 70 A3).
Не-greedy путь — это истинное вероятностное rejection sampling (проверка отношения к
вероятностям драфта).

- **Бонусный токен**: если *все* драфты приняты, целевая модель также выдаёт один
  дополнительный «бесплатный» токен в конце принятых. Таким образом, ширина выходной
  сетки — `[num_reqs, num_spec + 1]`.
- **`PLACEHOLDER_TOKEN_ID = -1`** (`rejection_sampler.py:30`): валидные токены идут
  непрерывно начиная со столбца 0; позиции отклонённых/дополнений — `-1`. `parse_output`
  (`:247`) оставляет `!= -1 & < vocab` (`:267`).

### 8.3 Путь spec-под-PP — **код, который мы изменяем** (выделено)

Это подмеханизм, вокруг которого сосредоточены все усилия (кирпичи 40/80/81). При PP>1 +
async:

```
LAST rank:  forward → sample → rejection oracle → sampled[num_reqs, num_spec+1]
            → _pp_broadcast_prev_sampled_token_ids  (GPU broadcast)
            → propose drafts → _draft_token_ids
                    │
                    ▼
NON-LAST:   _pp_receive_… : recv[num_reqs, num_spec+1]; store; rebuild
            prev_req_id_to_index  (AFTER condense)
            → next step _prepare_input_ids:
                 common (prev_pos ≥ 0): scatter recv[:,0] → input_ids   ✓
                 non-common (prev_pos < 0): EARLY RETURN → reads token_ids_cpu
                                            which still holds -1 → EMBED OOB  ✗ break #2
SCHEDULER:  AsyncScheduler reserves -1 placeholders even for requests that
            won't get the worker overwrite  → the root #40768 fixes
```

Транспорт теперь **агностичен по ширине** благодаря новому CUDA-free
`vllm/v1/worker/pp_spec_broadcast.py` (B1a, выполнено):
`broadcast_sampled_token_ids` (`:32`), `receive_sampled_token_ids` (`:42`),
`count_valid_sampled_tokens_per_req` (`:21`). Протестировано на gloo
(`tests/v1/spec_decode/test_pp_spec_broadcast.py`) и валидировано на реальной MiMo-7B
PP=2+MTP (проходит **мимо** старого assert `[num_reqs,1]`). Оставшаяся блокировка —
**break #2**: заглушка `-1`, просачивающаяся в embedding-поиск на не-последнем ранге,
когда запрос попадает в «non-common» — причина в upstream **PR #40768** («stale async
placeholder tokens in spec decode», исправляет #37159), дисциплина на стороне планировщика,
дополняющая B1a. Согласование учёта токенов на не-последнем ранге
(`num_tokens_no_spec` advance, n-gram-гейтированные `:1330/:1490` vs гибридный `:1502`) —
это целостная работа C4 (кирпич 81). Полная хроника — в кирпичах 40 §«Session 5» и 80.

---

## 9. Структуры данных на границах и типизация

Файлы: `vllm/v1/outputs.py`, `vllm/v1/core/sched/output.py`,
`vllm/v1/worker/gpu_input_batch.py`, `vllm/v1/request.py`, `vllm/v1/engine/__init__.py`.
vLLM запускает **mypy в CI**, поэтому статические типы — это «compile-time» контракт vLLM.

**Контракты, пересекающие границу:**

| Структура | `file:line` | Граница |
|---|---|---|
| `EngineCoreRequest` (`msgspec.Struct`) | `engine/__init__.py:~83` | client → core |
| `SchedulerOutput` (`@dataclass`) | `sched/output.py:180` | scheduler → executor |
| `ModelRunnerOutput` (`@dataclass`) | `outputs.py:234` | worker → scheduler |
| `DraftTokenIds` (`@dataclass`) | `outputs.py:311` | proposer → scheduler (отдельно от `ModelRunnerOutput`!) |
| `EngineCoreOutputs` (`msgspec.Struct`) | `engine/__init__.py:~215` | core → client |

**Хорошие идиомы, используемые в коде:** `NamedTuple` (`LogprobsLists`/`LogprobsTensors`,
`outputs.py:27/52`), `TypeAlias` (`PoolerOutput`), `IntEnum` (`RequestStatus`,
`FinishReason`), `Literal` (`PauseMode`), `msgspec.Struct` для IPC, импорты под
`TYPE_CHECKING`.

**Пробелы, где более строгие типы задокументировали бы инварианты** (кирпич 81 + чтение
структур данных):

- Магический **`-1`** — это голый литерал в runner'е/input-batch, несмотря на то что
  `PLACEHOLDER_TOKEN_ID` существует в `rejection_sampler.py:30` → сделать его общим
  `Final[int]` + охранником `is_placeholder()`, используемым повсеместно.
- `sampled_token_ids` / `prev_sampled_token_ids` несут **неявную форму
  `[num_reqs, num_spec+1]` и раскладку `-1`** без обёртки → замороженный
  `SampledTokenGrid` с методом `valid_per_req()` закодирует контракт.
- `scheduled_spec_decode_tokens: dict[str, list[int]]` с **неявным «отсутствующий =
  нет»** → `TypedDict`/замороженная per-request запись.
- голый `req_index: int` для индексирования `token_ids_cpu` → `ReqIndex = NewType('ReqIndex',
  int)`, валидируемый при `add_request`.
- поверхность proposer'а (Eagle/Ngram/MTP/Draft) использует цепочки `isinstance`/`hasattr` →
  `Protocol` `Proposer` (`propose(...) -> DraftTokens`).

**Вводящие в заблуждение имена:** `ModelRunnerOutput` **не** имеет поля `spec_token_ids` —
драфт-токены путешествуют через отдельный `DraftTokenIds`. `SamplerOutput.sampled_token_ids`
(GPU-тензор) дополнен `-1`; `ModelRunnerOutput.sampled_token_ids` (`list[list[int]]`) —
**нет**. Эти два одноимённых поля означают разные вещи по разные стороны границы.

---

## 10. Глоссарий — термин vLLM ↔ общепринятый термин ↔ что это на самом деле

| Термин vLLM | Что предполагают люди | Что это на самом деле | `file:line` |
|---|---|---|---|
| **EngineCore** | «движок» | только **логика** (scheduler+executor+KV); транспорт — отдельный `EngineCoreProc`/`EngineCoreClient`. | `core.py:95` |
| **client / engine split** | сетевой клиент | in-process vs ZMQ-обёрнутый движок; `InprocClient` не имеет *никаких* сокетов. | `core_client.py` |
| **batch_queue** | FIFO-очередь ожидающих батчей | **конвейерное кольцо** in-flight future'ов (`deque(maxlen=pp_size)`); применяется *самый старый*, пока планируется самый новый. ⚠️ не очередь накопления. | `core.py:192-198, 484` |
| **async scheduling** | Python asyncio | **перекрытие scheduler/execution**: планировать шаг N+1 до возврата вывода N, через заглушки `-1`. ⚠️ цикл движка — обычный `while`. | `async_scheduler.py:12`, `config/vllm.py:497` |
| **AsyncLLM** vs **async_scheduling** | одно и то же «async» | **разные вещи**: `AsyncLLM` — asyncio *frontend*; `async_scheduling` — режим *движка* с перекрытием. | `async_llm.py:70` vs `async_scheduler.py:12` |
| **continuous batching** | один большой скользящий батч | объекта батча нет; каждый шаг заново выводит работающее множество из токенного бюджета. | `scheduler.py:336` |
| **chunked prefill** | специальный режим prefill | просто зажатие бюджетом для `num_new_tokens`; промпт потребляется за несколько шагов. | `scheduler.py:~675` |
| **prefix caching** | дедупликация KV | поиск по хешу самого длинного закешированного блочного префикса; **последний токен всегда перевычисляется**. | `kv_cache_manager.py:196,~221` |
| **PagedAttention** | ядро | схема памяти: KV в фиксированных блоках + таблица блоков + slot mapping. | `block_table.py:70,75` |
| **num_computed_tokens** | токены, вычисленные GPU | **счётчик решений планировщика**; откатывается при отклонении, сбрасывается при вытеснении. | `request.py:~149` |
| **num_tokens_with_spec** | «токены, с включённым spec» | оптимистичная длина **при принятии всех драфтов** = `num_tokens + len(spec_token_ids)`. | `request.py:253` |
| **num_output_placeholders** | размер выходного буфера | только для async: количество **обещанных, но не материализованных** выходных токенов. | `request.py:141` |
| **persistent batch / InputBatch** | батч | CPU-буферы, переиспользуемые между шагами для избежания перестройки тензоров. | `gpu_input_batch.py:91` |
| **condense** | сжать данные | **компактировать** плотный префикс запросов после удалений (swap слотов). | `gpu_input_batch.py:683` |
| **GroupCoordinator** | планировщик/лидер | обёртка `ProcessGroup` (device+cpu группы) на каждую ось параллелизма. | `parallel_state.py:290` |
| **IntermediateTensors** | произвольные тензоры | словарь `{hidden_states, residual}`, пересекающий границу PP-стадии. | `sequence.py:~12` |
| **drafter / proposer** | два разных понятия | одно понятие — генератор спекулятивных драфтов. | `llm_base_proposer.py:427` |
| **bonus token** | награда | бесплатный токен целевой модели, выдаваемый при принятии *всех* драфтов. | `rejection_sampler.py:~47` |
| **rejection sampling (greedy)** | вероятностное принятие/отклонение | **детерминированное совпадение с argmax** — не зависит от весов; «rejection sampling» буквально применимо только к не-greedy пути. | `rejection_sampler.py:708,452` |
| **PLACEHOLDER_TOKEN_ID / -1** | реальный токен | сигнал для отклонённых/дополненных позиций; OOB при попадании в embedding-поиск (break #2). | `rejection_sampler.py:30` |
| **uniproc / multiproc executor** | пулы потоков | in-process worker vs один OS-процесс на ранг (ZMQ/MQ). | `uniproc_executor.py:45`, `multiproc_executor.py:103` |
| **V1 vs V2 runner** | версии vLLM | две реализации model-runner'а; V2 уменьшает PP-пузыри, но заблокирован для не-MoE/не-квантованных whitelisted архитектур (то есть *не* наша Qwen3.5). | кирпич 40 |
| **EP / EPLB** | «всё параллельно» | **expert** parallel для MoE (all-to-all); EPLB = отдельная группа для load-balancing коллективов. | `parallel_state.py:~1289,1301` |
| **PCP / DCP** | один «context parallel» | **prefill** vs **decode** context (sequence) parallel; DCP переиспользует TP-GPU (`dcp_size ≤ tp_size`). | `parallel_state.py:~1313,1265` |

---

## 11. Слабые места и «не самые сильные решения»

Честно, со ссылками на код. Severity (серьёзность) = влияние на корректность/поддержку;
Entrenchment (укоренённость) = насколько трудно изменить (количество call-сайтов / степень
центральности). Для каждого указано, исправляет ли это upstream.

| # | Слабое место | `file:line` | Severity × Entrenchment | Как выглядит чистый дизайн | Upstream? |
|---|---|---|---|---|---|
| **W1** | **God-объект `gpu_model_runner.py`** (~7583 строк): forward, attention, sampling, spec, PP, multimodal, pooling, cudagraphs, profiling в одном классе с запутанными ветками `is_last_rank`/`is_ngram_gpu`/`use_async_spec_decode`. | `gpu_model_runner.py:422` (весь файл); узел ветвлений `~1286-1500` | **High × Very high** | Извлечь коллабораторы `SpecDecodeFlow`, `SamplingFlow`, `PPTransport` за типизированными интерфейсами; runner оркестрирует, а не реализует. (Вердикт кирпича 81: владеть *подмеханизмом spec-flow*, не перепис ывать.) | частично — рефактор runner'а V2 (#42187), но недоступен нам |
| **W2** | **Нет владельца конечного автомата запроса**: `request.status = ...` и счётчики токенов (`num_computed_tokens`, `num_output_placeholders`, `spec_token_ids`) мутируются из многих мест планировщика; FSM неявен и хрупок. | `scheduler.py:831, ~960-977, 1446, 1450`; `async_scheduler.py:31,63` | **High × High** | Владелец `RequestState` с охраняемыми переходами + единый метод учёта; утверждение инвариантов (`num_rejected ≤ num_computed_tokens`). (Кирпич 80 §2 «болезни».) | нет |
| **W3** | **Магические заглушки `-1`** + оптимистичное расширение/исправление: `-1` вставляется и планировщиком (`_spec_token_placeholders`), и runner'ом (оптимистичное расширение), заполняется только на общем пути → утечка на non-common/повторно добавленных запросах → **embed OOB (break #2)**. | `async_scheduler.py:~16`; `rejection_sampler.py:30`; `_prepare_input_ids` `gpu_model_runner.py:1708`; кирпич 40 §«Session 5» | **High × High** | Типизированный `SampledTokenGrid` + дисциплина «вставлять `-1` только когда можно заполнить» (req ∈ `prev_step_scheduled_req_ids`). | **да — PR #40768** (исправляет #37159) |
| **W4** | **Два sync/async пути drafte-токенов без общей абстракции**: sync тянет через `post_step`→`take_draft_token_ids`; async инжектирует в worker'е. Sync **дедлочит** для MTP+PP; async **падает**. | `core.py:474` (sync); `gpu_model_runner.py` worker-inject (async); кирпич 40 §«Session 5» | **High × Medium** | Одна абстракция `DraftTokenChannel` с двумя бэкендами за типизированным контрактом (кирпич 81 C2). | частично (#39704 sync, #40768 async) |
| **W5** | **Учёт с гейтом `is_ngram_gpu` оставил MTP позади**: продвижение/коррекция `num_tokens_no_spec` (`:1330/:1490`) гейтировано на ngram_gpu; гибридный MTP использует `_update_states_after_model_execute` (`:1502`) — два подхода не компонуются, поэтому учёт MTP несогласован (Q17). | `gpu_model_runner.py:~1330,~1490,~1502` (модифицированное дерево) | **Medium × Medium** | Учёт принятого количества, агностичный к методу (C4); целостное разрешение асимметрии ngram/hybrid. | нет (наш C4) |
| **W6** | **`prev_req_id_to_index` перестраивается после `condense()`**: персистентный батч переупорядочивается, затем перестраивается карта индексов предыдущего шага — риск устаревшего маппинга под k-шаговой конвейерной задержкой. | `gpu_input_batch.py:683` (condense); перестройка карты `gpu_model_runner.py:~4686` | **Medium × Medium** | `condense()` возвращает remap индексов; потребители используют его, без post-hoc реконструкции. | нет |
| **W7** | **`deque(maxlen=…)` тихо дропает при переполнении**: batch_queue опирается на инвариант pop-before-append; логическое изменение, добавляющее без pop, *тихо* выбросит future и подвесит конвейер без ошибки. | `core.py:198` | **Low × Low** (в настоящее время безопасно по инварианту) | Ограниченная очередь, которая *бросает* исключение при переполнении, или явный assert при добавлении. | нет |
| **W8** | **Нет broadcast-хелпера для спекуляции в coordinator'е**: broadcast сэмплированных токенов написан вручную в runner'е (теперь вынесен в `pp_spec_broadcast.py`), отдельно от `GroupCoordinator`. Нормально для тестируемости, но PP spec transport живёт вне распределённой абстракции. | `parallel_state.py:~637`; `pp_spec_broadcast.py:32` | **Low × Low** | Оставить CUDA-free хелпер (хорошо для gloo-тестов), но зарегистрировать его как метод coordinator'а, чтобы контракт был обнаруживаем. | наш B1a (выполнено) |
| **W9** | **`IntermediateTensors` — нетипизированный `dict[str, Tensor]`**: PP-стадии должны договариваться о ключах (`hidden_states`/`residual`) по соглашению; несовпадение ключей — runtime `KeyError`. | `sequence.py:~12-62` | **Low × Medium** | Замороженный dataclass с типизированными полями. | нет |
| **W10** | **`array_like=True` msgspec structs делают порядок полей wire-контрактом**: добавление/переупорядочивание поля в `EngineCoreRequest`/`EngineCoreOutputs` тихо ломает кросс-версионную совместимость. | `engine/__init__.py:~84,~215` | **Low × Medium** | Версионированная схема или именованное (map) кодирование для критичных к эволюции структур. | нет |

**Топ-5** (по severity×entrenchment): **W1** (god-объект), **W2** (нет владельца FSM),
**W3** (утечка `-1` / break #2), **W4** (два пути драфтов), **W5** (n-gram-гейтированный
учёт MTP). W3 и W4 — это именно то место, где живут усилия spec-под-PP; W3 исправляется
upstream через **#40768**, а наш **B1a** — комплементарная половина на стороне worker'а.

---

## 12. Где наше изменение находится в общей картине

Путь, который мы изменяем, — **поток спекулятивных токенов под PP** — это тонкий, но
несущий шов: последний PP-ранг сэмплирует + запускает rejection oracle, широковещательно
передаёт сетку `[num_reqs, num_spec+1]` ранним рангам, а те реконструируют свой следующий
вход из broadcast + снимка планировщика. Это затрагивает §3 (учёт заглушек в планировщике),
§4 (k-шаговая задержка batch_queue), §5 (методы runner'а
`_update_states`/`_prepare_input_ids`/PP transport), §6 (broadcast, написанный вручную)
и §8 (rejection oracle как шлюз корректности). Изменение с точки зрения дизайна невелико
(Design C: один флаг прямого прохода «standalone draft», кирпич 60); **объём и риск** — в
этой сантехнике корректности (кирпичи 40/80), которая *не зависит от дизайна*. Наши
завершённые автономные результаты — **A1c** (int4-квантование встраивания драфта при
загрузке, memory wall) и **B1a** (broadcast-транспорт, агностичный по ширине) — и
запланированные **C0/C1** (исполняемая спецификация + типизированная модель состояний)
непосредственно укрепляют W3/W4/W5 и пробелы типизации §9. Дуга вклада C0–C5 и
согласование с #40768 — в кирпиче 81.

---

## Приложение — перекрёстные ссылки на кирпичи

| Раздел этого документа | Существующий кирпич (подробнее / источник) |
|---|---|
| §1 модель процессов | — (новое); кирпич 80 §1 engine loop |
| §2 жизненный цикл запроса | — (новое) |
| §3 scheduler / KV | кирпич 80 §2; кирпич 30 (draft KV); кирпич 40 (batch_queue) |
| §4 режимы цикла движка | кирпич 80 §1; кирпич 40 (два пути сантехники) |
| §5 executor / runner | кирпич 80 §3; кирпич 81 (область охвата) |
| §6 distributed / PP | **кирпич 10** (PP и группы), кирпич 20 (embeddings) |
| §7 KV и внимание | кирпич 30, кирпич 60 (draft attn PP), кирпич 70 (hybrid/memory) |
| §8 spec decode | **кирпичи 30/40/80**; кирпич 70 A3 (oracle, не зависящий от весов) |
| §9 типизация | **кирпич 81** |
| §11 слабые места | кирпичи 80 §«болезни», 81 §3 |
| §12 наше изменение | кирпичи 40/60/70/80/81; README + 00-map |
