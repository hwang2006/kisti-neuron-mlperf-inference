# KISTI Neuron Server SUT Patch

This directory contains KISTI-specific copies of the MLCommons Llama 3.1 8B
Server reference implementation.

The original MLCommons source files were preserved unchanged.

## Upstream Reference

Repository:

https://github.com/mlcommons/inference

Version used:

```text
Tag    : v6.0.0pre
Commit : 7f42a83e543660fd4699f1f85a05ef06b4dc334a
```

Original files:

```text
language/llama3.1-8b/SUT_VLLM.py
language/llama3.1-8b/main.py
```

## KISTI-specific Files

```text
SUT_VLLM_serverfix.py
main_serverfix.py
```

## Why the Patch Was Needed

The original Server implementation uses vLLM `AsyncLLMEngine`.

In the tested KISTI Neuron environment, each Server query was processed using
a separate `asyncio.run()` invocation.

Conceptually, the reference behavior was:

```text
request 1
  -> asyncio.run()
  -> create event loop
  -> process request
  -> destroy event loop

request 2
  -> asyncio.run()
  -> create another event loop
  -> process request
  -> destroy event loop
```

With vLLM `AsyncLLMEngine`, this caused the asynchronous engine lifecycle to
be disrupted after the first request, and the Server benchmark stalled.

## Persistent Event Loop

`SUT_VLLM_serverfix.py` creates one persistent asyncio event loop in the
Server worker thread.

The same loop is reused for subsequent requests:

```text
Server worker
  -> create persistent event loop
       -> request 1
       -> request 2
       -> request 3
       -> ...
  -> close event loop
```

This allowed Server requests to be processed continuously.

## Shutdown Handling

The tested reference shutdown path also attempted to access:

```text
ft_response_thread
```

although this attribute was not created in the tested execution path.

The KISTI-specific copy therefore protects this shutdown path with an
attribute-existence check.

Conceptually:

```python
if hasattr(self, "ft_response_thread"):
    ...
```

## main_serverfix.py

`main_serverfix.py` is a copy of the reference `main.py`.

Its purpose is to import the KISTI-specific Server implementation instead of
the original `SUT_VLLM.py`.

The original MLCommons `main.py` remains unchanged.

## Remaining Warning

Some successful Server runs still produced asynchronous cleanup warnings at
process shutdown, such as:

```text
Task was destroyed but it is pending!
```

and multiprocessing shared-memory cleanup warnings.

These occurred after LoadGen completed and did not prevent benchmark runs
from receiving a VALID result.

However, the asynchronous engine shutdown lifecycle should be fully resolved
before treating this implementation as submission-quality code.

## Scope

These files are provided for:

- reproducibility
- portability analysis
- performance engineering
- comparison with the MLCommons reference implementation

They should not be interpreted as an official MLCommons Server submission
implementation without further compliance review.
