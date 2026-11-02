// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

const factories = eval(readFileSync(new URL('../src/callback_factories.js', import.meta.url), 'utf8'));

function neverSettles() {
  return new Promise(() => {});
}

function invoke(wrapper, callId, { abort = false, streamResult = false } = {}) {
  wrapper(null, abort, callId, { value: 1 }, false, undefined, false, undefined, undefined, streamResult);
}

describe('promise callback cancellation', () => {
  it('drops a non-stream callback that ignores cancellation and never settles', async () => {
    let signal;
    const wrapper = factories.promise(
      (_value, nextSignal) => {
        signal = nextSignal;
        return neverSettles();
      },
      () => {
        throw new Error('cancelled callback settled');
      },
      () => {
        throw new Error('cancelled callback rejected');
      },
      () => {},
      () => {},
      () => {},
    );

    invoke(wrapper, 1);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(wrapper.pendingCalls(), 1);
    assert.equal(signal.aborted, false);

    invoke(wrapper, 1, { abort: true });
    assert.equal(signal.aborted, true);
    assert.equal(wrapper.pendingCalls(), 0);

    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(wrapper.pendingCalls(), 0);
  });

  it('does not register a non-stream callback cancelled before it starts', async () => {
    let started = false;
    const wrapper = factories.promise(
      () => {
        started = true;
        return neverSettles();
      },
      () => {},
      () => {},
      () => {},
      () => {},
      () => {},
    );

    invoke(wrapper, 2, { abort: true });
    invoke(wrapper, 2);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(started, false);
    assert.equal(wrapper.pendingCalls(), 0);
  });

  it('keeps a stream callback registered when cancellation starts', () => {
    const wrapper = factories.promise(
      neverSettles,
      () => {},
      () => {},
      () => {},
      () => {},
      () => {},
    );

    invoke(wrapper, 3, { streamResult: true });
    assert.equal(wrapper.pendingCalls(), 1);
    invoke(wrapper, 3, { abort: true, streamResult: true });
    assert.equal(wrapper.pendingCalls(), 1);
  });
});
