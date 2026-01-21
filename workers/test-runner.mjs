import { run } from 'node:test';

import './__tests__/main.test.js';

const runner = run();
let failed = false;

runner.on('test:fail', () => {
  failed = true;
});

runner.on('error', () => {
  failed = true;
});

runner.on('end', () => {
  process.exitCode = failed ? 1 : 0;
});
