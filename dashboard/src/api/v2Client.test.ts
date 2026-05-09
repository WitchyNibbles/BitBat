import {
  resolveV2OperatorToken,
  setV2OperatorToken,
} from './v2Client';

describe('v2Client operator token resolution', () => {
  it('uses the local demo token on localhost when no explicit token is configured', () => {
    window.sessionStorage.removeItem('bitbat.v2OperatorToken');

    expect(['localhost', '127.0.0.1']).toContain(window.location.hostname);

    expect(resolveV2OperatorToken()).toBe('bitbat-local-dev-token');
  });

  it('prefers a session-scoped override token when present', () => {
    setV2OperatorToken('custom-token');

    expect(resolveV2OperatorToken()).toBe('custom-token');

    setV2OperatorToken('');
  });
});
