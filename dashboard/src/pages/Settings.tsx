import { useState, useEffect, useCallback } from 'react';
import { PresetSelector } from '../components/PresetSelector.tsx';
import { useApi } from '../hooks/useApi.ts';
import { api } from '../api/client.ts';
import styles from './Settings.module.css';

const DEFAULTS = {
  preset: 'balanced',
  freq: '1h',
  horizon: '4h',
  tau: 0.005,
  confidence: 70,
};

export function Settings() {
  const settings = useApi(() => api.getSettings(), []);

  const [preset, setPreset] = useState(DEFAULTS.preset);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [freq, setFreq] = useState(DEFAULTS.freq);
  const [horizon, setHorizon] = useState(DEFAULTS.horizon);
  const [tau, setTau] = useState(DEFAULTS.tau);
  const [confidence, setConfidence] = useState(DEFAULTS.confidence);
  const [saving, setSaving] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  // Populate from loaded settings
  useEffect(() => {
    if (settings.data) {
      setPreset(settings.data.preset || DEFAULTS.preset);
      setFreq(settings.data.freq || DEFAULTS.freq);
      setHorizon(settings.data.horizon || DEFAULTS.horizon);
      setTau(settings.data.tau ?? DEFAULTS.tau);
      setConfidence(
        settings.data.enter_threshold != null
          ? Math.round(settings.data.enter_threshold * 100)
          : DEFAULTS.confidence,
      );
    }
  }, [settings.data]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setStatusMsg('');
    setErrorMsg('');

    try {
      await api.updateSettings({
        preset,
        freq: showAdvanced ? freq : null,
        horizon: showAdvanced ? horizon : null,
        tau: showAdvanced ? tau : null,
        enter_threshold: showAdvanced ? confidence / 100 : null,
      });
      setStatusMsg('Settings saved successfully.');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }, [preset, freq, horizon, tau, confidence, showAdvanced]);

  const handleReset = useCallback(() => {
    setPreset(DEFAULTS.preset);
    setFreq(DEFAULTS.freq);
    setHorizon(DEFAULTS.horizon);
    setTau(DEFAULTS.tau);
    setConfidence(DEFAULTS.confidence);
    setStatusMsg('');
    setErrorMsg('');
  }, []);

  return (
    <div className={styles.page}>
      <h2>Settings</h2>

      <PresetSelector selected={preset} onSelect={setPreset} />

      <div className="divider">&#x2726;</div>

      <button
        className={styles.toggleLink}
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
      </button>

      {showAdvanced && (
        <div className={styles.form}>
          <div className={styles.field}>
            <label className={styles.fieldLabel}>Frequency</label>
            <select
              className={styles.select}
              value={freq}
              onChange={(e) => setFreq(e.target.value)}
            >
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
          </div>

          <div className={styles.field}>
            <label className={styles.fieldLabel}>Horizon</label>
            <select
              className={styles.select}
              value={horizon}
              onChange={(e) => setHorizon(e.target.value)}
            >
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="24h">24h</option>
            </select>
          </div>

          <div className={styles.field}>
            <label className={styles.fieldLabel}>
              Sensitivity (tau): {tau.toFixed(3)}
            </label>
            <input
              type="range"
              className={styles.slider}
              min={0.001}
              max={0.05}
              step={0.001}
              value={tau}
              onChange={(e) => setTau(parseFloat(e.target.value))}
            />
          </div>

          <div className={styles.field}>
            <label className={styles.fieldLabel}>
              Confidence: {confidence}%
            </label>
            <input
              type="range"
              className={styles.slider}
              min={50}
              max={95}
              step={5}
              value={confidence}
              onChange={(e) => setConfidence(parseInt(e.target.value, 10))}
            />
          </div>
        </div>
      )}

      <div className={styles.actions}>
        <button
          className={styles.saveBtn}
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save & Apply'}
        </button>
        <button className={styles.resetBtn} onClick={handleReset}>
          Reset to Default
        </button>
      </div>

      {statusMsg && <p className={styles.statusMsg}>{statusMsg}</p>}
      {errorMsg && <p className={styles.errorMsg}>{errorMsg}</p>}
    </div>
  );
}
