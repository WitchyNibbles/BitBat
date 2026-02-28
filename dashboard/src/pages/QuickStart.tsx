import { useState, useCallback, useRef, useEffect } from 'react';
import { PresetSelector } from '../components/PresetSelector.tsx';
import { TimelineChart } from '../components/TimelineChart.tsx';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import { api } from '../api/client.ts';
import styles from './QuickStart.module.css';

type Phase = 'choose' | 'training' | 'running';

const PROGRESS_STEPS = [
  { pct: 10, msg: 'Validating configuration...' },
  { pct: 30, msg: 'Preparing dataset...' },
  { pct: 55, msg: 'Training model...' },
  { pct: 75, msg: 'Running walk-forward CV...' },
  { pct: 90, msg: 'Finalizing...' },
];

export function QuickStart() {
  const [phase, setPhase] = useState<Phase>('choose');
  const [preset, setPreset] = useState('balanced');
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const abortRef = useRef(false);

  const history = useApi(
    () => api.predictionHistory('1h', '4h', 7, 50),
    [],
  );

  usePolling(
    () => {
      history.refetch();
    },
    60_000,
    phase === 'running',
  );

  // Cleanup abort ref on unmount
  useEffect(() => {
    return () => {
      abortRef.current = true;
    };
  }, []);

  const startTraining = useCallback(async () => {
    abortRef.current = false;
    setPhase('training');
    setProgress(0);
    setStatusMsg('Initializing...');
    setErrorMsg('');

    try {
      // Simulate progress steps
      for (const step of PROGRESS_STEPS) {
        if (abortRef.current) return;
        await new Promise((r) => setTimeout(r, 800));
        if (abortRef.current) return;
        setProgress(step.pct);
        setStatusMsg(step.msg);
      }

      // Actual API call
      const result = await api.startTraining(preset);

      if (abortRef.current) return;

      if (result.status === 'error') {
        setErrorMsg(result.error ?? 'Training failed');
        setPhase('choose');
        return;
      }

      setProgress(100);
      setStatusMsg('Training complete!');
      await new Promise((r) => setTimeout(r, 600));
      if (abortRef.current) return;
      setPhase('running');
      history.refetch();
    } catch (err) {
      if (abortRef.current) return;
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setPhase('choose');
    }
  }, [preset, history]);

  if (phase === 'choose') {
    return (
      <div className={styles.page}>
        <h2>Quick Start</h2>
        <p className={styles.intro}>
          Choose a prediction preset and begin training. The model will be
          trained using walk-forward cross-validation on historical BTC data.
        </p>

        <PresetSelector selected={preset} onSelect={setPreset} />

        {errorMsg && <p className={styles.errorMsg}>{errorMsg}</p>}

        <button className={styles.trainBtn} onClick={startTraining}>
          Begin Training
        </button>
      </div>
    );
  }

  if (phase === 'training') {
    return (
      <div className={styles.page}>
        <h2>Training in Progress</h2>
        <div className={styles.progressBar}>
          <div
            className={styles.progressFill}
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className={styles.statusMsg}>{statusMsg}</p>
      </div>
    );
  }

  // phase === 'running'
  return (
    <div className={styles.page}>
      <div className={styles.controls}>
        <span className={styles.activeLabel}>Model active</span>
        <button
          className={styles.retrain}
          onClick={() => {
            setPhase('choose');
            setProgress(0);
            setStatusMsg('');
            setErrorMsg('');
          }}
        >
          Retrain
        </button>
      </div>

      <div className="divider">&#x2726;</div>

      {history.data && history.data.predictions.length > 0 ? (
        <TimelineChart predictions={history.data.predictions} />
      ) : (
        <p className={styles.statusMsg}>No predictions yet. Data will appear after the first inference cycle.</p>
      )}
    </div>
  );
}
